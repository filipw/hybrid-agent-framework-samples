import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

from agent_framework import (
    WorkflowBuilder,
    WorkflowContext,
    Executor,
    handler,
    Message,
)
from agent_framework_foundry import FoundryChatClient
from azure.identity.aio import AzureCliCredential

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from local_models import create_local_client, LocalGenerationConfig

# -------------------------------------------------------------------------
# Based on: "Chain of Agents: Large Language Models Collaborating on 
# Long-Context Tasks" (Wang et al., arXiv:2406.02818)
#
# CoA splits a long document into chunks and assigns each to a Worker agent.
# Workers process chunks sequentially, each receiving the previous worker's
# "Communication Unit" (CU) - a running summary of findings so far - and
# outputting an updated CU that incorporates its own chunk.  A Manager agent
# receives the final CU and synthesizes it into the answer.
#
# This demo uses a local SLM (Phi-4-mini, 8-bit) for the Workers and a
# cloud LLM for the Manager, demonstrating a hybrid local/remote pattern.
# -------------------------------------------------------------------------

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("agent_framework").setLevel(logging.ERROR)
load_dotenv()

# Cap on the Communication Unit length passed between workers.  Prevents
# the CU from growing unboundedly and crowding out the new chunk in the
# SLM's context window.  When exceeded, the tail is kept (most recent
# findings carry the fullest accumulated context).
MAX_CU_CHARS = 1500


def ensure_stateless(msgs):
    """Ensures we don't build up long conversation histories for stateless nodes."""
    return [msgs[-1]]


def truncate_cu(cu: str, limit: int = MAX_CU_CHARS) -> str:
    """Keep the CU within budget so the SLM can attend to the new chunk."""
    if len(cu) <= limit:
        return cu
    return "..." + cu[-(limit - 3):]


class WorkerExecutor(Executor):
    """Worker agent (Stage 1 of CoA): reads a chunk, updates the CU, passes it on."""
    
    def __init__(self, name: str, client, query: str, chunk: str, worker_idx: int, total_workers: int):
        super().__init__(id=name)
        self.client = client
        self.query = query
        self.chunk = chunk
        self.worker_idx = worker_idx
        self.total_workers = total_workers

    @handler
    async def process_chunk(self, message: str, ctx: WorkflowContext[str]):
        previous_cu = truncate_cu(message.strip())
        
        # Handle the first worker (no previous CU yet - paper: CU_0 = empty)
        if previous_cu:
            cu_section = f"Here is the summary of the previous source text: {previous_cu}"
        else:
            cu_section = "There is no previous summary yet - this is the first chunk."

        # Worker prompt - directly from the paper's query-based template (Table 9).
        # The query is shown as future context so the worker extracts all facts,
        # not just those that appear relevant at this stage.
        prompt = (
            f"{self.chunk}\n\n"
            f"{cu_section}\n\n"
            f"Question that will be answered later: {self.query}\n\n"
            "You need to read the current source text and the summary of the previous source text "
            "(if any) and generate a summary to include them both. "
            "Later, this summary will be used for other agents to answer the question. "
            "So please write the summary that can include the evidence for answering the question. "
            "Do NOT invent or infer anything not explicitly stated in the source text or previous summary. "
            "Output only the updated factual summary, 3-5 sentences, no commentary."
        )
        
        response = await self.client.get_response([Message("user", [prompt])])
        output_text = response.messages[-1].text.strip()
        
        print(f"\n   [{self.id} ({self.worker_idx}/{self.total_workers})] Chunk processed. CU length: {len(output_text)} chars")
        print(f"   {'-'*60}\n   {output_text}\n   {'-'*60}")
        
        # Pass the updated CU to the next agent in the chain
        await ctx.send_message(output_text)


class ManagerExecutor(Executor):
    """Manager agent (Stage 2 of CoA): receives the final CU and synthesizes the answer."""

    def __init__(self, name: str, client, query: str):
        super().__init__(id=name)
        self.client = client
        self.query = query

    @handler
    async def synthesize(self, message: str, ctx: WorkflowContext[str]):
        final_cu = message.strip()

        # CU + question + "Answer:" nudge for the manager
        prompt = (
            "The following are given passages. However, the source text is too long "
            "and has been summarized. You need to answer based on the summary:\n\n"
            f"{final_cu}\n\n"
            f"Question: {self.query}\n\n"
            "Answer:"
        )

        print(f"\n\n   ☁️  {self.id}:\n   ", end="", flush=True)
        response = await self.client.get_response([Message("user", [prompt])])
        print(response.messages[-1].text.strip())


async def main():
    print("===============================================================")
    print("   Chain of Agents (CoA) Pattern (arXiv:2406.02818)")
    print("===============================================================\n")

    # Load and chunk the document
    text_file_path = os.path.join(os.path.dirname(__file__), "quantum_mechanics_history.txt")
    with open(text_file_path, "r", encoding="utf-8") as f:
        full_text = f.read()
        
    # Split into paragraphs - one worker per pair of paragraphs
    lines = [l for l in full_text.strip().split("\n") if l.strip()]
    chunk_size = 2
    document_chunks = ["\n".join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]

    query = "How did quantum mechanics evolve from Planck's initial hypothesis to a complete mathematical framework? Trace the key contributors and what each one added."
    
    print(f"❔ Query: {query}")
    print(f"📄 Document split into {len(document_chunks)} sequential chunks.\n")
    
    # 1. Local SLM for the Workers (Stage 1)
    local_config = LocalGenerationConfig(max_tokens=250, temp=0.1, repetition_penalty=1.15)
    local_client = create_local_client(
        model_path=os.environ.get("LOCAL_MODEL_PATH", "phi-4-8bit"),
        generation_config=local_config,
        message_preprocessor=ensure_stateless,
    )
    
    # 2. Cloud LLM for the Manager (Stage 2)
    async with AzureCliCredential() as credential:
        manager = ManagerExecutor(name="Cloud_Manager", client=FoundryChatClient(project_endpoint=os.environ.get("AZURE_AI_PROJECT_ENDPOINT"), model=os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME"), credential=credential), query=query)
        
        # 3. Build the sequential worker chain → manager
        workers = []
        for i, chunk in enumerate(document_chunks):
            worker = WorkerExecutor(
                name=f"Worker_{i+1}", 
                client=local_client,
                query=query, 
                chunk=chunk,
                worker_idx=i+1,
                total_workers=len(document_chunks)
            )
            workers.append(worker)

        builder = WorkflowBuilder(start_executor=workers[0])
        
        for i in range(len(workers) - 1):
            builder.add_edge(source=workers[i], target=workers[i+1])
            
        builder.add_edge(source=workers[-1], target=manager)
        
        workflow = builder.build()
        
        print("🚀 Starting Chain...\n")

        # Kick off with an empty CU (paper Algorithm 1: CU_0 ← empty string)
        async for _ in workflow.run("", stream=True):
            pass

        print("\n\n✅ Workflow Complete.")
        

if __name__ == "__main__":
    asyncio.run(main())