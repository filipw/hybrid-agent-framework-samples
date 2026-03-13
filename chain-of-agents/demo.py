import os
import asyncio
import logging
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agent_framework_mlx import MLXChatClient, MLXGenerationConfig
from agent_framework import (
    WorkflowBuilder,
    WorkflowContext,
    Executor,
    handler,
    Role,
    AgentRunUpdateEvent,
    AgentExecutorResponse,
    ChatMessage
)
from agent_framework_azure_ai import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential

# -------------------------------------------------------------------------
# Based on: "Chain of Agents: Large Language Models Collaborating on 
# Long-Context Tasks" (Wang et al., arXiv:2406.02818)
#
# CoA splits a long document into chunks and assigns each to a Worker agent.
# Workers process chunks sequentially, each receiving the previous worker's
# "Communication Unit" (CU) — a running summary of findings so far — and
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
    
    def __init__(self, name: str, client: MLXChatClient, query: str, chunk: str, worker_idx: int, total_workers: int):
        super().__init__(id=name)
        self.client = client
        self.query = query
        self.chunk = chunk
        self.worker_idx = worker_idx
        self.total_workers = total_workers

    @handler
    async def process_chunk(self, message: str, ctx: WorkflowContext[str]):
        # Extract the CU text from the previous worker's output
        if hasattr(message, "agent_response") and hasattr(message.agent_response, "text"):
            message = message.agent_response.text or ""

        previous_cu = truncate_cu(message.strip())
        
        # Handle the first worker (no previous CU yet — paper: CU_0 = empty)
        if previous_cu:
            cu_section = f"Here is the summary of the previous source text: {previous_cu}"
        else:
            cu_section = "There is no previous summary yet — this is the first chunk."

        # Worker prompt — adapted from the paper's query-based template (Table 5).
        # The query is presented as context ("will be answered later") so the
        # worker focuses on extracting all facts rather than filtering by keyword.
        prompt = (
            f"{self.chunk}\n\n"
            f"{cu_section}\n\n"
            f"Question that will be answered later: {self.query}\n\n"
            "Summarize ALL events from the current source text together with the previous summary. "
            "Include every event with its timestamp and details — do not skip events even if they "
            "seem unrelated to the question. Do NOT invent or infer any events that are not "
            "explicitly stated in the source text or previous summary. "
            "Another agent will use your summary to answer the question. "
            "Output only the updated factual summary, 3-5 sentences, no commentary."
        )
        
        response = await self.client.get_response([ChatMessage(role=Role.USER, text=prompt)])
        output_text = response.messages[-1].text.strip()
        
        print(f"\n   [{self.id} ({self.worker_idx}/{self.total_workers})] Chunk processed. CU length: {len(output_text)} chars")
        print(f"   {'-'*60}\n   {output_text}\n   {'-'*60}")
        
        # Pass the updated CU to the next agent in the chain
        await ctx.send_message(output_text)


async def main():
    print("===============================================================")
    print("   Chain of Agents (CoA) Pattern (arXiv:2406.02818)")
    print("===============================================================\n")

    # Load and chunk the document
    text_file_path = os.path.join(os.path.dirname(__file__), "security_logs.txt")
    with open(text_file_path, "r", encoding="utf-8") as f:
        full_text = f.read()
        
    # Split into small chunks (2 lines each) — one worker per chunk
    lines = [l for l in full_text.strip().split("\n") if l.strip()]
    chunk_size = 2
    document_chunks = ["\n".join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]

    query = "Create a brief chronological timeline of the ransomware attack and its resolution. Include the root cause."
    
    print(f"❔ Query: {query}")
    print(f"📄 Document split into {len(document_chunks)} sequential chunks.\n")
    
    # 1. Local SLM for the Workers (Stage 1)
    mlx_config = MLXGenerationConfig(max_tokens=400, temp=0.1, repetition_penalty=1.15)
    mlx_client = MLXChatClient(
        model_path="mlx-community/Phi-4-mini-instruct-8bit",
        generation_config=mlx_config,
        message_preprocessor=ensure_stateless
    )
    
    # 2. Cloud LLM for the Manager (Stage 2)
    async with AzureCliCredential() as credential:
        azure_client = AzureAIAgentClient(credential=credential)
        manager = azure_client.as_agent(
            name="Cloud_Manager",
            instructions=(
                "You are the Manager Agent in a Chain of Agents workflow. "
                "The source text was too long, so worker agents read it in chunks and produced "
                "the summary you will receive. Treat this summary as your source material — "
                "do not critique it or point out inconsistencies in it. "
                "Use it to directly answer the following question.\n\n"
                f"Question: {query}"
            )
        )
        
        # 3. Build the sequential worker chain → manager
        builder = WorkflowBuilder()
        
        workers = []
        for i, chunk in enumerate(document_chunks):
            worker = WorkerExecutor(
                name=f"Worker_{i+1}", 
                client=mlx_client, 
                query=query, 
                chunk=chunk,
                worker_idx=i+1,
                total_workers=len(document_chunks)
            )
            workers.append(worker)
            
        builder.set_start_executor(workers[0])
        
        for i in range(len(workers) - 1):
            builder.add_edge(source=workers[i], target=workers[i+1])
            
        builder.add_edge(source=workers[-1], target=manager)
        
        workflow = builder.build()
        
        print("🚀 Starting Chain...\n")
        current_agent = None
        
        # Kick off with an empty CU (paper Algorithm 1: CU_0 ← empty string)
        async for event in workflow.run_stream(""):
            if isinstance(event, AgentRunUpdateEvent):
                # Stream only the Manager's final answer to the console
                if event.executor_id == "Cloud_Manager":
                    if current_agent != "Cloud_Manager":
                        current_agent = "Cloud_Manager"
                        print(f"\n\n   ☁️  {current_agent}: \n   ", end="", flush=True)
                    if event.data and event.data.text:
                        print(event.data.text, end="", flush=True)
                        
        print("\n\n✅ Workflow Complete.")
        
        await azure_client.close()

if __name__ == "__main__":
    asyncio.run(main())