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
# This pattern implements a sequential 'Bucket Brigade'. A long context is 
# chunked, and a low-cost local Worker (SLM) sequentially extracts relevant 
# information, passing its state forward. Finally, a Cloud Manager (LLM) 
# synthesizes the accumulated state into a final answer.
# -------------------------------------------------------------------------

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("agent_framework").setLevel(logging.ERROR)

load_dotenv()

def ensure_stateless(msgs):
    """Ensures we don't build up long conversation histories for stateless node."""
    return [msgs[-1]]

class WorkerExecutor(Executor):
    """Local worker that reads a chunk and passes updated findings down the chain."""
    
    def __init__(self, name: str, client: MLXChatClient, query: str, chunk: str, worker_idx: int, total_workers: int):
        super().__init__(id=name)
        self.client = client
        self.query = query
        self.chunk = chunk
        self.worker_idx = worker_idx
        self.total_workers = total_workers

    @handler
    async def process_chunk(self, message: str, ctx: WorkflowContext[str]):
        # 'message' contains the context accumulated by the previous agent in the chain
        # When receiving an AgentExecutorResponse, extract the text payload
        if hasattr(message, "agent_response") and hasattr(message.agent_response, "text"):
            message = message.agent_response.text or ""
            
        prompt = (
            f"User Query: {self.query}\n\n"
            f"Previous Worker's Findings:\n{message}\n\n"
            f"Your Assigned Text Chunk:\n---\n{self.chunk}\n---\n\n"
            "Task: You are a Worker Agent in a Chain of Agents. Read your assigned text chunk. "
            "If it contains information relevant to the user query, integrate it with the Previous Worker's Findings to create an updated, comprehensive summary of events. "
            "If it contains no relevant information, output exactly the Previous Worker's Findings. "
            "Output ONLY the updated findings summary without conversational filler."
        )
        
        response = await self.client.get_response([ChatMessage(role=Role.USER, text=prompt)])
        output_text = response.messages[-1].text.strip()
        
        print(f"\n   [{self.id} ({self.worker_idx}/{self.total_workers})] Chunk processed. Extracted Info Length: {len(output_text)} chars")
        
        # Pass the extracted information as a message to the next agent in the sequence
        await ctx.send_message(output_text)

async def main():
    print("===============================================================")
    print("   Chain of Agents (CoA) Pattern (arXiv:2406.02818)")
    print("===============================================================\n")

    # Load and chunk the "long" document
    text_file_path = os.path.join(os.path.dirname(__file__), "security_logs.txt")
    with open(text_file_path, "r", encoding="utf-8") as f:
        full_text = f.read()
        
    # Naive chunking (e.g. by newlines or blocks) just for the demo
    lines = full_text.strip().split("\n")
    # Group lines into pairs to simulate larger chunks
    document_chunks = ["\n".join(lines[i:i+3]) for i in range(0, len(lines), 3)]

    query = "Create a brief chronological timeline of the ransomware attack and its resolution. Include the root cause."
    
    print(f"❔ Query: {query}")
    print(f"📄 Document split into {len(document_chunks)} sequential chunks.\n")
    
    # 1. Local MLX client for the Workers
    mlx_config = MLXGenerationConfig(max_tokens=300, temp=0.1)
    mlx_client = MLXChatClient(
        model_path="mlx-community/Phi-4-mini-instruct-4bit",
        generation_config=mlx_config,
        message_preprocessor=ensure_stateless
    )
    
    # 2. Cloud Agent for the final Manager/Refiner
    async with AzureCliCredential() as credential:
        azure_client = AzureAIAgentClient(credential=credential)
        manager = azure_client.as_agent(
            name="Cloud_Manager",
            instructions=(
                "You are the Manager Agent in a Chain of Agents workflow. "
                "You will receive the final accumulated findings from a chain of local worker agents that scanned a large context. "
                "Synthesize this information to provide a clear, final answer to the user's query."
            )
        )
        
        # 3. Create distinct worker agents and build the chain logic
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
        
        # Link workers sequentially (Worker 1 -> Worker 2 -> ... -> Worker N)
        for i in range(len(workers) - 1):
            builder.add_edge(source=workers[i], target=workers[i+1])
            
        # End of chain: route the final worker directly to the manager
        builder.add_edge(source=workers[-1], target=manager)
        
        workflow = builder.build()
        
        print("🚀 Starting Chain...\n")
        current_agent = None
        
        # Run workflow passing the initial context string
        initial_findings = "No relevant information found yet."
        async for event in workflow.run_stream(initial_findings):
            if isinstance(event, AgentRunUpdateEvent):
                # We only want to stream out the Manager's final answer to the user
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