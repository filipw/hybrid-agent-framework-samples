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

class CoAState(BaseModel):
    query: str = ""
    chunks: List[str] = Field(default_factory=list)
    current_idx: int = 0
    accumulated_message: str = "No relevant information found yet."

def ensure_stateless(msgs):
    """Ensures we don't build up long conversation histories for stateless node."""
    return [msgs[-1]]

class WorkerExecutor(Executor):
    """Local worker that reads a chunk and passes updated findings down the chain."""
    
    def __init__(self, name: str, client: MLXChatClient, state: CoAState):
        super().__init__(id=name)
        self.client = client
        self.state = state

    @handler
    async def process_chunk(self, message: str, ctx: WorkflowContext[str]):
        if self.state.current_idx >= len(self.state.chunks):
            await ctx.send_message(self.state.accumulated_message)
            return
            
        chunk = self.state.chunks[self.state.current_idx]
        prev_msg = self.state.accumulated_message
        
        prompt = (
            f"User Query: {self.state.query}\n\n"
            f"Previous Worker's Findings:\n{prev_msg}\n\n"
            f"Your Assigned Text Chunk:\n---\n{chunk}\n---\n\n"
            "Task: You are a Worker Agent in a Chain of Agents. Read your assigned text chunk. "
            "If it contains information relevant to the user query, integrate it with the Previous Worker's Findings to create an updated, comprehensive summary of events. "
            "If it contains no relevant information, output exactly the Previous Worker's Findings. "
            "Output ONLY the updated findings summary without conversational filler."
        )
        
        response = await self.client.get_response([ChatMessage(role=Role.USER, text=prompt)])
        output_text = response.messages[-1].text.strip()
        
        print(f"\n   [Worker {self.state.current_idx + 1}/{len(self.state.chunks)}] Chunk processed. Extracted Info Length: {len(output_text)} chars")
        
        self.state.accumulated_message = output_text
        self.state.current_idx += 1
        
        if self.state.current_idx < len(self.state.chunks):
            # Pass control back to this worker pool for the next chunk
            await ctx.send_message("CONTINUE_CHAIN")
        else:
            # Chain is done, pass control to Manager with the accumulated message
            await ctx.send_message(self.state.accumulated_message)

def is_chain_complete(msg) -> bool:
    # Just need to check that it isn't CONTINUE_CHAIN to move to complete since worker
    # now outputs pure text instead of a control signal
    text = msg if isinstance(msg, str) else (msg.agent_response.text or "")
    if text == "CONTINUE_CHAIN":
        return False
    return self_check_continue(text) == False

def self_check_continue(text):
    return text == "CONTINUE_CHAIN"

def is_chain_continue(msg) -> bool:
    if isinstance(msg, str): return msg == "CONTINUE_CHAIN"
    return isinstance(msg, AgentExecutorResponse) and "CONTINUE_CHAIN" in (msg.agent_response.text or "")



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

    state = CoAState(query=query, chunks=document_chunks)
    
    # 1. Local MLX client for the Workers
    mlx_config = MLXGenerationConfig(max_tokens=300, temp=0.1)
    mlx_client = MLXChatClient(
        model_path="mlx-community/Phi-4-mini-instruct-4bit",
        generation_config=mlx_config,
        message_preprocessor=ensure_stateless
    )
    
    chain_worker = WorkerExecutor(name="Local_Worker_Chain", client=mlx_client, state=state)
    
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
        
        # 3. Build the graph logic
        builder = WorkflowBuilder()
        builder.set_start_executor(chain_worker)
        
        # Chain loop
        builder.add_edge(
            source=chain_worker, 
            target=chain_worker, 
            condition=is_chain_continue
        )
        # End of chain, route to manager
        builder.add_edge(
            source=chain_worker, 
            target=manager, 
            condition=is_chain_complete
        )
        
        workflow = builder.build()
        
        print("🚀 Starting Chain...\n")
        current_agent = None
        
        # Run workflow 
        async for event in workflow.run_stream("START"):
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