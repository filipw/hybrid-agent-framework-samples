import asyncio
import os
import sys
from typing import Literal
from agent_framework import (
    Agent,
    WorkflowBuilder,
    WorkflowEvent,
    AgentResponseUpdate,
    AgentExecutorResponse,
    AgentResponse,
    Executor,
    handler,
    WorkflowContext,
    Message,
    BaseChatClient,
)
from agent_framework_foundry import FoundryChatClient
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from local_models import create_local_client, LocalGenerationConfig

load_dotenv()

# 1. define the Routing Logic
# using a few-shot prompt for classification
ROUTER_INSTRUCTIONS = """
You are a high-precision query classifier. 
Your ONLY job is to route the user's query to the appropriate model.
- 'ROUTE: WEAK': For simple facts, formatting, summaries, or questions with obvious answers.
- 'ROUTE: STRONG': For reasoning, coding, creative writing, analysis, or complex multi-step tasks.

EXAMPLES:
Input: "What is the capital of France?"
Output: ROUTE: WEAK

Input: "Write a Python script to parse a CSV and plot the data."
Output: ROUTE: STRONG

Input: "Summarize this short text."
Output: ROUTE: WEAK

Input: "Explain the mathematical difference between matrix mechanics and wave mechanics in quantum mechanics."
Output: ROUTE: STRONG

You must output ONLY 'ROUTE: WEAK' or 'ROUTE: STRONG'. Do not answer the user query.
"""

class ValidationState:
    route: Literal["WEAK", "STRONG"] = "WEAK"

class RouterExecutor(Executor):
    def __init__(self, client: BaseChatClient, state: ValidationState):
        super().__init__(id="Router_Control_Plane")
        self.client = client
        self.state = state

    @handler
    async def route_query(self, query: str, ctx: WorkflowContext[AgentExecutorResponse]):
        msgs = [
            Message("system", [ROUTER_INSTRUCTIONS]),
            Message("user", [f"Input: \"{query}\"\nOutput:"])
        ]
        
        response = await self.client.get_response(msgs)
        decision_text = response.messages[-1].text or ""
        
        if "ROUTE: STRONG" in decision_text:
            self.state.route = "STRONG"
            print("   [🔀 Decision]: STRONG (Complex Query -> Azure)")
        else:
            self.state.route = "WEAK"
            print("   [🔀 Decision]: WEAK (Simple/Factual -> Local)")
            
        user_msg = Message("user", [query])
        await ctx.send_message(AgentExecutorResponse(
            executor_id=self.id,
            agent_response=AgentResponse(messages=[user_msg]),
            full_conversation=[user_msg],
        ))

def is_route_strong(msg: object) -> bool: return validation_state.route == "STRONG"
def is_route_weak(msg: object) -> bool: return validation_state.route == "WEAK"

validation_state = ValidationState()

async def main():
    print("====================================================")
    print(" predictive-router-pattern (arXiv:2501.01818)")
    print("====================================================\n")

    # 2. Setup Clients
    model_path = os.environ.get("LOCAL_MODEL_PATH", "phi-4-4bit")

    # single local client shared by router and worker
    local_client = create_local_client(model_path, LocalGenerationConfig(temp=0.1))
    
    # strong Model (Azure)
    async with AzureCliCredential() as credential:
        # 3. Define Agents
        router_agent = RouterExecutor(client=local_client, state=validation_state)

        # weak Worker: The 'Mw' model (Runs locally)
        weak_agent = Agent(
            local_client,
            instructions="You are a concise assistant. Answer the user's question directly.",
            name="Weak_Model_Worker",
        )

        # strong Worker: The 'Ms' model (Runs in Cloud)
        strong_agent = Agent(
            FoundryChatClient(project_endpoint=os.environ.get("AZURE_AI_PROJECT_ENDPOINT"), model=os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME"), credential=credential),
            name="Strong_Model_Worker",
            instructions="You are an expert assistant. Provide detailed, reasoning-heavy answers.",
        )

        # 4. build the Workflow Graph
        builder = WorkflowBuilder(start_executor=router_agent)

        builder.add_edge(
            source=router_agent,
            target=strong_agent,
            condition=is_route_strong
        )
        
        builder.add_edge(
            source=router_agent,
            target=weak_agent,
            condition=is_route_weak
        )

        workflow = builder.build()

        # 5. run two demo queries
        queries = [
            # example 1: Complex -> Should route to Strong
            "Explain shortly the implications of quantum computing on cryptography",
            
            # example 2: Simple -> Should route to Weak
            "Write a haiku about ice hockey"
        ]

        for query in queries:
            print(f"\n❔ Query: {query}")
            print("-" * 50)

            # reset state
            validation_state.route = "WEAK"

            current_agent = None
            async for event in workflow.run(query, stream=True):
                if event.type == "output" and isinstance(event.data, AgentResponseUpdate):
                    # print agent name changes
                    if event.executor_id != current_agent:
                        if current_agent: print() 
                        current_agent = event.executor_id
                        print(f" 🤖 {current_agent}: ", end="", flush=True)
                    
                    # print content
                    if event.data and event.data.text:
                        print(event.data.text, end="", flush=True)
            print("\n" + "="*50)
            

if __name__ == "__main__":
    asyncio.run(main())