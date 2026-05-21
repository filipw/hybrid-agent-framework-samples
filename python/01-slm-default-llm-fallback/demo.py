import os
import sys
import re
import asyncio
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from agent_framework import Agent, WorkflowBuilder, AgentExecutorResponse, AgentResponse, WorkflowEvent, AgentResponseUpdate, Executor, handler, WorkflowContext, Message
from azure.identity.aio import AzureCliCredential
from agent_framework_foundry import FoundryChatClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from local_models import create_local_client, LocalGenerationConfig

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("agent_framework").setLevel(logging.ERROR) 

load_dotenv()

class ConfidenceResult(BaseModel):
    score: int = Field(alias="confidence")
    
    @classmethod
    def parse_from_text(cls, text: str) -> "ConfidenceResult":
        match = re.search(r"CONFIDENCE:\s*(\d+)", text, re.IGNORECASE)
        if match:
            return cls(confidence=int(match.group(1)))
        return cls(confidence=0)

def should_fallback_to_cloud(message: AgentExecutorResponse) -> bool:
    text = message.agent_response.text or ""
    result = ConfidenceResult.parse_from_text(text)
    
    print(f"\n\n   📊 Verifier Score: {result.score}/10")
    
    if result.score < 8:
        print("   ⚠️ Low Confidence. Routing to Cloud...")
        return True
    
    print("   ✅ High Confidence. Workflow Complete.")
    return False

def inject_confidence(msgs): 
    if msgs: 
        msgs[-1]["content"] += "\nIMPORTANT: End response with 'CONFIDENCE: X' (1-10). You are allowed to  output a score of 8 or higher ONLY IF you are very sure of your answer."
    return msgs

class InputForwarder(Executor):
    def __init__(self):
        super().__init__(id="Input")

    @handler
    async def forward(self, query: str, ctx: WorkflowContext[AgentExecutorResponse]):
        user_msg = Message("user", [query])
        await ctx.send_message(AgentExecutorResponse(
            executor_id=self.id,
            agent_response=AgentResponse(messages=[user_msg]),
            full_conversation=[user_msg],
        ))

async def main():
    print("====================================================")
    print("   Cascade Pattern with Microsoft Agent Framework")
    print("====================================================\n")

    queries = [
        # 1. Easy Fact
        "What is the capital of France?",

        # 1b. Tricky Fact
        "In which year was Wisloka Debica founded?",
        
        # 2. Extraction
        "Convert this list to a JSON shopping list: Apple 2 items, Banana 3 items, Cherries 1 item. Return pure JSON no additional text or formatting.",
        
        # 3. Amiguous
        "Where is the city of Springfield located?",

        # 4. Hallucination Trap
        "Explain in 2 sentences the role of quantum healing in modeling proteins.",
    ]

    local_config = LocalGenerationConfig(max_tokens=300)
    local_client = create_local_client(
        model_path=os.environ.get("LOCAL_MODEL_PATH", "phi-4-4bit"),
        generation_config=local_config,
        message_preprocessor=inject_confidence,
    )

    for q in queries:
        print(f"\n❔ Query: {q}")
        print("-" * 40)
            
        # Agents hold conversation history, so for each query demoinstration we create a new pair of local/remote agents
        async with (
            AzureCliCredential() as credential,
            Agent(
                FoundryChatClient(project_endpoint=os.environ.get("AZURE_AI_PROJECT_ENDPOINT"), model=os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME"), credential=credential),
                name="Cloud_LLM",
                instructions="You are a fallback expert. The previous assistant was unsure. Provide a complete answer.",
            ) as cloud_agent,
        ):
            local_agent = Agent(
                local_client,
                instructions="You are a helpful assistant. Always end your response with 'CONFIDENCE: X' where X is a number from 1-10 reflecting how confident you are in your answer. If you are sure of your answer, you MUST output a score of 8 or higher.",
                name="Local_SLM",
            )

            input_forwarder = InputForwarder()
            builder = WorkflowBuilder(start_executor=input_forwarder)
            builder.add_edge(source=input_forwarder, target=local_agent)
            builder.add_edge(
                source=local_agent,
                target=cloud_agent,
                condition=should_fallback_to_cloud
            )
            
            workflow = builder.build()

            current_agent = None
            
            async for event in workflow.run(q, stream=True):
                if event.type == "output" and isinstance(event.data, AgentResponseUpdate):
                    if event.executor_id != current_agent:
                        if current_agent: print() 
                        current_agent = event.executor_id
                        print(f"   🤖 {current_agent}: ", end="", flush=True)
                    
                    if event.data and event.data.text:
                        print(event.data.text, end="", flush=True)
            print("\n")

if __name__ == "__main__":
    asyncio.run(main())