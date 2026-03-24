import os
import sys
import re
import asyncio
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from agent_framework import ChatAgent, WorkflowBuilder, AgentRunUpdateEvent, AgentExecutorResponse
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity.aio import AzureCliCredential
from agent_framework_azure_ai import AzureAIAgentClient

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
    if msgs: msgs[-1]["content"] += "\nIMPORTANT: End response with 'CONFIDENCE: X' (1-10). If you are sure of your answer, you MUST output a score of 8 or higher."
    return msgs

async def main():
    print("====================================================")
    print("   Cascade Pattern with Microsoft Agent Framework")
    print("====================================================\n")

    queries = [
        # 1. Easy Fact (High Confidence)
        "What is the capital of France?",
        
        # 2. Logic/Code (High Confidence)
        "Convert this list to a JSON array: Apple, Banana, Cherry",
        
        # 3. Amiguous
        "Where is the city of Springfield located?",

        # 4. Hallucination Trap
        "Explain in 2 sentences the role of quantum healing in modeling proteins.",
        
        # 5. Reasoning
        "If I have a cabbage, a goat, and a wolf, and I need to cross a river but can only take one item at a time, and I can't leave the goat with the cabbage or the wolf with the goat, how do I do it?",
    ]

    local_config = LocalGenerationConfig(max_tokens=300)
    local_client = create_local_client(
        model_path=os.environ.get("LOCAL_MODEL_PATH", "Phi-4-mini-instruct-4bit"),
        generation_config=local_config,
        message_preprocessor=inject_confidence,
    )

    for q in queries:
        print(f"\n❔ Query: {q}")
        print("-" * 40)
            
        # Agents hold conversation history, so for each query demoinstration we create a new pair of local/remote agents
        async with (
            AzureCliCredential() as credential,
            AzureAIAgentClient(credential=credential).as_agent(
                name="Cloud_LLM",
                instructions="You are a fallback expert. The previous assistant was unsure. Provide a complete answer.",
            ) as cloud_agent,
        ):
            local_agent = ChatAgent(
                name="Local_SLM",
                instructions="You are a helpful assistant.",
                chat_client=local_client
            )

            builder = WorkflowBuilder()
            builder.set_start_executor(local_agent)
            
            builder.add_edge(
                source=local_agent,
                target=cloud_agent,
                condition=should_fallback_to_cloud
            )
            
            workflow = builder.build()

            current_agent = None
            
            async for event in workflow.run_stream(q):
                if isinstance(event, AgentRunUpdateEvent):
                    if event.executor_id != current_agent:
                        if current_agent: print() 
                        current_agent = event.executor_id
                        print(f"   🤖 {current_agent}: ", end="", flush=True)
                    
                    if event.data and event.data.text:
                        print(event.data.text, end="", flush=True)
            print("\n")

if __name__ == "__main__":
    asyncio.run(main())