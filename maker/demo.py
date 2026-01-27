import os
import re
import json
import asyncio
import logging
from collections import Counter
from typing import Any, List, MutableSequence, AsyncIterable
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from agent_framework_mlx import MLXChatClient, MLXGenerationConfig
from agent_framework import (
    ChatAgent, 
    ChatMessage,
    ChatResponseUpdate, 
    WorkflowBuilder, 
    WorkflowContext,
    Executor,
    handler,
    Role,
    BaseChatClient,
    ChatResponse,
    ChatOptions,
    AgentRunUpdateEvent,
    AgentExecutorResponse
)
from agent_framework_azure_ai import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("agent_framework").setLevel(logging.ERROR) 
load_dotenv()

class MakerState(BaseModel):
    steps: List[str] = Field(default_factory=list)
    current_step_idx: int = 0
    results: List[str] = Field(default_factory=list)
    current_votes: Counter = Field(default_factory=Counter)
    attempts: int = 0
    k_threshold: int = 3
    max_attempts: int = 15
    is_complete: bool = False

class ManagerClient(BaseChatClient):
    """
    UNIVERSAL ORCHESTRATOR: 
    Instead of forcing time math, this asks the model to update the 'State'.
    """
    def __init__(self, state: MakerState, **kwargs):
        super().__init__(**kwargs)
        self.state = state

    async def _generate_text(self) -> str:
        if self.state.is_complete:
            return f"WORKFLOW_COMPLETE: {self.state.results[-1] if self.state.results else 'N/A'}"

        if self.state.current_step_idx < len(self.state.steps):
            current_step = self.state.steps[self.state.current_step_idx]
            
            cot = (
                "INSTRUCTION: \n"
                "1. Read the Current Task/Action.\n"
                "2. Apply this action to the Previous State.\n"
                "3. Think step-by-step: describe what changes and what stays the same.\n"
                "4. You MUST end your response with exactly 'Final Answer: [The Updated State or Value]'. Use ONLY the raw state value not any additional expression like 'The Final Answer is...' or 'The updated state is ...'.\n"
            )

            if not self.state.results:
                # STEP 1
                prompt = (
                    f"Current Task: {current_step}\n\n"
                    f"{cot}\n"
                    "Since this is the first step, establish the initial state based on the text."
                )
            else:
                # STEP 2+
                last_result = self.state.results[-1]
                prompt = (
                    f"Previous State: {last_result}\n"
                    f"Current Task: {current_step}\n\n"
                    f"{cot}\n"
                )
            return prompt
        
        return "ERROR: No steps remaining."

    async def _inner_get_response(
        self,
        *,
        messages: MutableSequence[ChatMessage],
        options: dict[str, Any],
        **kwargs: Any,
    ) -> ChatResponse:
        text = await self._generate_text()
        return ChatResponse(messages=[ChatMessage(role=Role.ASSISTANT, text=text)])

    async def _inner_get_streaming_response(
        self,
        *,
        messages: MutableSequence[ChatMessage],
        options: dict[str, Any],
        **kwargs: Any,
    ) -> AsyncIterable[ChatResponseUpdate]:
        text = await self._generate_text()
        yield ChatResponseUpdate(role=Role.ASSISTANT, text=text)

class VotingExecutor(Executor):
    def __init__(self, name: str, client: BaseChatClient, state: MakerState):
        super().__init__(id=name)
        self.client = client
        self.state = state

    def _extract_answer(self, text: str) -> str:
        # answer should be in format: Final Answer: [answer], and we trim the [] too
        match = re.search(r"Final Answer:\s*(.*)", text, re.IGNORECASE)
        if match:
            clean = match.group(1).strip().strip('"').strip("'").rstrip(".")
            if clean.startswith("["):
                clean = clean[1:]
            if clean.endswith("]"):
                clean = clean[:-1]
            return clean
        return "PARSE_ERROR"

    @handler
    async def handle_agent_response(self, message: AgentExecutorResponse, ctx: WorkflowContext[ChatMessage]): 
        await self._resolve_step(message.agent_response.text or "", ctx)

    @handler
    async def handle_chat_message(self, message: ChatMessage, ctx: WorkflowContext[ChatMessage]):
        await self._resolve_step(message.text or "", ctx)

    async def _resolve_step(self, input_text: str, ctx: WorkflowContext[ChatMessage]):
        
        if self.state.attempts == 0 and "Current Task:" in input_text:
            task_line = input_text.split("Current Task:")[1].split("\n")[0].strip()
            print(f"\n⏳ Processing Step {self.state.current_step_idx + 1}: {task_line}")

        msgs = [ChatMessage(role=Role.USER, text=input_text)]
        response = await self.client.get_response(msgs)

        ans = self._extract_answer(response.messages[-1].text or "")
        self.state.attempts += 1
        
        status_msg = ""
        if ans == "PARSE_ERROR":
            print(f"   ❌ Attempt {self.state.attempts}: Parse Error")
            if self.state.attempts >= self.state.max_attempts:
                 print(f"   ⚠️ ABORTING STEP: Parse Error Limit Reached")
                 status_msg = "RESOLVED: ERROR"
                 self._commit_step("ERROR")
            else:
                 status_msg = "RETRY"
        else:
            self.state.current_votes[ans] += 1
            leader, count = self.state.current_votes.most_common(1)[0]
            
            runner_up = 0
            if len(self.state.current_votes) > 1:
                runner_up = self.state.current_votes.most_common(2)[1][1]
            margin = count - runner_up
            
            print(f"   - Attempt {self.state.attempts}: {ans} | Leader (+{margin})")

            if margin >= self.state.k_threshold:
                print(f"   🎉 CONVERGENCE: {ans}")
                status_msg = f"RESOLVED: {ans}"
                self._commit_step(ans)
            
            elif self.state.attempts >= self.state.max_attempts:
                print(f"   ⚠️ FORCED: {ans}")
                status_msg = f"RESOLVED: {ans}"
                self._commit_step(ans)
            else:
                status_msg = "RETRY"

        await ctx.send_message(ChatMessage(role=Role.ASSISTANT, text=status_msg))

    def _commit_step(self, result: str):
        self.state.results.append(result)
        self.state.current_step_idx += 1
        self.state.current_votes.clear()
        self.state.attempts = 0
        if self.state.current_step_idx >= len(self.state.steps):
            self.state.is_complete = True

class DecompositionPlan(BaseModel):
    steps: List[str | dict] = Field(description="A list of imperative instructions (e.g. 'Add 5 and 3'). Do NOT include results (e.g. '5+3=8').")

def create_transitions(state: MakerState):
    def parse_plan(response: AgentExecutorResponse) -> bool:
        # Check for structured output first
        if getattr(response.agent_response, "value", None) and isinstance(response.agent_response.value, DecompositionPlan):
             raw_steps = response.agent_response.value.steps
             # Normalize dictionary steps if necessary
             state.steps = [
                 step.get("content", step.get("step", str(step))) if isinstance(step, dict) else str(step) 
                 for step in raw_steps
             ]
             print("\n📋 DECOMPOSITION PLAN (Structured):")
             print(json.dumps(state.steps, indent=2))
             print("-" * 40)
             return True

        text = response.agent_response.text or "[]"
        clean_text = text.replace("```json", "").replace("```", "").strip()
        try:
            state.steps = json.loads(clean_text)
            print("\n📋 DECOMPOSITION PLAN:")
            print(json.dumps(state.steps, indent=2))
            print("-" * 40)
            return True
        except: return False

    def to_solver(response: AgentExecutorResponse) -> bool: return not state.is_complete
    def to_manager(response: AgentExecutorResponse) -> bool: return True
    return parse_plan, to_solver, to_manager

def ensure_stateless(msgs): 
    trimmed = msgs[-1]
    return [trimmed]
        
async def main():
    print("====================================================")
    print("   MAKER Protocol   ")
    print("====================================================\n")

    state = MakerState()
    t_parse, t_to_solver, t_to_manager = create_transitions(state)

    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(credential=credential).as_agent(
            name="Cloud_Planner",
            instructions=(
                "You are a decomposition engine. Your goal is to break a complex user request into a sequence of atomic, actionable steps for a worker agent to execute one by one.\n\n"
                "RULES:\n"
                "1. INSTRUCTION NOT SOLUTION: Output commands (e.g., 'Find X', 'Calculate Y', 'Search for Z'), NOT results or facts. Do not perform the task yourself.\n"
                "2. ATOMIC STEPS: Each step must be a single, distinct action.\n"
                "3. FORWARD REFERENCES: Refer to the output of previous steps as 'the result', 'the output', or 'the previous state'. Do not assume you know what the value is.\n"
                "4. DATA FIDELITY: Preserve specific names, numbers, and entities from the user's request exactly.\n"
                "5. NO PREDICTIONS: Do not predict, simulate, or include the outcome of the steps in the instructions.\n"
                "6. FORMAT: Return a list of strings where each string is an imperative instruction.\n"
                "7. NO EQUALS SIGNS: Do not use '=' to show results.\n"
            ),
            default_options={"response_format": DecompositionPlan},
        ) as cloud_planner,
    ):
        manager = ChatAgent(
            name="Manager", 
            instructions="Orchestrator", 
            chat_client=ManagerClient(state)
        )
        
        mlx_generation_config = MLXGenerationConfig(max_tokens=300, temp=0.8)
        mlx_client = MLXChatClient(model_path="mlx-community/Phi-4-mini-instruct-4bit", generation_config=mlx_generation_config, message_preprocessor=ensure_stateless)
        
        solver = VotingExecutor(
            name="Voting_Solver", 
            client=mlx_client, 
            state=state
        )

        builder = WorkflowBuilder()
        builder.set_start_executor(cloud_planner)
        builder.add_edge(source=cloud_planner, target=manager, condition=t_parse)
        builder.add_edge(source=manager, target=solver, condition=t_to_solver)
        builder.add_edge(source=solver, target=manager, condition=t_to_manager)

        workflow = builder.build()

        user_query = "Calculate ((5 + 3) * 10) / 2. Then divide this result by 4 and add 6."

        print(f"🚀 Query: {user_query}")

        async for event in workflow.run_stream(user_query):
            if isinstance(event, AgentRunUpdateEvent):
                if event.executor_id == "Manager" and "WORKFLOW_COMPLETE" in (event.data.text or ""):
                    print(f"\n==========================================")
                    print(f"🤖 Final State: {event.data.text.split('WORKFLOW_COMPLETE: ')[1]}")
                    print(f"==========================================")
                    break

if __name__ == "__main__":
    asyncio.run(main())
