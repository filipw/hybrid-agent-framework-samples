import os
import json
import asyncio
import logging
import time
import re
from typing import List
from pydantic import BaseModel, Field
from agent_framework_mlx import MLXChatClient, MLXGenerationConfig
from agent_framework import (
    ChatMessage,
    WorkflowBuilder,
    WorkflowContext,
    Executor,
    handler,
    Role,
    AgentRunUpdateEvent,
    AgentExecutorResponse,
)
from agent_framework_azure_ai import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("agent_framework").setLevel(logging.ERROR)

LOCAL_MODEL_PATH = "mlx-community/Phi-4-mini-instruct-8bit"

with open("quantum_mechanics_history.txt", "r", encoding="utf-8") as f:
    QUANTUM_MECHANICS_HISTORY = f.read()

class MinionsState(BaseModel):
    user_query: str = ""
    jobs: List[str] = Field(default_factory=list)
    results: List[str] = Field(default_factory=list)
    final_answer: str = ""
    local_chars_processed: int = 0

LOCAL_WORKER_PROMPT = """You are a meticulous and literal fact-checker. Your process is a strict two-step evaluation:
1.  First, analyze the 'Context' to determine if it contains any information that can directly answer the 'Task'.
2.  If the 'Context' is NOT relevant to the 'Task', you MUST immediately stop and respond with the single word: none.

- If and ONLY IF the information is present, extract the relevant facts verbatim or as a close paraphrase.
- Do NOT invent, guess, or mix information from different people or concepts. If the primary subject of the 'Task' (e.g., a person's name) is not mentioned in the 'Context', the answer is always 'none'.
- Your final output must be ONLY the extracted data or the word 'none'.

Context:
---
{chunk}
---
Task: Based ONLY on the text in the 'Context' above, {job}
"""

DECOMPOSER_INSTRUCTIONS = """You are a task decomposition expert. Your job is to break down a user's complex query into simple, atomic extraction tasks.

Rules:
- Each task should be a single, focused question that can be answered from a text chunk
- Tasks should be specific and actionable (e.g., "Find X and the year Y") and intended for finding information in the attached text
- Create 3-7 tasks depending on the complexity of the query
- Return ONLY a JSON array of task strings, nothing else
- Format: ["task 1", "task 2", "task 3"]"""

SYNTHESIZER_INSTRUCTIONS = """You are a science historian. You have received a list of facts extracted from a document about the history of physics.
- Your task is to synthesize this information into a clear, structured answer to the user's original query.
- Organize the information by scientist.
- Do not mention the extraction process, just provide the final answer."""

EVALUATOR_INSTRUCTIONS_TEMPLATE = """You are an expert evaluator of AI-generated responses. Your task is to assess the quality of an answer given the original document, user query, and the generated response.

Please evaluate the response on a scale of 1-5 where:
1 = Very Poor (completely inaccurate or irrelevant)
2 = Poor (mostly inaccurate with some relevant information)
3 = Fair (some accuracy but missing key information or has notable errors)
4 = Good (mostly accurate and complete with minor issues)
5 = Excellent (highly accurate, complete, and well-structured)

Consider these criteria:
- Accuracy: Does the answer correctly reflect the information in the source document?
- Completeness: Does it address all parts of the user's query?
- Clarity: Is the answer well-organized and easy to understand?
- Relevance: Does it stay focused on what was asked?

Provide your evaluation in this exact format:
Score: [1-5]
Reasoning: [Brief explanation of your assessment]

Original Document:
{document}

User Query: {query}

You will receive the generated answer as input. Evaluate it against the document and query above."""

def ensure_stateless(msgs):
    return [msgs[-1]]

class LocalWorkerExecutor(Executor):
    """Processes document chunks with the local SLM to extract relevant facts for each job."""

    def __init__(self, name: str, client: MLXChatClient, state: MinionsState, document: str, chunk_size: int = 500):
        super().__init__(id=name)
        self.client = client
        self.state = state
        self.document = document
        self.chunk_size = chunk_size

    @handler
    async def handle_decomposer_response(self, message: AgentExecutorResponse, ctx: WorkflowContext[str]):
        jobs = self.state.jobs
        chunks = [self.document[i:i + self.chunk_size] for i in range(0, len(self.document), self.chunk_size)]

        print(f"\n--- Step 2: Job Execution (LocalLM) ---")
        print(f"Document split into {len(chunks)} chunks. Running {len(jobs)} jobs per chunk...")

        start_time = time.time()

        for i, chunk in enumerate(chunks):
            for job in jobs:
                prompt = LOCAL_WORKER_PROMPT.format(chunk=chunk, job=job)
                self.state.local_chars_processed += len(prompt)

                response = await self.client.get_response(
                    [ChatMessage(role=Role.USER, text=prompt)]
                )
                result = (response.messages[-1].text or "").strip()

                result_lower = result.lower()
                is_failure = result_lower == "none"

                if not is_failure and result:
                    print(f"  - SUCCESS: Found relevant result in chunk {i + 1}!")
                    self.state.results.append(result)
                else:
                    print(f"  - No relevant info found in chunk {i + 1}.")

                print(f"    (LocalLM response: '{result}')")

        duration = time.time() - start_time
        print(f"Local job execution finished in {duration:.2f}s.")
        print(f"Filtered results to be sent to RemoteLM: {self.state.results}")

        if self.state.results:
            results_str = "\n".join([f"- {res}" for res in self.state.results])
            synthesis_request = (
                f"Original Query: {self.state.user_query}\n\n"
                f"Extracted Information:\n{results_str}\n\n"
                f"Please provide a final, synthesized answer."
            )
            await ctx.send_message(synthesis_request)
        else:
            await ctx.send_message("NO_RESULTS")


class EvalFormatterExecutor(Executor):
    """Captures the synthesizer's answer and formats an explicit evaluation request."""

    def __init__(self, state: MinionsState):
        super().__init__(id="Eval_Formatter")
        self.state = state

    @handler
    async def format_for_eval(self, message: AgentExecutorResponse, ctx: WorkflowContext[str]):
        self.state.final_answer = message.agent_response.text or ""
        eval_request = (
            f"Evaluate the following AI-generated answer.\n\n"
            f"Generated Answer:\n{self.state.final_answer}\n\n"
            f"Provide your evaluation in this exact format:\n"
            f"Score: [1-5]\n"
            f"Reasoning: [Brief explanation of your assessment]"
        )
        await ctx.send_message(eval_request)


def create_transitions(state: MinionsState):

    def parse_jobs(response: AgentExecutorResponse) -> bool:
        """Parses the decomposer's JSON job list into shared state."""
        print("\n--- Step 1: Job Preparation (RemoteLM) ---")
        text = response.agent_response.text or ""
        try:
            clean = text.strip()
            if not clean.startswith("["):
                start_idx = clean.find("[")
                end_idx = clean.rfind("]") + 1
                if start_idx != -1 and end_idx > start_idx:
                    clean = clean[start_idx:end_idx]
            state.jobs = json.loads(clean)
            print(f"Jobs created: {json.dumps(state.jobs, indent=2)}")
            return True
        except json.JSONDecodeError:
            # Fallback to predefined jobs
            state.jobs = [
                "Find the contribution of Max Planck and the year it was made.",
                "Find the contribution of Albert Einstein and the year it was made.",
                "Find the contribution of Niels Bohr and the year it was made.",
            ]
            print(f"Warning: Could not parse JSON. Using fallback jobs: {state.jobs}")
            return True

    def has_results(msg) -> bool:
        """Only proceed to synthesis if the local worker found results."""
        if isinstance(msg, str) and msg == "NO_RESULTS":
            print("\nLocalLM could not find any relevant information. Halting.")
            return False
        return len(state.results) > 0

    return parse_jobs, has_results


async def main():
    start_time = time.time()
    print("=" * 50)
    print("      MINIONS Protocol Demo (Agent Framework)")
    print("=" * 50)

    user_query = "what did Planck, Einstein, and Bohr contribute to quantum mechanics?"
    print(f"\nUser Query: {user_query}")

    state = MinionsState(user_query=user_query)
    parse_jobs, has_results = create_transitions(state)

    async with AzureCliCredential() as credential:
        azure_client = AzureAIAgentClient(credential=credential)

        decomposer = azure_client.as_agent(
            name="Cloud_Decomposer",
            instructions=DECOMPOSER_INSTRUCTIONS,
        )

        synthesizer = azure_client.as_agent(
            name="Cloud_Synthesizer",
            instructions=SYNTHESIZER_INSTRUCTIONS,
        )

        evaluator = azure_client.as_agent(
            name="Cloud_Evaluator",
            instructions=EVALUATOR_INSTRUCTIONS_TEMPLATE.format(
                document=QUANTUM_MECHANICS_HISTORY,
                query=user_query,
            ),
        )

        mlx_config = MLXGenerationConfig(max_tokens=250, temp=0.1)
        mlx_client = MLXChatClient(
            model_path=LOCAL_MODEL_PATH,
            generation_config=mlx_config,
            message_preprocessor=ensure_stateless,
        )

        local_worker = LocalWorkerExecutor(
            name="Local_Worker",
            client=mlx_client,
            state=state,
            document=QUANTUM_MECHANICS_HISTORY,
        )

        eval_formatter = EvalFormatterExecutor(state=state)

        # Cloud_Decomposer → (parse_jobs) → Local_Worker → (has_results) → Cloud_Synthesizer → Eval_Formatter → Cloud_Evaluator
        builder = WorkflowBuilder()
        builder.set_start_executor(decomposer)
        builder.add_edge(source=decomposer, target=local_worker, condition=parse_jobs)
        builder.add_edge(source=local_worker, target=synthesizer, condition=has_results)
        builder.add_edge(source=synthesizer, target=eval_formatter)
        builder.add_edge(source=eval_formatter, target=evaluator)

        workflow = builder.build()

        current_agent = None
        evaluator_text = ""

        decomposition_input = (
            f"User Query: {user_query}\n\n"
            f"Break this query down into simple extraction tasks. Return only the JSON array."
        )

        INTERNAL_EXECUTORS = {"Local_Worker", "Eval_Formatter"}

        async for event in workflow.run_stream(decomposition_input):
            if isinstance(event, AgentRunUpdateEvent):
                if event.executor_id != current_agent:
                    if current_agent:
                        print()
                    current_agent = event.executor_id

                    if current_agent == "Cloud_Synthesizer":
                        print("\n--- Step 3: Aggregation & Synthesis (RemoteLM) ---")
                    elif current_agent == "Cloud_Evaluator":
                        print("\n--- Step 4: Response Quality Evaluation (RemoteLM) ---")

                    # Don't print headers for internal executors
                    if current_agent not in INTERNAL_EXECUTORS:
                        print(f"🤖 {current_agent}: ", end="", flush=True)

                if event.data and event.data.text:
                    if current_agent not in INTERNAL_EXECUTORS:
                        print(event.data.text, end="", flush=True)
                    if current_agent == "Cloud_Evaluator":
                        evaluator_text += event.data.text

        eval_score = 0
        try:
            score_match = re.search(r"Score:\s*(\d+)", evaluator_text)
            if score_match:
                eval_score = int(score_match.group(1))
        except (IndexError, ValueError):
            pass

        total_duration = time.time() - start_time

        print("\n\n" + "=" * 50)
        print("--- FINAL ANSWER ---")
        print(state.final_answer)
        print("=" * 50)

        print("\n--- RESPONSE QUALITY EVALUATION ---")
        print(f"Quality Score: {eval_score}/5")
        print(f"Evaluation Details:\n{evaluator_text}")
        print("=" * 50)

        print("\n--- Performance & Results Report ---")
        print(f"Total Workflow Duration: {total_duration:.2f}s")
        print(f"\nCost & Efficiency Analysis (using character counts):")
        print(f"  - Characters processed by FREE LocalLM: ~{state.local_chars_processed}")
        print(f"  - Jobs created by cloud: {len(state.jobs)}")
        print(f"  - Results extracted locally: {len(state.results)}")
        print(f"\nAnswer Quality:")
        print(f"  - AI Judge Score: {eval_score}/5")
        print("=" * 50)

        await azure_client.close()


if __name__ == "__main__":
    asyncio.run(main())