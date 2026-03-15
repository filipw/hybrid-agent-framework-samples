// =============================================================================
// MAKER Protocol
// Based on: arXiv:2511.09030
//
// A cloud LLM Planner decomposes a complex task into atomic steps.
// A Manager orchestrates a stateful loop: for each step it generates a CoT
// prompt for the local SLM Solver.  The Solver runs the step multiple times
// and applies majority-vote convergence (k_threshold=3 margin).
//
// Backend configuration (see dotnet/.env.example):
//   SLM_BACKEND  — inference backend for the SLM role (default: ollama)
//   LLM_BACKEND  — inference backend for the LLM role (default: azure-openai)
// =============================================================================

using System.Text.Json;
using System.Text.RegularExpressions;
using HybridAgentDemos.Shared;
using Maker;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

// ── Main ─────────────────────────────────────────────────────────────────────

Console.WriteLine("====================================================");
Console.WriteLine("   MAKER Protocol (arXiv:2511.09030)");
Console.WriteLine("====================================================\n");

var llmClient = BackendFactory.CreateLlm();
var slmClient = BackendFactory.CreateSlm();

var state = new MakerState();

var plannerExecutor = new CloudPlannerExecutor(llmClient, state);
var managerExecutor = new ManagerExecutor(state);
var solverExecutor  = new VotingExecutor(slmClient, state);

// Cloud_Planner → Manager → Voting_Solver ↩ (loop until all steps converge)
// Manager terminates by calling YieldOutputAsync without SendMessageAsync
var workflow = new WorkflowBuilder(plannerExecutor)
    .AddEdge(plannerExecutor, managerExecutor)
    .AddEdge(managerExecutor, solverExecutor)
    .AddEdge(solverExecutor,  managerExecutor)
    .WithOutputFrom(managerExecutor)
    .Build();

string userQuery = "Calculate ((5 + 3) * 10) / 2. Then divide this result by 4 and add 6.";
Console.WriteLine($"🚀 Query: {userQuery}");

await using var run = await InProcessExecution.RunStreamingAsync(workflow, userQuery);
await foreach (var evt in run.WatchStreamAsync())
    if (evt is WorkflowErrorEvent err) throw err.Exception!;

// ── Types & Executors ────────────────────────────────────────────────────────

namespace Maker
{
    class MakerState
    {
        public List<string> Steps { get; set; } = [];
        public int CurrentStepIdx { get; set; }
        public List<string> Results { get; set; } = [];
        public Dictionary<string, int> CurrentVotes { get; set; } = new(StringComparer.Ordinal);
        public int Attempts { get; set; }
        public int KThreshold { get; set; } = 3;
        public int MaxAttempts { get; set; } = 15;
        public bool IsComplete { get; set; }
    }

    /// <summary>
    /// [LLM] Cloud_Planner – decomposes the user query into atomic ordered steps.
    /// Sends "PLAN_READY" to the Manager once state.Steps is populated.
    /// </summary>
    sealed class CloudPlannerExecutor(IChatClient llmClient, MakerState state) : Executor<string>("Cloud_Planner")
    {
        private const string PlannerInstructions = """
            You are a decomposition engine. Break the user request into atomic, actionable steps.
            RULES:
            1. Output commands (e.g. 'Add 5 and 3'), NOT results.
            2. Each step must be a single action.
            3. Refer to previous step output as 'the result' or 'the previous state'.
            4. Preserve exact numbers and entities from the user request.
            5. Return ONLY a JSON array of strings. Example: ["Step 1", "Step 2"]
            6. Do NOT use '=' to show results.
            """;

        public override async ValueTask HandleAsync(
            string query, IWorkflowContext context, CancellationToken cancellationToken = default)
        {
            var response = await llmClient.GetResponseAsync(
                [
                    new ChatMessage(ChatRole.System, PlannerInstructions),
                    new ChatMessage(ChatRole.User, query)
                ], cancellationToken: cancellationToken);

            string cleaned = (response.Text ?? "[]").Replace("```json", "").Replace("```", "").Trim();
            try
            {
                state.Steps = JsonSerializer.Deserialize<List<string>>(cleaned) ?? [];
            }
            catch
            {
                int s = cleaned.IndexOf('['), e = cleaned.LastIndexOf(']');
                if (s != -1 && e > s)
                    try { state.Steps = JsonSerializer.Deserialize<List<string>>(cleaned[s..(e + 1)]) ?? []; }
                    catch { /* stay empty */ }
            }

            Console.WriteLine("\n📋 DECOMPOSITION PLAN:");
            Console.WriteLine(JsonSerializer.Serialize(state.Steps, new JsonSerializerOptions { WriteIndented = true }));
            Console.WriteLine(new string('-', 40));

            await context.SendMessageAsync("PLAN_READY", cancellationToken);
        }
    }

    /// <summary>
    /// Manager – pure orchestration, no LLM calls.
    /// Sends a CoT prompt to the Solver, or calls YieldOutputAsync to terminate.
    /// </summary>
    sealed class ManagerExecutor(MakerState state) : Executor<string, string>("Manager")
    {
        private const string CoTInstructions = """
            INSTRUCTION:
            1. Read the Current Task/Action.
            2. Apply this action to the Previous State.
            3. Think step-by-step: what changes and what stays the same.
            4. End with exactly 'Final Answer: [The Updated State or Value]'.
               Use ONLY the raw value – no extra phrases like 'The Final Answer is...'.
            """;

        public override async ValueTask<string> HandleAsync(
            string _, IWorkflowContext context, CancellationToken cancellationToken = default)
        {
            if (state.IsComplete)
            {
                string finalResult = state.Results.Count > 0 ? state.Results[^1] : "N/A";
                Console.WriteLine("\n==========================================");
                Console.WriteLine($"🤖 Final State: {finalResult}");
                Console.WriteLine("==========================================");
                await context.YieldOutputAsync($"WORKFLOW_COMPLETE: {finalResult}", cancellationToken);
                return $"WORKFLOW_COMPLETE: {finalResult}";
            }

            if (state.CurrentStepIdx >= state.Steps.Count)
            {
                await context.YieldOutputAsync("ERROR: No steps remaining.", cancellationToken);
                return "ERROR: No steps remaining.";
            }

            string currentStep = state.Steps[state.CurrentStepIdx];
            string prompt = state.Results.Count == 0
                ? $"Current Task: {currentStep}\n\n{CoTInstructions}\n" +
                  "Since this is the first step, establish the initial state based on the text."
                : $"Previous State: {state.Results[^1]}\n" +
                  $"Current Task: {currentStep}\n\n{CoTInstructions}";

            await context.SendMessageAsync(prompt, cancellationToken);
            return prompt;
        }
    }

    /// <summary>
    /// [SLM] Voting_Solver – executes one step multiple times and applies majority voting.
    /// Always sends a status back to Manager to drive the loop.
    /// </summary>
    sealed class VotingExecutor(IChatClient slmClient, MakerState state) : Executor<string>("Voting_Solver")
    {
        public override async ValueTask HandleAsync(
            string prompt, IWorkflowContext context, CancellationToken cancellationToken = default)
        {
            if (state.Attempts == 0 && prompt.Contains("Current Task:"))
            {
                int start = prompt.IndexOf("Current Task:") + "Current Task:".Length;
                int end   = prompt.IndexOf('\n', start);
                string task = end > start ? prompt[start..end].Trim() : prompt[start..].Trim();
                Console.WriteLine($"\n⏳ Processing Step {state.CurrentStepIdx + 1}: {task}");
            }

            var response = await slmClient.GetResponseAsync(
                [new ChatMessage(ChatRole.User, prompt)], cancellationToken: cancellationToken);
            string answer = ExtractAnswer(response.Text ?? string.Empty);
            state.Attempts++;

            string status;
            if (answer == "PARSE_ERROR")
            {
                Console.WriteLine($"   ❌ Attempt {state.Attempts}: Parse Error");
                if (state.Attempts >= state.MaxAttempts)
                {
                    Console.WriteLine("   ⚠️  ABORTING STEP: Parse Error Limit Reached");
                    CommitStep("ERROR");
                    status = "RESOLVED: ERROR";
                }
                else { status = "RETRY"; }
            }
            else
            {
                state.CurrentVotes[answer] = state.CurrentVotes.GetValueOrDefault(answer) + 1;

                string leader = string.Empty;
                int leaderCount = 0, runnerUp = 0;
                foreach (var (k, v) in state.CurrentVotes)
                {
                    if (v > leaderCount) { runnerUp = leaderCount; leaderCount = v; leader = k; }
                    else if (v > runnerUp) { runnerUp = v; }
                }
                int margin = leaderCount - runnerUp;
                Console.WriteLine($"   - Attempt {state.Attempts}: {answer} | Leader (+{margin})");

                if (margin >= state.KThreshold)
                {
                    Console.WriteLine($"   🎉 CONVERGENCE: {leader}");
                    CommitStep(leader);
                    status = $"RESOLVED: {leader}";
                }
                else if (state.Attempts >= state.MaxAttempts)
                {
                    Console.WriteLine($"   ⚠️  FORCED: {leader}");
                    CommitStep(leader);
                    status = $"RESOLVED: {leader}";
                }
                else { status = "RETRY"; }
            }

            await context.SendMessageAsync(status, cancellationToken);
        }

        private static string ExtractAnswer(string text)
        {
            var match = Regex.Match(text, @"Final Answer:\s*(.*)", RegexOptions.IgnoreCase);
            if (!match.Success) return "PARSE_ERROR";
            string clean = match.Groups[1].Value.Trim().Trim('"').Trim('\'').TrimEnd('.');
            if (clean.StartsWith('[')) clean = clean[1..];
            if (clean.EndsWith(']'))  clean = clean[..^1];
            return string.IsNullOrWhiteSpace(clean) ? "PARSE_ERROR" : clean;
        }

        private void CommitStep(string result)
        {
            state.Results.Add(result);
            state.CurrentStepIdx++;
            state.CurrentVotes.Clear();
            state.Attempts = 0;
            if (state.CurrentStepIdx >= state.Steps.Count)
                state.IsComplete = true;
        }
    }
}
