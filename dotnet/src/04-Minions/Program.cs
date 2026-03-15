// =============================================================================
// MINIONS Protocol (Local-Remote Map-Reduce)
// Based on: arXiv:2502.15964
//
// A cloud LLM Decomposer breaks the user query into 3-7 atomic extraction
// jobs.  A local SLM Worker runs each job on every chunk of the document
// (map phase) and filters out irrelevant chunks ("none" responses).  A cloud
// Synthesizer aggregates the extracted facts into a final answer, and a cloud
// Evaluator scores the answer quality.
//
// Backend configuration (see dotnet/.env.example):
//   SLM_BACKEND  — inference backend for the SLM role (default: ollama)
//   LLM_BACKEND  — inference backend for the LLM role (default: azure-openai)
// =============================================================================

using System.Diagnostics;
using System.Text.RegularExpressions;
using HybridAgentDemos.Shared;
using Minions;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

// ── Main ─────────────────────────────────────────────────────────────────────

var totalSw = Stopwatch.StartNew();

Console.WriteLine(new string('=', 50));
Console.WriteLine("      MINIONS Protocol Demo (Agent Framework)");
Console.WriteLine(new string('=', 50));

string userQuery    = "what did Planck, Einstein, and Bohr contribute to quantum mechanics?";
string documentPath = Path.Combine(AppContext.BaseDirectory, "quantum_mechanics_history.txt");
string document     = File.ReadAllText(documentPath);

Console.WriteLine($"\nUser Query: {userQuery}");

var state = new MinionsState { UserQuery = userQuery };

var llmClient = BackendFactory.CreateLlm();
var slmClient = BackendFactory.CreateSlm();

var decomposer    = new CloudDecomposerExecutor(llmClient, state);
var localWorker   = new LocalWorkerExecutor(slmClient, state, document);
var synthesizer   = new CloudSynthesizerExecutor(llmClient);
var evalFormatter = new EvalFormatterExecutor(state);
var evaluator     = new CloudEvaluatorExecutor(llmClient, document, userQuery);

// Cloud_Decomposer → Local_Worker → Cloud_Synthesizer → Eval_Formatter → Cloud_Evaluator
Func<object?, bool> jobsReady   = _ => state.Jobs.Count > 0;
Func<object?, bool> hasResults  = msg => msg is string s && s != "NO_RESULTS";

var workflow = new WorkflowBuilder(decomposer)
    .AddEdge(decomposer,    localWorker,   jobsReady)
    .AddEdge(localWorker,   synthesizer,   hasResults)
    .AddEdge(synthesizer,   evalFormatter)
    .AddEdge(evalFormatter, evaluator)
    .WithOutputFrom(evaluator)
    .Build();

string evalOutput = string.Empty;
await using var run = await InProcessExecution.RunStreamingAsync(workflow, userQuery);
await foreach (var evt in run.WatchStreamAsync())
{
    if (evt is WorkflowErrorEvent err) throw err.Exception!;
    if (evt is WorkflowOutputEvent output && output.As<string>() is string evalText)
        evalOutput = evalText;
}

int evalScore  = 0;
var scoreMatch = Regex.Match(evalOutput, @"Score:\s*(\d+)");
if (scoreMatch.Success) evalScore = int.Parse(scoreMatch.Groups[1].Value);

totalSw.Stop();

Console.WriteLine("\n\n" + new string('=', 50));
Console.WriteLine("--- FINAL ANSWER ---");
Console.WriteLine(state.FinalAnswer);
Console.WriteLine(new string('=', 50));
Console.WriteLine("\n--- RESPONSE QUALITY EVALUATION ---");
Console.WriteLine($"Quality Score: {evalScore}/5");
Console.WriteLine($"Evaluation Details:\n{evalOutput}");
Console.WriteLine(new string('=', 50));
Console.WriteLine("\n--- Performance & Results Report ---");
Console.WriteLine($"Total Workflow Duration: {totalSw.Elapsed.TotalSeconds:F2}s");
Console.WriteLine($"  - Characters processed by LocalLM (SLM): ~{state.LocalCharsProcessed}");
Console.WriteLine($"  - Jobs created by cloud: {state.Jobs.Count}");
Console.WriteLine($"  - Results extracted locally: {state.Results.Count}");
Console.WriteLine($"  - AI Judge Score: {evalScore}/5");
Console.WriteLine(new string('=', 50));

// ── Types & Executors ────────────────────────────────────────────────────────

namespace Minions
{
    static class Prompts
    {
        public const string LocalWorkerTemplate = """
            You are a meticulous and literal fact-checker. Your process is a strict two-step evaluation:
            1. First, analyze the 'Context' to determine if it contains any information that can directly answer the 'Task'.
            2. If the 'Context' is NOT relevant to the 'Task', you MUST immediately stop and respond with the single word: none.

            - If and ONLY IF the information is present, extract the relevant facts verbatim or as a close paraphrase.
            - Do NOT invent, guess, or mix information from different people or concepts.
            - Your final output must be ONLY the extracted data or the word 'none'.

            Context:
            ---
            {chunk}
            ---
            Task: Based ONLY on the text in the 'Context' above, {job}
            """;

        public const string Decomposer = """
            You are a task decomposition expert. Break down the user's complex query
            into simple, atomic extraction tasks.
            Rules:
            - Each task should be a single, focused question answerable from a text chunk.
            - Create 3-7 tasks depending on query complexity.
            - Return ONLY a JSON array of task strings. Format: ["task 1", "task 2"]
            """;

        public const string Synthesizer = """
            You are a science historian. You have received a list of facts extracted from a document.
            - Synthesize this information into a clear, structured answer to the user's original query.
            - Organize the information by scientist.
            - Do not mention the extraction process, just provide the final answer.
            """;
    }

    class MinionsState
    {
        public string UserQuery { get; set; } = string.Empty;
        public List<string> Jobs { get; set; } = [];
        public List<string> Results { get; set; } = [];
        public string FinalAnswer { get; set; } = string.Empty;
        public int LocalCharsProcessed { get; set; }
    }

    /// <summary>
    /// [LLM] Cloud_Decomposer – breaks the user query into atomic extraction jobs.
    /// </summary>
    sealed class CloudDecomposerExecutor(IChatClient llmClient, MinionsState state)
        : Executor<string, string>("Cloud_Decomposer")
    {
        public override async ValueTask<string> HandleAsync(
            string _, IWorkflowContext context, CancellationToken cancellationToken = default)
        {
            Console.WriteLine("\n--- Step 1: Job Preparation (RemoteLM) ---");
            Console.Write("🤖 Cloud_Decomposer: ");

            string fullText = string.Empty;
            await foreach (var update in llmClient.GetStreamingResponseAsync(
                [
                    new ChatMessage(ChatRole.System, Prompts.Decomposer),
                    new ChatMessage(ChatRole.User,
                        $"User Query: {state.UserQuery}\n\nBreak this query into extraction tasks. Return only the JSON array.")
                ], cancellationToken: cancellationToken))
            {
                Console.Write(update.Text);
                fullText += update.Text ?? string.Empty;
            }
            Console.WriteLine();

            string cleaned = fullText.Trim();
            if (!cleaned.StartsWith("["))
            {
                int s = cleaned.IndexOf('['), e = cleaned.LastIndexOf(']') + 1;
                if (s != -1 && e > s) cleaned = cleaned[s..e];
            }
            try
            {
                state.Jobs = System.Text.Json.JsonSerializer.Deserialize<List<string>>(cleaned) ?? [];
            }
            catch
            {
                state.Jobs = [
                    "Find the contribution of Max Planck and the year it was made.",
                    "Find the contribution of Albert Einstein and the year it was made.",
                    "Find the contribution of Niels Bohr and the year it was made.",
                ];
                Console.WriteLine("Warning: Could not parse JSON. Using fallback jobs.");
            }

            Console.WriteLine($"Jobs created: {System.Text.Json.JsonSerializer.Serialize(state.Jobs,
                new System.Text.Json.JsonSerializerOptions { WriteIndented = true })}");
            return "JOBS_READY";
        }
    }

    /// <summary>
    /// [SLM] Local_Worker – map phase: for each chunk × job, run the local model.
    /// </summary>
    sealed class LocalWorkerExecutor(IChatClient slmClient, MinionsState state, string document, int chunkSize = 500)
        : Executor<string, string>("Local_Worker")
    {
        public override async ValueTask<string> HandleAsync(
            string _, IWorkflowContext context, CancellationToken cancellationToken = default)
        {
            var chunks = Enumerable.Range(0, (document.Length + chunkSize - 1) / chunkSize)
                .Select(i => document.Substring(i * chunkSize, Math.Min(chunkSize, document.Length - i * chunkSize)))
                .ToList();

            Console.WriteLine($"\n--- Step 2: Job Execution (LocalLM) ---");
            Console.WriteLine($"Document split into {chunks.Count} chunks. Running {state.Jobs.Count} jobs per chunk...");

            var sw = Stopwatch.StartNew();
            for (int i = 0; i < chunks.Count; i++)
            {
                foreach (string job in state.Jobs)
                {
                    string prompt = Prompts.LocalWorkerTemplate
                        .Replace("{chunk}", chunks[i])
                        .Replace("{job}", job);

                    state.LocalCharsProcessed += prompt.Length;

                    var response = await slmClient.GetResponseAsync(
                        [new ChatMessage(ChatRole.User, prompt)], cancellationToken: cancellationToken);
                    string result = (response.Text ?? string.Empty).Trim();

                    bool isNone = string.Equals(result, "none", StringComparison.OrdinalIgnoreCase);
                    if (!isNone && !string.IsNullOrEmpty(result))
                    {
                        Console.WriteLine($"  - SUCCESS: Found relevant result in chunk {i + 1}!");
                        state.Results.Add(result);
                    }
                    else
                    {
                        Console.WriteLine($"  - No relevant info found in chunk {i + 1}.");
                    }
                    Console.WriteLine($"    (LocalLM response: '{result}')");
                }
            }
            sw.Stop();
            Console.WriteLine($"Local job execution finished in {sw.Elapsed.TotalSeconds:F2}s.");

            if (state.Results.Count > 0)
            {
                string resultsList = string.Join("\n", state.Results.Select(r => $"- {r}"));
                return
                    $"Original Query: {state.UserQuery}\n\n" +
                    $"Extracted Information:\n{resultsList}\n\n" +
                    "Please provide a final, synthesized answer.";
            }
            return "NO_RESULTS";
        }
    }

    /// <summary>[LLM] Cloud_Synthesizer – aggregates extracted facts into a final answer.</summary>
    sealed class CloudSynthesizerExecutor(IChatClient llmClient) : Executor<string, string>("Cloud_Synthesizer")
    {
        public override async ValueTask<string> HandleAsync(
            string input, IWorkflowContext context, CancellationToken cancellationToken = default)
        {
            Console.WriteLine("\n--- Step 3: Aggregation & Synthesis (RemoteLM) ---");
            Console.Write("🤖 Cloud_Synthesizer: ");

            string fullText = string.Empty;
            await foreach (var update in llmClient.GetStreamingResponseAsync(
                [
                    new ChatMessage(ChatRole.System, Prompts.Synthesizer),
                    new ChatMessage(ChatRole.User, input)
                ], cancellationToken: cancellationToken))
            {
                Console.Write(update.Text);
                fullText += update.Text ?? string.Empty;
            }
            Console.WriteLine();
            return fullText;
        }
    }

    /// <summary>Eval_Formatter – stores synthesizer output and builds the evaluator prompt.</summary>
    sealed class EvalFormatterExecutor(MinionsState state) : Executor<string, string>("Eval_Formatter")
    {
        public override ValueTask<string> HandleAsync(
            string synthesizerOutput, IWorkflowContext context, CancellationToken cancellationToken = default)
        {
            state.FinalAnswer = synthesizerOutput;
            return ValueTask.FromResult(
                $"Evaluate the following AI-generated answer.\n\n" +
                $"Generated Answer:\n{state.FinalAnswer}\n\n" +
                "Provide your evaluation in this exact format:\n" +
                "Score: [1-5]\nReasoning: [Brief explanation]");
        }
    }

    /// <summary>[LLM] Cloud_Evaluator – scores the quality of the synthesized answer.</summary>
    sealed class CloudEvaluatorExecutor(IChatClient llmClient, string document, string userQuery)
        : Executor<string>("Cloud_Evaluator")
    {
        private readonly string _instructions =
            $"""
            You are an expert evaluator. Assess the quality of an AI-generated answer on a scale of 1-5:
            1 = Very Poor  2 = Poor  3 = Fair  4 = Good  5 = Excellent
            Criteria: accuracy, completeness, clarity, relevance.
            Format: Score: [1-5]\nReasoning: [Brief explanation]

            Original Document:
            {document}

            User Query: {userQuery}
            """;

        public override async ValueTask HandleAsync(
            string evalRequest, IWorkflowContext context, CancellationToken cancellationToken = default)
        {
            Console.WriteLine("\n--- Step 4: Response Quality Evaluation (RemoteLM) ---");
            Console.Write("🤖 Cloud_Evaluator: ");

            string evalText = string.Empty;
            await foreach (var update in llmClient.GetStreamingResponseAsync(
                [
                    new ChatMessage(ChatRole.System, _instructions),
                    new ChatMessage(ChatRole.User, evalRequest)
                ], cancellationToken: cancellationToken))
            {
                Console.Write(update.Text);
                evalText += update.Text ?? string.Empty;
            }
            Console.WriteLine();
            await context.YieldOutputAsync(evalText, cancellationToken);
        }
    }
}
