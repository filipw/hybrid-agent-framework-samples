// =============================================================================
// Predictive Router Pattern
// Based on: arXiv:2501.01818
//
// A dedicated router classifies each query as WEAK (simple/factual) or
// STRONG (complex/reasoning).  WEAK queries go to the local SLM; STRONG
// queries go to the cloud LLM.
//
// Backend configuration (see dotnet/.env.example):
//   SLM_BACKEND  — inference backend for the SLM role (default: ollama)
//   LLM_BACKEND  — inference backend for the LLM role (default: azure-openai)
// =============================================================================

using HybridAgentDemos.Shared;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using RouterAgent;

// ── Main ─────────────────────────────────────────────────────────────────────

Console.WriteLine("====================================================");
Console.WriteLine(" Predictive Router Pattern (arXiv:2501.01818)");
Console.WriteLine("====================================================\n");

var routerClient = BackendFactory.CreateSlm();
var slmClient    = BackendFactory.CreateSlm();
var llmClient    = BackendFactory.CreateLlm();

var routerExecutor = new RouterExecutor(routerClient);
var weakWorker     = new WeakModelWorker(slmClient);
var strongWorker   = new StrongModelWorker(llmClient);

Func<object?, bool> toStrong = msg => msg is RouterDecision d && d.IsStrong;
Func<object?, bool> toWeak   = msg => msg is RouterDecision d && !d.IsStrong;

var workflow = new WorkflowBuilder(routerExecutor)
    .AddEdge(routerExecutor, strongWorker, toStrong)
    .AddEdge(routerExecutor, weakWorker,   toWeak)
    .WithOutputFrom(weakWorker, strongWorker)
    .Build();

string[] queries =
[
    "Explain the implications of quantum computing on cryptography",
    "What are the three primary colors?",
];

foreach (var query in queries)
{
    Console.WriteLine($"\n❔ Query: {query}");
    Console.WriteLine(new string('-', 50));

    await using var run = await InProcessExecution.RunStreamingAsync(workflow, query);
    await foreach (var evt in run.WatchStreamAsync())
        if (evt is WorkflowErrorEvent err) throw err.Exception!;

    Console.WriteLine("\n" + new string('=', 50));
}

// ── Types & Executors ────────────────────────────────────────────────────────

namespace RouterAgent
{
    /// <summary>Decision produced by the router; carries the original query to the chosen worker.</summary>
    record RouterDecision(string OriginalQuery, bool IsStrong);

    /// <summary>
    /// [SLM] Router_Control_Plane – few-shot classifier at low temperature.
    /// </summary>
    sealed class RouterExecutor(IChatClient routerClient) : Executor<string, RouterDecision>("Router_Control_Plane")
    {
        private const string RouterInstructions = """
            You are a high-precision query classifier.
            - 'ROUTE: WEAK': For simple facts, formatting, summaries, or questions with obvious answers.
            - 'ROUTE: STRONG': For reasoning, coding, creative writing, analysis, or complex multi-step tasks.

            EXAMPLES:
            Input: "What is the capital of France?" → Output: ROUTE: WEAK
            Input: "Write a Python script to parse a CSV and plot the data." → Output: ROUTE: STRONG
            Input: "Summarize this short text." → Output: ROUTE: WEAK
            Input: "Explain the implications of quantum computing on modern cryptography." → Output: ROUTE: STRONG

            Output ONLY 'ROUTE: WEAK' or 'ROUTE: STRONG'. Do not answer the query.
            """;

        public override async ValueTask<RouterDecision> HandleAsync(
            string query, IWorkflowContext context, CancellationToken cancellationToken = default)
        {
            var options  = new ChatOptions { Temperature = 0.1f, MaxOutputTokens = 10 };
            var response = await routerClient.GetResponseAsync(
                [
                    new ChatMessage(ChatRole.System, RouterInstructions),
                    new ChatMessage(ChatRole.User, $"Input: \"{query}\"\nOutput:")
                ],
                options, cancellationToken);

            bool isStrong = (response.Text ?? string.Empty).Contains("ROUTE: STRONG");
            Console.WriteLine(isStrong
                ? "   [🔀 Decision]: STRONG (Complex Query -> Cloud LLM)"
                : "   [🔀 Decision]: WEAK   (Simple/Factual -> Local SLM)");

            return new RouterDecision(query, isStrong);
        }
    }

    /// <summary>
    /// [SLM] Weak_Model_Worker – handles simple queries using the local model.
    /// </summary>
    sealed class WeakModelWorker(IChatClient slmClient) : Executor<RouterDecision, string>("Weak_Model_Worker")
    {
        public override async ValueTask<string> HandleAsync(
            RouterDecision decision, IWorkflowContext context, CancellationToken cancellationToken = default)
        {
            Console.Write(" 🤖 Weak_Model_Worker [Local SLM]: ");
            string fullText = string.Empty;
            await foreach (var update in slmClient.GetStreamingResponseAsync(
                [
                    new ChatMessage(ChatRole.System, "You are a concise assistant. Answer the user's question directly."),
                    new ChatMessage(ChatRole.User, decision.OriginalQuery)
                ], cancellationToken: cancellationToken))
            {
                Console.Write(update.Text);
                fullText += update.Text ?? string.Empty;
            }
            Console.WriteLine();
            await context.YieldOutputAsync(fullText, cancellationToken);
            return fullText;
        }
    }

    /// <summary>
    /// [LLM] Strong_Model_Worker – handles complex queries using the cloud LLM.
    /// </summary>
    sealed class StrongModelWorker(IChatClient llmClient) : Executor<RouterDecision, string>("Strong_Model_Worker")
    {
        public override async ValueTask<string> HandleAsync(
            RouterDecision decision, IWorkflowContext context, CancellationToken cancellationToken = default)
        {
            Console.Write(" 🤖 Strong_Model_Worker [Cloud LLM]: ");
            string fullText = string.Empty;
            await foreach (var update in llmClient.GetStreamingResponseAsync(
                [
                    new ChatMessage(ChatRole.System, "You are an expert assistant. Provide detailed, reasoning-heavy answers."),
                    new ChatMessage(ChatRole.User, decision.OriginalQuery)
                ], cancellationToken: cancellationToken))
            {
                Console.Write(update.Text);
                fullText += update.Text ?? string.Empty;
            }
            Console.WriteLine();
            await context.YieldOutputAsync(fullText, cancellationToken);
            return fullText;
        }
    }
}
