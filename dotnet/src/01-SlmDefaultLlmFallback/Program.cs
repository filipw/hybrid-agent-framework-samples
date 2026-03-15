// =============================================================================
// SLM-Default, LLM-Fallback (Cascade) Pattern
// Based on: arXiv:2510.03847
//
// The local SLM processes every query first and self-reports a confidence
// score (1-10).  If confidence < 8 the workflow cascades to the cloud LLM.
// =============================================================================

using System.Text.RegularExpressions;
using HybridAgentDemos.Shared;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using SlmDefaultLlmFallback;

Console.WriteLine("====================================================");
Console.WriteLine("   Cascade Pattern with Microsoft Agent Framework");
Console.WriteLine("====================================================\n");

var slmClient = BackendFactory.CreateSlm();
var llmClient = BackendFactory.CreateLlm();

var slmExecutor   = new LocalSLMExecutor(slmClient);
var cloudExecutor = new CloudLLMExecutor(llmClient);

Func<object?, bool> shouldFallback = msg => msg is SLMResult r && r.Confidence < 8;

var workflow = new WorkflowBuilder(slmExecutor)
    .AddEdge(slmExecutor, cloudExecutor, shouldFallback)
    .WithOutputFrom(slmExecutor, cloudExecutor)
    .Build();

string[] queries =
[
    "What is the capital of France?",
    "Convert this list to a JSON array: Apple, Banana, Cherry",
    "Where is the city of Springfield located?",
    "Explain in 2 sentences the role of quantum healing in modeling proteins.",
    "If I have a cabbage, a goat, and a wolf, and I need to cross a river " +
    "but can only take one item at a time, and I can't leave the goat with " +
    "the cabbage or the wolf with the goat, how do I do it?",
];

foreach (var query in queries)
{
    Console.WriteLine($"\n❔ Query: {query}");
    Console.WriteLine(new string('-', 40));

    await using var run = await InProcessExecution.RunStreamingAsync(workflow, query);
    await foreach (var evt in run.WatchStreamAsync())
        if (evt is WorkflowErrorEvent err) throw err.Exception!;

    Console.WriteLine();
}

// ── Types & Executors ────────────────────────────────────────────────────────

namespace SlmDefaultLlmFallback
{
    /// <summary>Carries the SLM's response together with its self-reported confidence.</summary>
    record SLMResult(string OriginalQuery, string Response, int Confidence);

    /// <summary>[SLM] Local_SLM – runs the smaller model and injects a confidence prompt.</summary>
    sealed class LocalSLMExecutor(IChatClient slmClient) : Executor<string, SLMResult>("Local_SLM")
    {
        public override async ValueTask<SLMResult> HandleAsync(
            string query, IWorkflowContext context, CancellationToken cancellationToken = default)
        {
            string prompt =
                query +
                "\nIMPORTANT: End your response with 'CONFIDENCE: X' (1-10). " +
                "If you are sure of your answer, you MUST output a score of 8 or higher.";

            Console.Write("   🤖 Local_SLM: ");
            string fullText = string.Empty;
            await foreach (var update in slmClient.GetStreamingResponseAsync(
                [new ChatMessage(ChatRole.User, prompt)], cancellationToken: cancellationToken))
            {
                Console.Write(update.Text);
                fullText += update.Text ?? string.Empty;
            }
            Console.WriteLine();

            var match      = Regex.Match(fullText, @"CONFIDENCE:\s*(\d+)", RegexOptions.IgnoreCase);
            int confidence = match.Success ? int.Parse(match.Groups[1].Value) : 0;

            Console.WriteLine($"\n   📊 Verifier Score: {confidence}/10");

            var result = new SLMResult(query, fullText, confidence);

            if (confidence >= 8)
            {
                Console.WriteLine("   ✅ High Confidence. Workflow Complete.");
                await context.YieldOutputAsync(result, cancellationToken);
            }
            else
            {
                Console.WriteLine("   ⚠️  Low Confidence. Routing to Cloud...");
            }

            return result;
        }
    }

    /// <summary>[LLM] Cloud_LLM – fallback for low-confidence SLM responses.</summary>
    sealed class CloudLLMExecutor(IChatClient llmClient) : Executor<SLMResult>("Cloud_LLM")
    {
        public override async ValueTask HandleAsync(
            SLMResult slmResult, IWorkflowContext context, CancellationToken cancellationToken = default)
        {
            Console.Write("   🤖 Cloud_LLM: ");
            string fullText = string.Empty;
            await foreach (var update in llmClient.GetStreamingResponseAsync(
                [
                    new ChatMessage(ChatRole.System,
                        "You are a fallback expert. The previous assistant was unsure. Provide a complete answer."),
                    new ChatMessage(ChatRole.User, slmResult.OriginalQuery)
                ], cancellationToken: cancellationToken))
            {
                Console.Write(update.Text);
                fullText += update.Text ?? string.Empty;
            }
            Console.WriteLine();
            await context.YieldOutputAsync(fullText, cancellationToken);
        }
    }
}
