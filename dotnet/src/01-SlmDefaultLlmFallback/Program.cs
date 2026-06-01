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
    // 1. Easy Fact
    "What is the capital of France?",

    // 1b. Tricky Fact
    "In which year was Wisloka Debica founded?",

    // 2. Extraction
    "Convert this list to a JSON shopping list: Apple 2 items, Banana 3 items, Cherries 1 item. Return pure JSON no additional text or formatting.",

    // 3. Ambiguous
    "Where is the city of Springfield located?",

    // 4. Hallucination Trap
    "Explain in 2 sentences the role of quantum healing in modeling proteins.",
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
                "\nIMPORTANT: End response with 'CONFIDENCE: X' (1-10). " +
                "You are allowed to output a score of 8 or higher ONLY IF you are very sure of your answer.";

            Console.Write("   🤖 Local_SLM: ");
            string fullText = string.Empty;
            await foreach (var update in slmClient.GetStreamingResponseAsync(
                [
                    new ChatMessage(ChatRole.System,
                        "You are a helpful assistant. Always end your response with 'CONFIDENCE: X' where X is a number from 1-10 reflecting how confident you are in your answer. If you are sure of your answer, you MUST output a score of 8 or higher."),
                    new ChatMessage(ChatRole.User, prompt)
                ], cancellationToken: cancellationToken))
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
    sealed class CloudLLMExecutor(IChatClient llmClient) : Executor<SLMResult, string>("Cloud_LLM")
    {
        public override async ValueTask<string> HandleAsync(
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
            return fullText;
        }
    }
}
