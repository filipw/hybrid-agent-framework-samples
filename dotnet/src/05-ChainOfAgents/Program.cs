// =============================================================================
// Chain of Agents (CoA) Pattern
// Based on: arXiv:2406.02818
//
// The document is split into small chunks (2 lines each).  Each chunk is
// assigned to a sequential Worker (local SLM).  Workers pass a running
// "Communication Unit" (CU) to the next worker.  Each worker reads its chunk
// + the previous CU and outputs an updated CU.  The final cloud LLM Manager
// receives the complete CU and synthesises the final answer.
// =============================================================================

using HybridAgentDemos.Shared;
using ChainOfAgents;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

Console.WriteLine("===============================================================");
Console.WriteLine("   Chain of Agents (CoA) Pattern (arXiv:2406.02818)");
Console.WriteLine("===============================================================\n");

string textFilePath = Path.Combine(AppContext.BaseDirectory, "quantum_mechanics_history.txt");
string fullText     = File.ReadAllText(textFilePath);

// Split into 2-line chunks
const int LinesPerChunk = 2;
var lines = fullText.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
var documentChunks = Enumerable.Range(0, (lines.Length + LinesPerChunk - 1) / LinesPerChunk)
    .Select(i => string.Join('\n', lines.Skip(i * LinesPerChunk).Take(LinesPerChunk)))
    .ToList();

string query = "How did quantum mechanics evolve from Planck's initial hypothesis to a complete mathematical framework? Trace the key contributors and what each one added.";

Console.WriteLine($"❔ Query: {query}");
Console.WriteLine($"📄 Document split into {documentChunks.Count} sequential chunks.\n");

var slmClient = BackendFactory.CreateSlm();
var llmClient = BackendFactory.CreateLlm();

// Build sequential chain: Worker_1 → Worker_2 → ... → Worker_N → Cloud_Manager
var workers = documentChunks
    .Select((chunk, i) => new WorkerExecutor(
        slmClient, query, chunk,
        workerIdx: i + 1, totalWorkers: documentChunks.Count,
        name: $"Worker_{i + 1}"))
    .ToList();

var manager = new CloudManagerExecutor(llmClient, query);

var builder = new WorkflowBuilder(workers[0]);
for (int i = 0; i < workers.Count - 1; i++)
    builder.AddEdge(workers[i], workers[i + 1]);
builder.AddEdge(workers[^1], manager);
builder.WithOutputFrom(manager);

var workflow = builder.Build();

Console.WriteLine("🚀 Starting Chain...\n");

// Initial CU is empty (paper Algorithm 1: CU₀ ← empty string)
await using var run = await InProcessExecution.RunStreamingAsync(workflow, string.Empty);
await foreach (var evt in run.WatchStreamAsync())
    if (evt is WorkflowErrorEvent err) throw err.Exception!;

Console.WriteLine("\n\n✅ Workflow Complete.");

namespace ChainOfAgents
{
    /// <summary>
    /// [SLM] Worker - reads one chunk and updates the Communication Unit (CU).
    /// CU is truncated to 1500 chars to respect context budget (paper MAX_CU_CHARS).
    /// </summary>
    sealed class WorkerExecutor(
        IChatClient slmClient,
        string query,
        string chunk,
        int workerIdx,
        int totalWorkers,
        string name) : Executor<string, string>(name)
    {
        private const int MaxCuChars = 1500;

        private static readonly ChatOptions GenerationOptions = new() { Temperature = 0f, MaxOutputTokens = 700 };

        public override async ValueTask<string> HandleAsync(
            string previousCu, IWorkflowContext context, CancellationToken cancellationToken = default)
        {
            string cu = previousCu.Trim();
            if (cu.Length > MaxCuChars) cu = "..." + cu[^(MaxCuChars - 3)..];

            string accumulated = string.IsNullOrEmpty(cu) ? "(none yet — this is the first chunk)" : cu;

            // Clearly delimited sections (accumulated knowledge / new chunk / task.
            // Explicitly forbids answering yet
            string prompt =
                $"""
                ### ACCUMULATED KNOWLEDGE SO FAR
                {accumulated}

                ### NEW SOURCE TEXT (chunk {workerIdx} of {totalWorkers})
                {chunk}

                ### TASK
                Extract every fact from NEW SOURCE TEXT that is relevant to this question: "{query}"
                Merge those facts into ACCUMULATED KNOWLEDGE, keeping everything already there.

                RULES:
                - Use ONLY facts stated in NEW SOURCE TEXT or ACCUMULATED KNOWLEDGE. Never use outside/prior knowledge.
                - Never drop or summarize away anything already in ACCUMULATED KNOWLEDGE.
                - Do NOT answer the question yet — only accumulate knowledge for a later step.
                - Output ONLY the updated ACCUMULATED KNOWLEDGE as plain factual sentences. No headers, no commentary.
                """;

            var response = await slmClient.GetResponseAsync(
                [new ChatMessage(ChatRole.User, prompt)], GenerationOptions, cancellationToken);
            string outputCu = (response.Text ?? string.Empty).Trim();

            Console.WriteLine($"\n   [{this.Id} ({workerIdx}/{totalWorkers})] CU length: {outputCu.Length} chars");
            Console.WriteLine($"   {new string('-', 60)}\n   {outputCu}\n   {new string('-', 60)}");

            return outputCu;
        }
    }

    /// <summary>
    /// [LLM] Cloud_Manager - receives the final CU and answers the query.
    /// </summary>
    sealed class CloudManagerExecutor(IChatClient llmClient, string query) : Executor<string, string>("Cloud_Manager")
    {
        public override async ValueTask<string> HandleAsync(
            string finalCu, IWorkflowContext context, CancellationToken cancellationToken = default)
        {
            string prompt =
                "The following are given passages. However, the source text is too long " +
                "and has been summarized. You need to answer based on the summary:\n\n" +
                $"{finalCu}\n\n" +
                $"Question: {query}\n\n" +
                "Answer:";

            Console.WriteLine("\n\n   ☁️  Cloud_Manager:\n   ");
            string fullText = string.Empty;
            await foreach (var update in llmClient.GetStreamingResponseAsync(
                [new ChatMessage(ChatRole.User, prompt)], cancellationToken: cancellationToken))
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
