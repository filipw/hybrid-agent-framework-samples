// =============================================================================
// Chain of Agents (CoA) Pattern
// Based on: arXiv:2406.02818
//
// The document is split into small chunks (2 lines each).  Each chunk is
// assigned to a sequential Worker (local SLM).  Workers pass a running
// "Communication Unit" (CU) to the next worker.  Each worker reads its chunk
// + the previous CU and outputs an updated CU.  The final cloud LLM Manager
// receives the complete CU and synthesises the final answer.
//
// Backend configuration (see dotnet/.env.example):
//   SLM_BACKEND  — inference backend for the SLM role (default: ollama)
//   LLM_BACKEND  — inference backend for the LLM role (default: azure-openai)
// =============================================================================

using HybridAgentDemos.Shared;
using ChainOfAgents;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

// ── Main ─────────────────────────────────────────────────────────────────────

Console.WriteLine("===============================================================");
Console.WriteLine("   Chain of Agents (CoA) Pattern (arXiv:2406.02818)");
Console.WriteLine("===============================================================\n");

string textFilePath = Path.Combine(AppContext.BaseDirectory, "security_logs.txt");
string fullText     = File.ReadAllText(textFilePath);

// Split into 2-line chunks (mirroring the Python demo)
var lines = fullText.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
var documentChunks = Enumerable.Range(0, (lines.Length + 1) / 2)
    .Select(i => string.Join('\n', lines.Skip(i * 2).Take(2)))
    .ToList();

string query = "Create a brief chronological timeline of the ransomware attack and its resolution. Include the root cause.";

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

// ── Types & Executors ────────────────────────────────────────────────────────

namespace ChainOfAgents
{
    /// <summary>
    /// [SLM] Worker – reads one chunk and updates the Communication Unit (CU).
    /// CU is truncated to 1500 chars to respect context budget (paper MAX_CU_CHARS).
    /// </summary>
    sealed class WorkerExecutor(
        IChatClient slmClient,
        string query,
        string chunk,
        int workerIdx,
        int totalWorkers,
        string name) : Executor<string>(name)
    {
        private const int MaxCuChars = 1500;

        public override async ValueTask HandleAsync(
            string previousCu, IWorkflowContext context, CancellationToken cancellationToken = default)
        {
            string cu = previousCu.Trim();
            if (cu.Length > MaxCuChars) cu = "..." + cu[^(MaxCuChars - 3)..];

            string cuSection = string.IsNullOrEmpty(cu)
                ? "There is no previous summary yet — this is the first chunk."
                : $"Here is the summary of the previous source text: {cu}";

            string prompt =
                $"{chunk}\n\n" +
                $"{cuSection}\n\n" +
                $"Question that will be answered later: {query}\n\n" +
                "Summarize ALL events from the current source text together with the previous summary. " +
                "Include every event with its timestamp and details — do not skip events even if they " +
                "seem unrelated to the question. Do NOT invent or infer any events not explicitly stated. " +
                "Output only the updated factual summary, 3-5 sentences, no commentary.";

            var response = await slmClient.GetResponseAsync(
                [new ChatMessage(ChatRole.User, prompt)], cancellationToken: cancellationToken);
            string outputCu = (response.Text ?? string.Empty).Trim();

            Console.WriteLine($"\n   [{name} ({workerIdx}/{totalWorkers})] CU length: {outputCu.Length} chars");
            Console.WriteLine($"   {new string('-', 60)}\n   {outputCu}\n   {new string('-', 60)}");

            await context.SendMessageAsync(outputCu, cancellationToken: cancellationToken);
        }
    }

    /// <summary>
    /// [LLM] Cloud_Manager – receives the final CU and answers the query.
    /// </summary>
    sealed class CloudManagerExecutor(IChatClient llmClient, string query) : Executor<string>("Cloud_Manager")
    {
        public override async ValueTask HandleAsync(
            string finalCu, IWorkflowContext context, CancellationToken cancellationToken = default)
        {
            string instructions =
                "You are the Manager Agent in a Chain of Agents workflow. " +
                "Worker agents read the source text in chunks and produced the summary below. " +
                "Treat this summary as your source material and use it to answer the question directly.\n\n" +
                $"Question: {query}";

            var messages = new List<ChatMessage>
            {
                new(ChatRole.System, instructions),
                new(ChatRole.User, finalCu)
            };

            Console.WriteLine("\n\n   ☁️  Cloud_Manager:\n   ");
            string fullText = string.Empty;
            await foreach (var update in llmClient.GetStreamingResponseAsync(messages, cancellationToken: cancellationToken))
            {
                Console.Write(update.Text);
                fullText += update.Text ?? string.Empty;
            }
            Console.WriteLine();
            await context.YieldOutputAsync(fullText, cancellationToken);
        }
    }
}
