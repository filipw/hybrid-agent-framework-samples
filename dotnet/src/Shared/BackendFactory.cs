using Microsoft.Extensions.AI;
using OpenAI;
using System.ClientModel;

namespace HybridAgentDemos.Shared;

/// <summary>
/// Creates <see cref="IChatClient"/> instances for the SLM and LLM roles.
///
/// Both roles use the Foundry Local backend (OpenAI-compatible /v1/chat/completions):
///   FOUNDRY_LOCAL_ENDPOINT   — server URL (default: http://localhost:5272)
///   FOUNDRY_LOCAL_SLM_MODEL  — model alias or name for the SLM role
///   FOUNDRY_LOCAL_LLM_MODEL  — model alias or name for the LLM role
/// </summary>
public static class BackendFactory
{
    private static readonly string Endpoint =
        (Environment.GetEnvironmentVariable("FOUNDRY_LOCAL_ENDPOINT") ?? "http://localhost:5272")
        .TrimEnd('/') + "/v1/";

    /// <summary>Creates an <see cref="IChatClient"/> for the SLM role.</summary>
    public static IChatClient CreateSlm()
    {
        var model = Environment.GetEnvironmentVariable("FOUNDRY_LOCAL_SLM_MODEL")
            ?? throw new InvalidOperationException("FOUNDRY_LOCAL_SLM_MODEL is not set.");
        return CreateClient(model);
    }

    /// <summary>Creates an <see cref="IChatClient"/> for the LLM role.</summary>
    public static IChatClient CreateLlm()
    {
        var model = Environment.GetEnvironmentVariable("FOUNDRY_LOCAL_LLM_MODEL")
            ?? throw new InvalidOperationException("FOUNDRY_LOCAL_LLM_MODEL is not set.");
        return CreateClient(model);
    }

    private static IChatClient CreateClient(string model)
    {
        var options = new OpenAIClientOptions { Endpoint = new Uri(Endpoint) };
        var client  = new OpenAIClient(new ApiKeyCredential("local"), options);
        return client.GetChatClient(model).AsIChatClient();
    }
}
