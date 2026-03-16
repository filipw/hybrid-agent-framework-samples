using Azure.Identity;
using Microsoft.Extensions.AI;
using OpenAI;
using OllamaSharp;
using System.ClientModel;
using System.ClientModel.Primitives;

namespace HybridAgentDemos.Shared;

/// <summary>
/// Creates <see cref="IChatClient"/> instances for the SLM and LLM roles.
///
/// Role assignment is controlled by environment variables:
///   SLM_BACKEND  — "ollama" | "openai-compatible" | "azure-ai"
///   LLM_BACKEND  — "ollama" | "openai-compatible" | "azure-ai"
///
/// Ollama  (native Ollama API via OllamaSharp — uses /api/chat):
///   OLLAMA_ENDPOINT   — default: http://localhost:11434
///   OLLAMA_SLM_MODEL  — e.g. phi4-mini, llama3.2:3b
///   OLLAMA_LLM_MODEL  — e.g. llama3.1:70b
///
/// OpenAI-compatible  (any server that speaks /v1/chat/completions — LM Studio, vLLM, etc.):
///   OPENAI_COMPATIBLE_ENDPOINT   — e.g. http://localhost:1234/v1
///   OPENAI_COMPATIBLE_SLM_MODEL  — model name as the server expects it
///   OPENAI_COMPATIBLE_LLM_MODEL  — model name as the server expects it
///
/// Azure AI Foundry  (OpenAI-compatible endpoint, bearer token auth via `az login`):
///   AZURE_AI_FOUNDRY_ENDPOINT      — e.g. https://&lt;resource&gt;.ai.azure.com/openai/v1/
///   AZURE_AI_SLM_DEPLOYMENT_NAME   — e.g. gpt-4o-mini
///   AZURE_AI_LLM_DEPLOYMENT_NAME   — e.g. gpt-4.1
/// </summary>
public static class BackendFactory
{
    /// <summary>Creates an <see cref="IChatClient"/> for the SLM role.</summary>
    public static IChatClient CreateSlm()
    {
        var backend = Environment.GetEnvironmentVariable("SLM_BACKEND") ?? "ollama";
        return CreateClient(backend, role: "SLM");
    }

    /// <summary>Creates an <see cref="IChatClient"/> for the LLM role.</summary>
    public static IChatClient CreateLlm()
    {
        var backend = Environment.GetEnvironmentVariable("LLM_BACKEND") ?? "azure-ai";
        return CreateClient(backend, role: "LLM");
    }

    private static IChatClient CreateClient(string backend, string role) =>
        backend.ToLowerInvariant() switch
        {
            "ollama"             => CreateOllamaClient(role),
            "openai-compatible"  => CreateOpenAICompatibleClient(role),
            "azure-ai"           => CreateAzureAiClient(role),
            _ => throw new InvalidOperationException(
                $"Unknown backend '{backend}'. Supported values: ollama, openai-compatible, azure-ai")
        };

    // ---------------------------------------------------------------------------
    // Ollama  — native Ollama API (/api/chat) via OllamaSharp
    // ---------------------------------------------------------------------------
    private static IChatClient CreateOllamaClient(string role)
    {
        var endpoint = Environment.GetEnvironmentVariable("OLLAMA_ENDPOINT") ?? "http://localhost:11434";
        var modelVar = $"OLLAMA_{role}_MODEL";
        var model    = Environment.GetEnvironmentVariable(modelVar)
            ?? throw new InvalidOperationException($"{modelVar} is not set.");
        return new OllamaApiClient(new Uri(endpoint), model);
    }

    // ---------------------------------------------------------------------------
    // OpenAI-compatible  — any server that exposes /v1/chat/completions
    // (LM Studio, vLLM, llama.cpp server, LocalAI, …)
    // ---------------------------------------------------------------------------
    private static IChatClient CreateOpenAICompatibleClient(string role)
    {
        var endpoint = Environment.GetEnvironmentVariable("OPENAI_COMPATIBLE_ENDPOINT")
            ?? throw new InvalidOperationException("OPENAI_COMPATIBLE_ENDPOINT is not set.");
        // /v1/ is implicit — append it so callers just set http://localhost:1234
        endpoint = endpoint.TrimEnd('/') + "/v1/";

        var modelVar = $"OPENAI_COMPATIBLE_{role}_MODEL";
        var model    = Environment.GetEnvironmentVariable(modelVar)
            ?? throw new InvalidOperationException($"{modelVar} is not set.");

        var options = new OpenAIClientOptions { Endpoint = new Uri(endpoint) };
        var client  = new OpenAIClient(new ApiKeyCredential("local"), options);
        return client.GetChatClient(model).AsIChatClient();
    }

    // ---------------------------------------------------------------------------
    // Azure AI Foundry  — OpenAI-compatible endpoint, bearer token via `az login`
    // ---------------------------------------------------------------------------
    private static IChatClient CreateAzureAiClient(string role)
    {
        var endpoint      = Environment.GetEnvironmentVariable("AZURE_AI_FOUNDRY_ENDPOINT")
            ?? throw new InvalidOperationException("AZURE_AI_FOUNDRY_ENDPOINT is not set.");
        var deploymentVar = $"AZURE_AI_{role}_DEPLOYMENT_NAME";
        var deployment    = Environment.GetEnvironmentVariable(deploymentVar)
            ?? throw new InvalidOperationException($"{deploymentVar} is not set.");

        var options = new OpenAIClientOptions { Endpoint = new Uri(endpoint) };
        var client  = new OpenAIClient(
            new BearerTokenPolicy(new DefaultAzureCredential(), "https://ai.azure.com/.default"),
            options);
        return client.GetChatClient(deployment).AsIChatClient();
    }
}
