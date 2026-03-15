using Azure.Identity;
using Microsoft.Extensions.AI;
using OpenAI;
using OllamaSharp;
using System.ClientModel.Primitives;

namespace HybridAgentDemos.Shared;

/// <summary>
/// Creates <see cref="IChatClient"/> instances for the SLM and LLM roles.
///
/// Role assignment is controlled by environment variables:
///   SLM_BACKEND  — "ollama" (default) | "azure-ai"
///   LLM_BACKEND  — "ollama" | "azure-ai" (default)
///
/// Ollama config (local inference):
///   OLLAMA_ENDPOINT   — base URL of the Ollama server (default: http://localhost:11434)
///   OLLAMA_SLM_MODEL  — model name for the SLM role  (e.g. "phi4-mini", "llama3.2:3b")
///   OLLAMA_LLM_MODEL  — model name for the LLM role  (e.g. "llama3.1:70b", "mistral-large")
///
/// Azure AI Foundry config:
///   AZURE_AI_FOUNDRY_ENDPOINT      — OpenAI-compatible endpoint, e.g. https://&lt;resource&gt;.ai.azure.com/openai/v1/
///   AZURE_AI_SLM_DEPLOYMENT_NAME   — deployment name for the SLM role, e.g. gpt-4o-mini
///   AZURE_AI_LLM_DEPLOYMENT_NAME   — deployment name for the LLM role, e.g. gpt-4.1
///
/// Authentication for Azure AI: run `az login` first.
/// </summary>
public static class BackendFactory
{
    /// <summary>Creates an <see cref="IChatClient"/> for the local SLM role.</summary>
    public static IChatClient CreateSlm()
    {
        var backend = Environment.GetEnvironmentVariable("SLM_BACKEND") ?? "ollama";
        return CreateClient(backend, role: "SLM");
    }

    /// <summary>Creates an <see cref="IChatClient"/> for the cloud LLM role.</summary>
    public static IChatClient CreateLlm()
    {
        var backend = Environment.GetEnvironmentVariable("LLM_BACKEND") ?? "azure-ai";
        return CreateClient(backend, role: "LLM");
    }

    private static IChatClient CreateClient(string backend, string role) =>
        backend.ToLowerInvariant() switch
        {
            "ollama"   => CreateOllamaClient(role),
            "azure-ai" => CreateAzureAiClient(role),
            _ => throw new InvalidOperationException(
                $"Unknown backend '{backend}'. Supported values: ollama, azure-ai")
        };

    // ---------------------------------------------------------------------------
    // Ollama  — local inference via OllamaSharp (implements IChatClient directly)
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
    // Azure AI Foundry  — OpenAI-compatible endpoint with bearer token auth.
    // SLM and LLM roles use separate deployment names.
    // ---------------------------------------------------------------------------
    private static IChatClient CreateAzureAiClient(string role)
    {
        var endpoint       = Environment.GetEnvironmentVariable("AZURE_AI_FOUNDRY_ENDPOINT")
            ?? throw new InvalidOperationException("AZURE_AI_FOUNDRY_ENDPOINT is not set.");
        var deploymentVar  = $"AZURE_AI_{role}_DEPLOYMENT_NAME";
        var deployment     = Environment.GetEnvironmentVariable(deploymentVar)
            ?? throw new InvalidOperationException($"{deploymentVar} is not set.");

        var options = new OpenAIClientOptions { Endpoint = new Uri(endpoint) };
        var client  = new OpenAIClient(
            new BearerTokenPolicy(new DefaultAzureCredential(), "https://ai.azure.com/.default"),
            options);

        return client.GetChatClient(deployment).AsIChatClient();
    }
}
