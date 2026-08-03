using Microsoft.Extensions.AI;
using OpenAI;
using System.ClientModel;

namespace HybridAgentDemos.Shared;

public static class BackendFactory
{
    public static IChatClient CreateSlm()
    {
        var endpoint = Environment.GetEnvironmentVariable("FOUNDRY_LOCAL_ENDPOINT")
            ?? throw new InvalidOperationException("FOUNDRY_LOCAL_ENDPOINT is not set.");
        var model = Environment.GetEnvironmentVariable("FOUNDRY_LOCAL_SLM_MODEL")
            ?? throw new InvalidOperationException("FOUNDRY_LOCAL_SLM_MODEL is not set.");
        return CreateClient($"{endpoint.TrimEnd('/')}/v1", model);
    }

    public static IChatClient CreateLlm()
    {
        var endpoint = Environment.GetEnvironmentVariable("OPENAI_ENDPOINT")
            ?? throw new InvalidOperationException("OPENAI_ENDPOINT is not set.");
        var model = Environment.GetEnvironmentVariable("OPENAI_LLM_MODEL")
            ?? throw new InvalidOperationException("OPENAI_LLM_MODEL is not set.");
        return CreateClient(endpoint, model);
    }

    private static IChatClient CreateClient(string endpoint, string model)
    {
        var options = new OpenAIClientOptions { Endpoint = new Uri(endpoint), };
        var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY")
            ?? "local";
        var client  = new OpenAIClient(new ApiKeyCredential(apiKey), options);
        return client.GetChatClient(model).AsIChatClient();
    }
}
