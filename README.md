# Hybrid Local-Remote Agent Framework Demos

A collection of examples showing how to build hybrid agentic workflows using the Microsoft Agent Framework, combining local Small Language Models (SLMs) and cloud-based Large Language Models (LLMs).

These demos illustrate different collaboration patterns to optimize for latency, privacy, and cost without sacrificing performance on complex tasks.

## Collaboration Patterns

| Pattern Name | Description | Paper | Key Concept |
|--------------|-------------|-------|-------------|
| 💻 SLM-Default, LLM-Fallback | Route queries to a local SLM first, escalating to cloud only if the local model's output fails verification. | [arXiv:2510.03847](https://arxiv.org/abs/2510.03847) | Cost & Latency Optimization |
| 💻 Predictive Router | Use a local router to classify queries as "weak" or "strong". Route simple tasks to local models and complex ones to the cloud. | [arXiv:2501.01818](https://arxiv.org/abs/2501.01818) | Dynamic Routing |
| 💻 MAKER Protocol | Decompose complex tasks using a cloud-based "Planner" and execute atomic steps using a local "Voting Solver" with convergence checks. | [arXiv:2511.09030](https://arxiv.org/abs/2511.09030) | Task Decomposition |
| 💻 MINIONS Protocol | Decompose extraction tasks into parallel jobs for local "minions" to process on document chunks, synthesizing results in the cloud. | [arXiv:2502.15964](https://arxiv.org/abs/2502.15964) | Local-Remote Map-Reduce |
| 💻 Chain of Agents | Process long contexts by chaining local SLMs to sequentially build context before final synthesis in the cloud. | [arXiv:2406.02818](https://arxiv.org/abs/2406.02818) | Sequential Bucket Brigade |

---

## Python

> Uses [`agent-framework-mlx`](https://pypi.org/project/agent-framework-mlx/) and is optimised for Apple Silicon (macOS).
> The SLM role is played by **Phi-4-mini-instruct** running locally via MLX.

### Prerequisites

- macOS with Apple Silicon
- Python 3.11+
- Azure CLI logged in (`az login`)

### Setup

```bash
cd python
cp .env.example .env # fill in your variables
pip install -r requirements.txt
```

### Running

```bash
python 01-slm-default-llm-fallback/demo.py
python 02-router-agent/demo.py
python 03-maker/demo.py
python 04-minions/demo.py
python 05-chain-of-agents/demo.py
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `AZURE_AI_PROJECT_ENDPOINT` | Azure AI Foundry project endpoint |
| `AZURE_AI_MODEL_DEPLOYMENT_NAME` | Deployment name for the LLM role in Azure AI Foundry |

---

## .NET

> Uses [`Microsoft.Agents.AI.Workflows`](https://www.nuget.org/packages/Microsoft.Agents.AI.Workflows) (RC4) and [OllamaSharp](https://www.nuget.org/packages/OllamaSharp) for local inference.
> All five patterns are ported 1-to-1 from the Python originals.

The .NET port supports **three interchangeable inference backends**, selected independently for the SLM and LLM roles via environment variables:

| Backend | `SLM_BACKEND` / `LLM_BACKEND` value | Use case |
|---------|--------------------------------------|----------|
| **Ollama** | `ollama` *(default for SLM)* | Local inference via Ollama's native API |
| **OpenAI-compatible** | `openai-compatible` | Any server exposing `/v1/chat/completions` (LM Studio, vLLM, llama.cpp, …) |
| **Azure AI Foundry** | `azure-ai` *(default for LLM)* | Hosted models on Azure AI Foundry via bearer-token auth |

### Prerequisites

- .NET 10 SDK
- At least one of:
  - [Ollama](https://ollama.com) running locally (for the SLM role)
  - An OpenAI-compatible server such as [LM Studio](https://lmstudio.ai) or [vLLM](https://github.com/vllm-project/vllm) (for the SLM role)
  - Azure CLI logged in (`az login`) with an Azure AI Foundry resource (for the LLM role)

### Setup

Each project reads configuration from its `Properties/launchSettings.json`.  A template is provided:

```bash
cd dotnet/src/<project>
cp ../../launchSettings.json.example Properties/launchSettings.json
# edit Properties/launchSettings.json and fill in your values
```

> `Properties/launchSettings.json` is gitignored — your credentials stay local.

### Running

Open `dotnet/HybridAgentDemos.slnx` in Visual Studio / Rider, or run from the CLI:

```bash
dotnet run --project dotnet/src/01-SlmDefaultLlmFallback
dotnet run --project dotnet/src/02-RouterAgent
dotnet run --project dotnet/src/03-Maker
dotnet run --project dotnet/src/04-Minions
dotnet run --project dotnet/src/05-ChainOfAgents
```

### Configuration Reference

All variables are set in `Properties/launchSettings.json` (see `dotnet/launchSettings.json.example`).

**Role selection**

| Variable | Values | Default |
|----------|--------|---------|
| `SLM_BACKEND` | `ollama` \| `openai-compatible` \| `azure-ai` | `ollama` |
| `LLM_BACKEND` | `ollama` \| `openai-compatible` \| `azure-ai` | `azure-ai` |

**Ollama backend**

| Variable | Description | Example |
|----------|-------------|---------|
| `OLLAMA_ENDPOINT` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_SLM_MODEL` | Model name for the SLM role | `phi4-mini` |
| `OLLAMA_LLM_MODEL` | Model name for the LLM role | `llama3.1:8b` |

**OpenAI-compatible backend**

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_COMPATIBLE_ENDPOINT` | Server base URL (without `/v1`) | `http://localhost:1234` |
| `OPENAI_COMPATIBLE_SLM_MODEL` | Model name for the SLM role | `phi-4-mini-instruct` |
| `OPENAI_COMPATIBLE_LLM_MODEL` | Model name for the LLM role | `llama3.1:8b` |

**Azure AI Foundry backend**

| Variable | Description | Example |
|----------|-------------|---------|
| `AZURE_AI_FOUNDRY_ENDPOINT` | Azure AI Foundry OpenAI endpoint | `https://<resource>.ai.azure.com/openai/v1/` |
| `AZURE_AI_SLM_DEPLOYMENT_NAME` | Deployment name for the SLM role | `gpt-4o-mini` |
| `AZURE_AI_LLM_DEPLOYMENT_NAME` | Deployment name for the LLM role | `gpt-4.1` |
