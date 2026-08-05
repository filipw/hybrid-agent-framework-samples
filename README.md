# Hybrid Local-Remote Agent Framework Demos

A collection of examples showing how to build hybrid agentic workflows using the Microsoft Agent Framework, combining local Small Language Models (SLMs) and cloud-based Large Language Models (LLMs).

These demos illustrate different collaboration patterns to optimize for latency, privacy, and cost without sacrificing performance on complex tasks.

## Collaboration Patterns

| Pattern Name | Description | Paper | Key Concept |
|--------------|-------------|-------|-------------|
| 💻 SLM-Default, LLM-Fallback | Route queries to a local SLM first, escalating to cloud only if the local model's output fails verification. | [arXiv:2510.03847](https://arxiv.org/abs/2510.03847) | Cost & Latency Optimization |
| 💻 Predictive Router | Use a local router to classify queries as "weak" or "strong". Route simple tasks to local models and complex ones to the cloud. | [arXiv:2406.18665](https://arxiv.org/abs/2406.18665) | Dynamic Routing |
| 💻 MAKER Protocol | Decompose complex tasks using a cloud-based "Planner" and execute atomic steps using a local "Voting Solver" with convergence checks. | [arXiv:2511.09030](https://arxiv.org/abs/2511.09030) | Task Decomposition |
| 💻 MINIONS Protocol | Decompose extraction tasks into parallel jobs for local "minions" to process on document chunks, synthesizing results in the cloud. | [arXiv:2502.15964](https://arxiv.org/abs/2502.15964) | Local-Remote Map-Reduce |
| 💻 Chain of Agents | Process long contexts by chaining local SLMs to sequentially build context before final synthesis in the cloud. | [arXiv:2406.02818](https://arxiv.org/abs/2406.02818) | Sequential Bucket Brigade |

---

## Python

> The SLM role is played by **Phi-4-mini-instruct** running locally.
> Two interchangeable local inference backends are supported, selected via the `LOCAL_BACKEND` environment variable:

| Backend | `LOCAL_BACKEND` value | Use case |
|---------|-----------------------|----------|
| **MLX** | `mlx` *(default)* | Apple Silicon (macOS) via [`agent-framework-mlx`](https://pypi.org/project/agent-framework-mlx/) |
| **Foundry Local** | `foundry_local` | Cross-platform (Windows, macOS, Linux) via [Foundry Local](https://www.foundrylocal.ai) |

Demos use short model alias names (e.g. `phi-4-mini`) that are automatically resolved to the correct backend-specific model path. You can override the model with the `LOCAL_MODEL_PATH` env var.

### Prerequisites

- Python 3.11+
- Azure CLI logged in (`az login`)
- For the MLX backend: macOS with Apple Silicon
- For the Foundry Local backend: any platform; install via `brew install microsoft/foundrylocal/foundrylocal` (macOS) or see the [Foundry Local docs](https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-local/get-started)

### Setup

```bash
cd python
cp .env.example .env # fill in your variables

# agent-framework-mlx pins an older agent-framework-core in its own metadata, which
# conflicts with the agent-framework version pinned in requirements.txt. Install
# everything else first, then install agent-framework-mlx separately with --no-deps
# (its real runtime deps, mlx/mlx-lm, are already installed via requirements.txt).
# On non-Apple-Silicon platforms, skip agent-framework-mlx and use LOCAL_BACKEND=foundry_local instead.
grep -v -E '^agent-framework-mlx==' requirements.txt | pip install -r /dev/stdin
pip install --no-deps agent-framework-mlx==0.6.0
```

### Running

```bash
# default (MLX backend, Apple Silicon)
python 01-slm-default-llm-fallback/demo.py

# use Foundry Local backend (cross-platform)
LOCAL_BACKEND=foundry_local python 01-slm-default-llm-fallback/demo.py
```

All five demos follow the same pattern:

```bash
python 01-slm-default-llm-fallback/demo.py
python 02-router-agent/demo.py
python 03-maker/demo.py
python 04-minions/demo.py
python 05-chain-of-agents/demo.py
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AZURE_AI_PROJECT_ENDPOINT` | Azure AI Foundry project endpoint | |
| `AZURE_AI_MODEL_DEPLOYMENT_NAME` | Deployment name for the LLM role in Azure AI Foundry | |
| `LOCAL_BACKEND` | Local inference backend (`mlx` or `foundry_local`) | `mlx` |
| `LOCAL_MODEL_PATH` | Override the model alias or path for the SLM | `phi-4-4bit` |

---

## .NET

> Uses [`Microsoft.Agents.AI.Workflows`](https://www.nuget.org/packages/Microsoft.Agents.AI.Workflows) for orchestration.
> All five patterns are ported 1-to-1 from the Python originals.

The .NET port uses **one local backend, [Foundry Local](https://www.foundrylocal.ai)**, for the SLM role, and any **OpenAI-compatible endpoint** (Azure OpenAI or OpenAI) for the LLM role.

### Prerequisites

- .NET 10 SDK
- [Foundry Local](https://www.foundrylocal.ai) running locally with a model loaded for the SLM role — install via `brew install microsoft/foundrylocal/foundrylocal` (macOS) or see the [Foundry Local docs](https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-local/get-started)
- An OpenAI-compatible endpoint and API key for the LLM role (e.g. an Azure OpenAI deployment or OpenAI itself)

### Setup

Configuration is read from plain process environment variables (not `launchSettings.json`). Create a `.env` file in `dotnet/src/` (gitignored) with your values:

```bash
export OPENAI_ENDPOINT="https://<resource>.openai.azure.com/openai/v1"
export OPENAI_API_KEY="<your-api-key>"
export OPENAI_LLM_MODEL="<deployment-or-model-name>"

export FOUNDRY_LOCAL_ENDPOINT="http://127.0.0.1:<port>"
export FOUNDRY_LOCAL_SLM_MODEL="phi-4-mini"
```

Then source it in your shell before running any demo:

```bash
cd dotnet/src
source .env
```

### Running

Open `dotnet/HybridAgentDemos.slnx` in Visual Studio / Rider, or run from the CLI (after sourcing `.env` as above):

```bash
dotnet run --project dotnet/src/01-SlmDefaultLlmFallback
dotnet run --project dotnet/src/02-RouterAgent
dotnet run --project dotnet/src/03-Maker
dotnet run --project dotnet/src/04-Minions
dotnet run --project dotnet/src/05-ChainOfAgents
```

### Environment Variables

| Variable | Description | Used for |
|----------|-------------|----------|
| `OPENAI_ENDPOINT` | Base URL (including `/openai/v1` for Azure OpenAI, or `/v1` for OpenAI) of an OpenAI-compatible API | LLM role |
| `OPENAI_API_KEY` | API key for the endpoint above | LLM role |
| `OPENAI_LLM_MODEL` | Model or deployment name | LLM role |
| `FOUNDRY_LOCAL_ENDPOINT` | Foundry Local server URL (no `/v1` suffix) | SLM role |
| `FOUNDRY_LOCAL_SLM_MODEL` | Model alias loaded in Foundry Local | SLM role |
