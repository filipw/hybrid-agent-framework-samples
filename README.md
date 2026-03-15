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
cp ../.env.example .env   # fill in your Azure OpenAI values
pip install -r requirements.txt
```

### Running

```bash
python slm-default-llm-fallback/demo.py
python router-agent/demo.py
python maker/demo.py
python minions/demo.py
python chain-of-agents/demo.py
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `AZURE_OPENAI_RESOURCE` | Azure OpenAI resource name |
| `AZURE_OPENAI_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Chat completion deployment name (LLM role) |

---

## .NET

> Uses [`Microsoft.Agents.AI.Workflows`](https://www.nuget.org/packages/Microsoft.Agents.AI.Workflows) (RC4).
> All five patterns are ported 1-to-1 from the Python originals.

### Note on the SLM Placeholder

The Python demos run **Phi-4-mini-instruct** locally via MLX.  No equivalent local-inference library exists for .NET yet.  As a temporary stand-in, the **SLM role** is played by a *smaller* Azure OpenAI deployment (e.g. `gpt-4o-mini`) configured via `AZURE_OPENAI_SLM_DEPLOYMENT`.  The interaction structure – confidence injection, routing decisions, voting loops, sequential chaining – is **identical** to the Python version.  Once a local .NET SLM path is available it will replace this placeholder.

### Prerequisites

- .NET 9 SDK
- Azure CLI logged in (`az login`)

### Setup

```bash
cd dotnet
cp .env.example .env   # fill in your Azure OpenAI values
# set environment variables (or use a .env loader)
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_LLM_DEPLOYMENT=gpt-4o
export AZURE_OPENAI_SLM_DEPLOYMENT=gpt-4o-mini
```

### Running

Open `dotnet/HybridAgentDemos.slnx` in Visual Studio or run from the CLI:

```bash
dotnet run --project dotnet/src/01-SlmDefaultLlmFallback
dotnet run --project dotnet/src/02-RouterAgent
dotnet run --project dotnet/src/03-Maker
dotnet run --project dotnet/src/04-Minions
dotnet run --project dotnet/src/05-ChainOfAgents
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Full Azure OpenAI endpoint URL |
| `AZURE_OPENAI_LLM_DEPLOYMENT` | Larger model deployment (LLM role, e.g. `gpt-4o`) |
| `AZURE_OPENAI_SLM_DEPLOYMENT` | Smaller model deployment (SLM placeholder, e.g. `gpt-4o-mini`) |

### Project Structure

```
dotnet/
├── HybridAgentDemos.slnx
└── src/
    ├── 01-SlmDefaultLlmFallback/   # Cascade pattern
    ├── 02-RouterAgent/             # Predictive router
    ├── 03-Maker/                   # MAKER protocol
    ├── 04-Minions/                 # MINIONS map-reduce
    └── 05-ChainOfAgents/           # Chain of Agents
```
