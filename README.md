# Hybrid Local-Remote Agent Framework Demos

A collection of examples showing how to build hybrid agentic workflows using the Microsoft Agent Framework, combining local Small Language Models (SLMs) and cloud-based Large Language Models (LLMs).

These demos illustrate different collaboration patterns to optimize for latency, privacy, and cost without sacrificing performance on complex tasks.

## Collaboration Patterns

| Pattern Name | Description | Source Code | Paper | Key Concept |
|--------------|-------------|-------------|-------|-------------|
| 💻 [SLM-Default, LLM-Fallback](./slm-default-llm-fallback) | Route queries to a local SLM first, escalating to cloud only if the local model's output fails verification. | [demo.py](./slm-default-llm-fallback/demo.py) | [arXiv:2510.03847](https://arxiv.org/abs/2510.03847) | Cost & Latency Optimization |
| 💻 [Predictive Router](./router-agent) | Use a local router to classify queries as "weak" or "strong". Route simple tasks to local models and complex ones to the cloud. | [demo.py](./router-agent/demo.py) | [arXiv:2501.01818](https://arxiv.org/abs/2501.01818) | Dynamic Routing |
| 💻 [MAKER Protocol](./maker) | Decompose complex tasks using a cloud-based "Planner" and execute atomic steps using a local "Voting Solver" with convergence checks. | [demo.py](./maker/demo.py) | [arXiv:2511.09030](https://arxiv.org/abs/2511.09030) | Task Decomposition |
| 💻 [MINIONS Protocol](./minions) | Decompose extraction tasks into parallel jobs for local "minions" to process on document chunks, synthesizing results in the cloud. | [demo.py](./minions/demo.py) | [arXiv:2502.15964](https://arxiv.org/abs/2502.15964) | Local-Remote Map-Reduce |
| 💻 [Chain of Agents](./chain-of-agents) | Process long contexts by chaining local SLMs to sequentially build context before final synthesis in the cloud. | [demo.py](./chain-of-agents/demo.py) | [arXiv:2406.02818](https://arxiv.org/abs/2406.02818) | Sequential Bucket Brigade |

## Getting Started

1. **Prerequisites**: These demos use `agent-framework-mlx` and are optimized for Apple Silicon (macOS).
2. **Environment**: Create a `.env` file based on `.env.example` and provide your Azure AI credentials.
3. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Execution**: Run any of the demos:
   ```bash
   python slm-default-llm-fallback/demo.py
   ```
