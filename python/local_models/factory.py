import os
from typing import Optional, Callable
from agent_framework import BaseChatClient
from .config import LocalGenerationConfig

_MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "phi-4-4bit": {
        "mlx": "mlx-community/Phi-4-mini-instruct-4bit",
        "foundry_local": "phi-4-mini",
    },
    "phi-4-8bit": {
        "mlx": "mlx-community/Phi-4-mini-instruct-8bit",
        "foundry_local": "phi-4-mini",
    },
}

def _resolve_model_path(model_path: str, backend: str) -> str:
    """Resolve a model name for the chosen backend.

    Accepts either a short name from the registry (e.g. "phi-4-4bit") or a
    fully-qualified model ID / alias passed directly to the backend.
    """
    if model_path in _MODEL_REGISTRY and backend in _MODEL_REGISTRY[model_path]:
        return _MODEL_REGISTRY[model_path][backend]
    return model_path


def create_local_client(
    model_path: str,
    generation_config: Optional[LocalGenerationConfig] = None,
    message_preprocessor: Optional[Callable[[list[dict[str, str]]], list[dict[str, str]]]] = None,
) -> BaseChatClient:
    """
    Factory that creates a local model client based on the LOCAL_BACKEND env var.

    Supported backends:
        - "mlx" (default): Uses agent-framework-mlx (Apple Silicon only)
        - "foundry_local": Uses Foundry Local via agent-framework-foundry-local (cross-platform)

    The *model_path* can be a short registry name (e.g. "phi-4-4bit") which is
    automatically resolved to the correct backend-specific alias or model ID.

    Args:
        model_path: Short registry name or backend-specific model ID / alias.
        generation_config: Backend-agnostic generation settings.
        message_preprocessor: Optional message transform callback (mlx only; ignored for foundry_local).
    """
    backend = os.environ.get("LOCAL_BACKEND", "mlx").lower()
    config = generation_config or LocalGenerationConfig()

    resolved_path = _resolve_model_path(model_path, backend)

    print(f"   [Local Backend: {backend}] Loading model: {resolved_path}")

    if backend == "mlx":
        return _create_mlx_client(resolved_path, config, message_preprocessor)
    elif backend == "foundry_local":
        return _create_foundry_local_client(resolved_path, config)
    else:
        raise ValueError(f"Unknown LOCAL_BACKEND: '{backend}'. Supported: 'mlx', 'foundry_local'")


def _create_mlx_client(
    model_path: str,
    config: LocalGenerationConfig,
    message_preprocessor: Optional[Callable],
) -> BaseChatClient:
    from agent_framework_mlx import MLXChatClient, MLXGenerationConfig

    mlx_config = MLXGenerationConfig(
        temp=config.temp,
        top_p=config.top_p,
        top_k=config.top_k,
        max_tokens=config.max_tokens,
        repetition_penalty=config.repetition_penalty,
        seed=config.seed,
    )
    return MLXChatClient(
        model_path=model_path,
        generation_config=mlx_config,
        message_preprocessor=message_preprocessor,
    )


def _create_foundry_local_client(
    model_alias: str,
    config: LocalGenerationConfig,
) -> BaseChatClient:
    from agent_framework_foundry_local import FoundryLocalClient

    return FoundryLocalClient(model=model_alias)
