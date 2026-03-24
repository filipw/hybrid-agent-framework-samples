import os
from typing import Optional, Callable
from agent_framework import BaseChatClient
from .config import LocalGenerationConfig

# Short model name -> full model path per backend.
_MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "Phi-4-mini-instruct-4bit": {
        "mlx": "mlx-community/Phi-4-mini-instruct-4bit",
        "transformers": "microsoft/Phi-4-mini-instruct",
    },
    "Phi-4-mini-instruct-8bit": {
        "mlx": "mlx-community/Phi-4-mini-instruct-8bit",
        "transformers": "microsoft/Phi-4-mini-instruct",
    },
}


def _resolve_model_path(model_path: str, backend: str) -> str:
    """Resolve a model path for the chosen backend.

    Accepts either a short name from the registry (e.g. "Phi-4-mini-instruct-4bit")
    or a fully-qualified HuggingFace model ID. Short names are expanded to the
    appropriate backend-specific path. Fully-qualified paths are returned as-is.
    """
    if model_path in _MODEL_REGISTRY:
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
        - "transformers": Uses HuggingFace transformers (cross-platform)

    The *model_path* can be a short registry name (e.g. "Phi-4-mini-instruct-4bit")
    which is automatically resolved to the correct backend-specific model, or a
    fully-qualified HuggingFace model ID.

    Args:
        model_path: HuggingFace model ID or local path.
        generation_config: Backend-agnostic generation settings.
        message_preprocessor: Optional callback to transform messages before inference.
    """
    backend = os.environ.get("LOCAL_BACKEND", "mlx").lower()
    config = generation_config or LocalGenerationConfig()

    resolved_path = _resolve_model_path(model_path, backend)

    print(f"   [Local Backend: {backend}] Loading model: {resolved_path}")

    if backend == "mlx":
        return _create_mlx_client(resolved_path, config, message_preprocessor)
    elif backend == "transformers":
        return _create_transformers_client(resolved_path, config, message_preprocessor)
    else:
        raise ValueError(f"Unknown LOCAL_BACKEND: '{backend}'. Supported: 'mlx', 'transformers'")


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


def _create_transformers_client(
    model_path: str,
    config: LocalGenerationConfig,
    message_preprocessor: Optional[Callable],
) -> BaseChatClient:
    from .transformers_backend import TransformersChatClient

    return TransformersChatClient(
        model_path=model_path,
        generation_config=config,
        message_preprocessor=message_preprocessor,
    )
