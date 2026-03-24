from typing import Optional
from pydantic import BaseModel


class LocalGenerationConfig(BaseModel):
    """Backend-agnostic generation configuration for local models."""
    temp: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 1000
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None
