import asyncio
import logging
import threading
from typing import Any, AsyncIterable, MutableSequence, Optional, Callable

from agent_framework import (
    BaseChatClient,
    ChatMessage,
    ChatOptions,
    ChatResponse,
    ChatResponseUpdate,
    Role,
    Content,
    UsageDetails,
)

from .config import LocalGenerationConfig

logger = logging.getLogger(__name__)


class TransformersChatClient(BaseChatClient):
    """
    A Chat Client that runs models locally using HuggingFace Transformers.

    This is a cross-platform alternative to MLXChatClient.  It loads models
    via ``transformers.AutoModelForCausalLM`` and generates text with the
    standard ``model.generate()`` API.
    """

    def __init__(
        self,
        model_path: str,
        generation_config: Optional[LocalGenerationConfig] = None,
        message_preprocessor: Optional[Callable[[list[dict[str, str]]], list[dict[str, str]]]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.generation_config = generation_config or LocalGenerationConfig()
        self.message_preprocessor = message_preprocessor
        self.model_id = model_path

        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def _prepare_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        msg_dicts: list[dict[str, str]] = []
        for m in messages:
            if isinstance(m.role, Role):
                role_str = m.role.value
            elif isinstance(m.role, str):
                role_str = m.role
            else:
                role_str = str(m.role)

            content_str = m.text if hasattr(m, "text") else str(m.contents)
            msg_dicts.append({"role": role_str, "content": content_str})

        if self.message_preprocessor:
            msg_dicts = self.message_preprocessor(msg_dicts)

        return msg_dicts


    def _prepare_inputs(self, messages: list[ChatMessage]):
        """Tokenize messages and return input_ids tensor + prompt token count."""
        msg_dicts = self._prepare_messages(messages)

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                msg_dicts,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in msg_dicts)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        return inputs, inputs["input_ids"].shape[-1]


    def _build_generate_kwargs(self) -> dict[str, Any]:
        cfg = self.generation_config
        kwargs: dict[str, Any] = {
            "max_new_tokens": cfg.max_tokens,
            "do_sample": cfg.temp > 0,
        }
        if cfg.temp > 0:
            kwargs["temperature"] = cfg.temp
            kwargs["top_p"] = cfg.top_p
            if cfg.top_k > 0:
                kwargs["top_k"] = cfg.top_k
        if cfg.repetition_penalty is not None:
            kwargs["repetition_penalty"] = cfg.repetition_penalty
        if cfg.seed is not None:
            import torch
            kwargs["generator"] = torch.Generator(device="cpu").manual_seed(cfg.seed)
        return kwargs


    async def _inner_get_response(
        self,
        *,
        messages: MutableSequence[ChatMessage],
        options: dict[str, Any] = {},
        **kwargs: Any,
    ) -> ChatResponse:
        inputs, prompt_tokens = self._prepare_inputs(list(messages))
        gen_kwargs = self._build_generate_kwargs()

        output_ids = await asyncio.to_thread(
            self.model.generate,
            **inputs,
            **gen_kwargs,
        )

        # Decode only the new tokens
        new_tokens = output_ids[0][prompt_tokens:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        completion_tokens = len(new_tokens)
        usage = UsageDetails(
            input_token_count=prompt_tokens,
            output_token_count=completion_tokens,
            total_token_count=prompt_tokens + completion_tokens,
        )

        return ChatResponse(
            messages=[ChatMessage(role=Role.ASSISTANT, contents=[Content.from_text(text=response_text)])],
            model_id=self.model_id,
            usage_details=usage,
        )

    async def _inner_get_streaming_response(
        self,
        *,
        messages: MutableSequence[ChatMessage],
        options: dict[str, Any] = {},
        **kwargs: Any,
    ) -> AsyncIterable[ChatResponseUpdate]:
        from transformers import TextIteratorStreamer

        inputs, prompt_tokens = self._prepare_inputs(list(messages))
        gen_kwargs = self._build_generate_kwargs()

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

        thread = threading.Thread(
            target=self.model.generate,
            kwargs={**inputs, **gen_kwargs},
        )
        thread.start()

        completion_tokens = 0
        for chunk in streamer:
            if chunk:
                completion_tokens += len(self.tokenizer.encode(chunk, add_special_tokens=False))
                yield ChatResponseUpdate(
                    role=Role.ASSISTANT,
                    contents=[Content.from_text(text=chunk)],
                    model_id=self.model_id,
                )

        thread.join()

        usage = UsageDetails(
            input_token_count=prompt_tokens,
            output_token_count=completion_tokens,
            total_token_count=prompt_tokens + completion_tokens,
        )
        yield ChatResponseUpdate(
            role=Role.ASSISTANT,
            contents=[Content.from_usage(usage_details=usage)],
            model_id=self.model_id,
        )
