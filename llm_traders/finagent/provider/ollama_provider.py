"""
Ollama Provider for FinAgent
=============================

OpenAI-compatible provider that routes LLM calls to a local Ollama server
and uses sentence-transformers for embeddings.
"""

import os
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union, Literal

import backoff
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

from llm_traders.finagent.provider import LLMProvider, EmbeddingProvider
from llm_traders.finagent.registry import PROVIDER
from llm_traders.finagent.utils import assemble_project_path, load_json


PROVIDER_SETTING_COMP_MODEL = "comp_model"
PROVIDER_SETTING_EMB_MODEL = "emb_model"


@PROVIDER.register_module(force=True)
class OllamaProvider(LLMProvider, EmbeddingProvider):
    """Provider that uses Ollama (OpenAI-compatible API) for chat completions
    and sentence-transformers for embeddings."""

    client: Any = None
    llm_model: str = ""
    embedding_model: str = ""
    _st_model: Any = None  # sentence-transformers model

    def __init__(self, provider_cfg_path) -> None:
        self.retries = 5
        provider_cfg_path = assemble_project_path(provider_cfg_path)
        provider_cfg = load_json(provider_cfg_path)
        self.init_provider(provider_cfg)

    def init_provider(self, provider_cfg) -> None:
        self.provider_cfg = provider_cfg

        # Ollama serves an OpenAI-compatible API
        base_url = provider_cfg.get("base_url", "http://localhost:11434/v1")
        self.client = OpenAI(base_url=base_url, api_key="ollama")

        self.llm_model = provider_cfg[PROVIDER_SETTING_COMP_MODEL]
        self.embedding_model = provider_cfg.get(PROVIDER_SETTING_EMB_MODEL, "local:all-MiniLM-L6-v2")

        # Initialize sentence-transformers for embedding
        emb_model_name = self.embedding_model
        if emb_model_name.startswith("local:"):
            emb_model_name = emb_model_name.split(":", 1)[1]

        from sentence_transformers import SentenceTransformer
        self._st_model = SentenceTransformer(emb_model_name)
        self._emb_dim = int(self._st_model.get_sentence_embedding_dimension())

    def get_embedding_dim(self) -> int:
        return self._emb_dim

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._st_model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        seed: int | None = 42,
        max_tokens: int = 512,
    ) -> Tuple[str, Dict[str, int]]:
        if model is None:
            model = self.llm_model

        @backoff.on_exception(
            backoff.constant,
            (APIError, RateLimitError, APITimeoutError),
            max_tries=self.retries,
            interval=10,
        )
        def _generate_response_with_retry(
            messages, model, temperature, seed, max_tokens=512,
        ) -> Tuple[str, Dict[str, int]]:
            MAX_RETRIES = 5
            for i in range(MAX_RETRIES):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except Exception as e:
                    print(f"Ollama API error (attempt {i+1}/{MAX_RETRIES}): {e}")
                    if i == MAX_RETRIES - 1:
                        return "", {}
                    continue

                if response is None:
                    print("Failed to get a response from Ollama. Try again.")
                    continue

                message = response.choices[0].message.content

                info = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0) if response.usage else 0,
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0) if response.usage else 0,
                    "total_tokens": getattr(response.usage, 'total_tokens', 0) if response.usage else 0,
                }

                return message, info

            print(f"Failed to get a response from Ollama after {MAX_RETRIES} retries.")
            return "", {}

        return _generate_response_with_retry(messages, model, temperature, seed, max_tokens)

    def num_tokens_from_messages(self, messages, model=None) -> int:
        """Approximate token count — Ollama models don't use tiktoken."""
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # message overhead
            for key, value in message.items():
                if key == "content":
                    if isinstance(value, str):
                        num_tokens += int(len(value.split()) * 1.3)
                    elif isinstance(value, list):
                        for content in value:
                            if isinstance(content, dict) and content.get("type") == "text":
                                num_tokens += int(len(content["text"].split()) * 1.3)
        num_tokens += 3  # reply priming
        return num_tokens

    def assemble_prompt(self, system_prompts: List[str], user_inputs: List[str], image_filenames: List[str]) -> List[str]:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompts[0]}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_inputs[0]}]
            }
        ]
        return messages
