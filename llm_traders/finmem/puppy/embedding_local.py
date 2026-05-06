"""
Local Sentence-Transformer Embedding for Ollama Integration
============================================================

Drop-in replacement for ``OpenAILongerThanContextEmb`` that uses a local
sentence-transformers model instead of the OpenAI embedding API.

Default model: ``all-MiniLM-L6-v2`` (384-dim, ~80 MB, CPU-only).
"""

import numpy as np
from typing import List, Union


class LocalSentenceTransformerEmb:
    """
    Embedding function backed by a local sentence-transformers model.

    Interface-compatible with ``OpenAILongerThanContextEmb`` so it can be
    swapped in without changing downstream code (MemoryDB, BrainDB).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        **kwargs,  # absorb unused OpenAI-specific kwargs
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()

    def _emb(self, text: Union[List[str], str]) -> List[List[float]]:
        if isinstance(text, str):
            text = [text]
        embeddings = self.model.encode(text, normalize_embeddings=True)
        return embeddings.tolist()

    def __call__(self, text: Union[List[str], str]) -> np.ndarray:
        if isinstance(text, str):
            text = [text]
        embeddings = self.model.encode(text, normalize_embeddings=True)
        return np.array(embeddings).astype("float32")

    def get_embedding_dimension(self) -> int:
        return int(self._dim)
