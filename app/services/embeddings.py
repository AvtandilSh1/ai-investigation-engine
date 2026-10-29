from functools import lru_cache
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class Embedder:
    """
    Local dense-ish embeddings via hashing + L2 norm.
    Keeps the stack free of torch so free-tier hosts can install cleanly.
    Hybrid search still mixes this with BM25.
    """

    def __init__(self, model_name: str = "hashing-384", dim: int = 384):
        self.model_name = model_name
        self.dim = dim
        self._vectorizer = HashingVectorizer(
            n_features=dim,
            alternate_sign=False,
            norm=None,
            ngram_range=(1, 2),
            lowercase=True,
        )

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        cleaned = [t if t and str(t).strip() else " " for t in texts]
        matrix = self._vectorizer.transform(cleaned)
        dense = matrix.astype(np.float32).toarray()
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0].tolist()


@lru_cache
def get_embedder(model_name: str) -> Embedder:
    return Embedder(model_name=model_name)
