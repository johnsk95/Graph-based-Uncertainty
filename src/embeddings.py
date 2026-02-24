"""
Embedding Utilities for Iterative Knowledge Expansion

Provides functions to compute embeddings for claims and priors using
sentence transformers.
"""

import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import json


class EmbeddingModel:
    """
    Wrapper for sentence-transformers to compute text embeddings.

    Uses a lightweight model by default for efficiency, but can be
    configured to use larger models for better quality.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: Optional[str] = None):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Options: 'all-MiniLM-L6-v2' (fast), 'all-mpnet-base-v2' (better quality)
            cache_dir: Optional directory to cache embeddings
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir
        self.embedding_cache: Dict[str, np.ndarray] = {}

        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            self._load_cache()

    def _load_cache(self):
        """Load cached embeddings if they exist."""
        if self.cache_dir:
            cache_file = Path(self.cache_dir) / f'embeddings_{self.model_name.replace("/", "_")}.json'
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    self.embedding_cache = {k: np.array(v) for k, v in cached.items()}
                print(f"Loaded {len(self.embedding_cache)} cached embeddings")

    def _save_cache(self):
        """Save embeddings to cache."""
        if self.cache_dir:
            cache_file = Path(self.cache_dir) / f'embeddings_{self.model_name.replace("/", "_")}.json'
            with open(cache_file, 'w') as f:
                json.dump({k: v.tolist() for k, v in self.embedding_cache.items()}, f)

    def encode(self, texts: Union[str, List[str]], show_progress: bool = False) -> np.ndarray:
        """
        Encode text(s) into embeddings.

        Args:
            texts: Single text or list of texts to encode
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings, shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        # Check cache for existing embeddings
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if text not in self.embedding_cache:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Compute embeddings for uncached texts
        if uncached_texts:
            new_embeddings = self.model.encode(
                uncached_texts,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )

            # Update cache
            for text, emb in zip(uncached_texts, new_embeddings):
                self.embedding_cache[text] = emb

            # Save cache periodically
            if self.cache_dir and len(uncached_texts) > 0:
                self._save_cache()

        # Assemble results in order
        embeddings = np.array([self.embedding_cache[text] for text in texts])

        return embeddings

    def encode_claims(self, claims: List[str], show_progress: bool = True) -> Dict[int, np.ndarray]:
        """
        Encode a list of claims and return as a dictionary mapping index to embedding.

        Args:
            claims: List of claim texts
            show_progress: Whether to show progress bar

        Returns:
            Dictionary mapping claim index to embedding vector
        """
        embeddings = self.encode(claims, show_progress=show_progress)
        return {i: embeddings[i] for i in range(len(claims))}

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embeddings."""
        return self.model.get_sentence_embedding_dimension()


def compute_claim_embeddings(
    claims: List[str],
    model: Optional[EmbeddingModel] = None,
    model_name: str = 'all-MiniLM-L6-v2',
    cache_dir: Optional[str] = None
) -> Dict[int, np.ndarray]:
    """
    Compute embeddings for a list of claims.

    Args:
        claims: List of claim texts
        model: Optional pre-initialized EmbeddingModel
        model_name: Model name if creating new model
        cache_dir: Cache directory if creating new model

    Returns:
        Dictionary mapping claim index to embedding vector
    """
    if model is None:
        model = EmbeddingModel(model_name=model_name, cache_dir=cache_dir)

    return model.encode_claims(claims)


def compute_prior_embeddings(
    prior_claims: List[str],
    model: Optional[EmbeddingModel] = None,
    model_name: str = 'all-MiniLM-L6-v2',
    cache_dir: Optional[str] = None
) -> np.ndarray:
    """
    Compute embeddings for prior claims.

    Args:
        prior_claims: List of prior claim texts
        model: Optional pre-initialized EmbeddingModel
        model_name: Model name if creating new model
        cache_dir: Cache directory if creating new model

    Returns:
        Numpy array of prior embeddings, shape (num_priors, embedding_dim)
    """
    if model is None:
        model = EmbeddingModel(model_name=model_name, cache_dir=cache_dir)

    if len(prior_claims) == 0:
        return np.array([])

    return model.encode(prior_claims, show_progress=True)


if __name__ == "__main__":
    # Test the embedding utilities
    print("Testing Embedding Utilities...")

    model = EmbeddingModel()

    # Test claims
    claims = [
        "Albert Einstein was a physicist.",
        "Albert Einstein won the Nobel Prize.",
        "The sky is blue.",
    ]

    embeddings = model.encode(claims)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dim: {model.embedding_dim}")

    # Test similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)
    print(f"Similarity between claim 0 and 1: {sim_matrix[0, 1]:.3f}")
    print(f"Similarity between claim 0 and 2: {sim_matrix[0, 2]:.3f}")

    print("\nEmbedding utilities working!")
