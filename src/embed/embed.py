# embed.py
# Doctown v11 – Embed subsystem: GPU-accelerated embeddings via SentenceTransformers

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Union
import sys
from pathlib import Path

try:
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "Required packages not installed. Install with: pip install sentence-transformers torch"
    ) from e


# ============================================================
# Configuration
# ============================================================

# Default model: Google's EmbeddingGemma-300M
DEFAULT_MODEL = "google/embeddinggemma-300m"

# Batch size for embedding (adjust based on GPU memory)
DEFAULT_BATCH_SIZE = 32

# Maximum sequence length for most embedding models
DEFAULT_MAX_LENGTH = 512


# ============================================================
# Device Detection
# ============================================================

def get_device_info() -> dict:
    """
    Get information about available compute devices.

    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": 0,
        "cuda_device_name": None,
        "device": "cpu",
    }

    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["device"] = "cuda"

        # Additional GPU info
        if hasattr(torch.cuda, 'get_device_properties'):
            props = torch.cuda.get_device_properties(0)
            info["cuda_total_memory"] = props.total_memory
            info["cuda_memory_gb"] = props.total_memory / (1024**3)

    return info


# ============================================================
# Embedding Model Wrapper
# ============================================================

@dataclass
class EmbeddingModel:
    """
    Wrapper for SentenceTransformer models with device management.
    """
    model_name: str
    device: str
    batch_size: int
    max_length: int
    _model: Optional[SentenceTransformer] = None

    def __post_init__(self):
        """Initialize the model after dataclass creation."""
        if self._model is None:
            self.load_model()

    def load_model(self) -> None:
        """Load the SentenceTransformer model."""
        print(f"Loading model: {self.model_name}")
        print(f"Device: {self.device}")

        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
            trust_remote_code=True,  # Required for some HuggingFace models
        )

        # Set max sequence length
        if hasattr(self._model, 'max_seq_length'):
            self._model.max_seq_length = self.max_length

        print(f"✓ Model loaded successfully")

        # Print model info
        if hasattr(self._model, 'get_sentence_embedding_dimension'):
            dim = self._model.get_sentence_embedding_dimension()
            print(f"Embedding dimension: {dim}")

    def encode(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding (default: use model's batch_size)
            show_progress: Show progress bar
            normalize_embeddings: L2 normalize embeddings (recommended for cosine similarity)

        Returns:
            numpy array of shape (N, D) where N is number of texts, D is embedding dimension
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not texts:
            return np.array([])

        batch_size = batch_size or self.batch_size

        # Encode texts
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
        )

        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        if hasattr(self._model, 'get_sentence_embedding_dimension'):
            return self._model.get_sentence_embedding_dimension()
        else:
            # Fallback: encode a dummy text
            dummy = self.encode(["test"], show_progress=False)
            return dummy.shape[1]

    def to(self, device: str) -> 'EmbeddingModel':
        """Move model to a different device."""
        if self._model is not None:
            self._model = self._model.to(device)
            self.device = device
        return self


# ============================================================
# Main Embedding Functions
# ============================================================

def create_embedding_model(
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> EmbeddingModel:
    """
    Create an embedding model with automatic device selection.

    Args:
        model_name: HuggingFace model identifier
        device: Device to use ("cuda", "cpu", or None for auto-detect)
        batch_size: Batch size for encoding
        max_length: Maximum sequence length

    Returns:
        EmbeddingModel instance
    """
    # Auto-detect device if not specified
    if device is None:
        device_info = get_device_info()
        device = device_info["device"]

        if device == "cuda":
            print(f"✓ CUDA available: {device_info['cuda_device_name']}")
            print(f"  Memory: {device_info['cuda_memory_gb']:.1f} GB")
        else:
            print("⚠ CUDA not available, using CPU")

    return EmbeddingModel(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )


def embed_chunks(
    chunks: Union[List[str], List],
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    normalize: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Embed a list of text chunks.

    This is the main function to use from other subsystems.

    Args:
        chunks: List of text strings or Chunk objects from chunk subsystem
        model_name: HuggingFace model identifier
        device: Device to use ("cuda", "cpu", or None for auto-detect)
        batch_size: Batch size for encoding
        normalize: L2 normalize embeddings (recommended for cosine similarity)
        show_progress: Show progress bar

    Returns:
        numpy array of shape (N, D) where N is number of chunks, D is embedding dimension
    """
    # Handle Chunk objects from chunk subsystem
    if chunks and hasattr(chunks[0], 'text'):
        texts = [chunk.text for chunk in chunks]
    else:
        texts = chunks

    if not texts:
        return np.array([])

    # Create model
    model = create_embedding_model(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )

    # Encode
    print(f"\nEmbedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        show_progress=show_progress,
        normalize_embeddings=normalize,
    )

    print(f"✓ Embeddings shape: {embeddings.shape}")

    return embeddings


# ============================================================
# Utility Functions
# ============================================================

def save_embeddings(embeddings: np.ndarray, filepath: str) -> None:
    """
    Save embeddings to disk.

    Args:
        embeddings: numpy array of embeddings
        filepath: Path to save file (.npy format)
    """
    np.save(filepath, embeddings)
    print(f"✓ Saved embeddings to {filepath}")


def load_embeddings(filepath: str) -> np.ndarray:
    """
    Load embeddings from disk.

    Args:
        filepath: Path to .npy file

    Returns:
        numpy array of embeddings
    """
    embeddings = np.load(filepath)
    print(f"✓ Loaded embeddings from {filepath}")
    print(f"  Shape: {embeddings.shape}")
    return embeddings


def compute_similarity(embeddings: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and all embeddings.

    Args:
        embeddings: Array of shape (N, D)
        query_embedding: Array of shape (D,) or (1, D)

    Returns:
        Array of shape (N,) with similarity scores
    """
    # Ensure query is 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Compute cosine similarity (assumes normalized embeddings)
    similarities = embeddings @ query_embedding.T

    return similarities.squeeze()


def find_most_similar(
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find most similar embeddings to query.

    Args:
        embeddings: Array of shape (N, D)
        query_embedding: Array of shape (D,) or (1, D)
        top_k: Number of results to return

    Returns:
        Tuple of (indices, similarities) for top-k results
    """
    similarities = compute_similarity(embeddings, query_embedding)

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_similarities = similarities[top_indices]

    return top_indices, top_similarities


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":
    print("Testing embed subsystem...")
    print("=" * 60)

    # Test 1: Device detection
    print("\n[Test 1] Device detection")
    print("-" * 60)
    device_info = get_device_info()
    for key, value in device_info.items():
        print(f"{key}: {value}")

    # Test 2: Simple embedding
    print("\n\n[Test 2] Simple text embedding")
    print("-" * 60)

    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "Where there's a will, there's a way.",
    ]

    print(f"Embedding {len(sample_texts)} sample texts...")
    embeddings = embed_chunks(
        sample_texts,
        model_name=DEFAULT_MODEL,
        batch_size=2,
        show_progress=True,
    )

    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"First embedding (first 10 dims): {embeddings[0][:10]}")

    # Test 3: Similarity computation
    print("\n\n[Test 3] Similarity computation")
    print("-" * 60)

    query_text = "A long journey starts with a first step."
    print(f"Query: '{query_text}'")

    # Embed query
    model = create_embedding_model(model_name=DEFAULT_MODEL)
    query_emb = model.encode([query_text], show_progress=False)

    # Find similar
    indices, similarities = find_most_similar(embeddings, query_emb[0], top_k=3)

    print("\nMost similar texts:")
    for i, (idx, sim) in enumerate(zip(indices, similarities), 1):
        print(f"{i}. (similarity: {sim:.4f}) {sample_texts[idx]}")

    # Test 4: Save/load
    print("\n\n[Test 4] Save and load embeddings")
    print("-" * 60)

    test_file = "/tmp/test_embeddings.npy"
    save_embeddings(embeddings, test_file)
    loaded = load_embeddings(test_file)

    assert np.allclose(embeddings, loaded), "Loaded embeddings don't match!"
    print("✓ Save/load test passed")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
