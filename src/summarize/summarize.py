# summarize.py
# Doctown v11 – Summarize subsystem: Generate natural-language summaries using local LLM

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import qwen-agent for better LLM handling
try:
    from qwen_agent.llm import get_chat_model
    QWEN_AGENT_AVAILABLE = True
except ImportError:
    QWEN_AGENT_AVAILABLE = False
    print("Warning: qwen-agent not available. Install with: uv pip install qwen-agent")


# ============================================================
# Configuration
# ============================================================

# Default model for summarization
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B"

# Generation parameters
DEFAULT_MAX_LENGTH = 150  # Maximum tokens in summary
DEFAULT_MIN_LENGTH = 30   # Minimum tokens in summary
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50

# Prompt templates
DEFAULT_SHORT_SUMMARY_TEMPLATE = """Analyze the following code/text chunks and provide a concise 1-2 sentence summary of the main topic or purpose:

{chunks}

Summary:"""

DEFAULT_LONG_SUMMARY_TEMPLATE = """Analyze the following code/text chunks and provide a detailed summary covering:
1. Main purpose and functionality
2. Key components or concepts
3. Important patterns or relationships

{chunks}

Detailed Summary:"""

DEFAULT_PROJECT_SUMMARY_TEMPLATE = """Based on the following cluster summaries from a codebase, provide a comprehensive project overview:

{cluster_summaries}

Project Overview:"""


# ============================================================
# Output Format
# ============================================================

@dataclass
class ClusterSummary:
    """Summary for a single cluster."""
    cluster_id: int
    short_summary: str
    long_summary: str
    num_chunks: int
    sample_chunks: List[str] = field(default_factory=list)  # Top chunks used for summary
    metadata: dict = field(default_factory=dict)


@dataclass
class SummaryResult:
    """Complete summarization result for all clusters."""
    cluster_summaries: List[ClusterSummary]
    project_summary: Optional[str] = None
    model_name: str = ""
    total_clusters: int = 0
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"SummaryResult(model={self.model_name}, "
            f"clusters={self.total_clusters}, "
            f"has_project_summary={self.project_summary is not None})"
        )


# ============================================================
# Model Management
# ============================================================

class SummaryModel:
    """
    Wrapper for the Qwen summarization model.

    Handles model loading, caching, and inference.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
    ):
        """
        Initialize the summarization model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on ("cuda", "cpu", or None for auto)
            load_in_8bit: Load model in 8-bit precision to save memory
        """
        self.model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading summarization model: {model_name}")
        print(f"Device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_kwargs = {
            "trust_remote_code": True,
        }

        if load_in_8bit and self.device == "cuda":
            model_kwargs["load_in_8bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        if not load_in_8bit:
            self.model = self.model.to(self.device) # type: ignore

        self.model.eval()

        print(f"✓ Model loaded successfully")

    def generate_summary(
        self,
        prompt: str,
        max_length: int = DEFAULT_MAX_LENGTH,
        min_length: int = DEFAULT_MIN_LENGTH,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
    ) -> str:
        """
        Generate a summary from a prompt.

        Args:
            prompt: Input prompt with context
            max_length: Maximum length of generated summary
            min_length: Minimum length of generated summary
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter

        Returns:
            Generated summary text
        """
        # Apply chat template if available (for Qwen and other chat models)
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,  # Context window limit
        ).to(self.device)

        input_length = inputs.input_ids.shape[1]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                min_new_tokens=min_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # CRITICAL FIX: Only decode the newly generated tokens, not the input
        # This prevents the prompt from being included in the output
        generated_tokens = outputs[0][input_length:]

        summary = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        return summary


# ============================================================
# Cluster Summarization
# ============================================================

def summarize_cluster(
    chunks: List[str],
    cluster_id: int,
    model: SummaryModel,
    max_chunks: int = 10,
    short_template: str = DEFAULT_SHORT_SUMMARY_TEMPLATE,
    long_template: str = DEFAULT_LONG_SUMMARY_TEMPLATE,
    verbose: bool = True,
) -> ClusterSummary:
    """
    Generate summaries for a single cluster.

    Args:
        chunks: List of text chunks in this cluster
        cluster_id: ID of the cluster
        model: SummaryModel instance
        max_chunks: Maximum number of chunks to include in summary prompt
        short_template: Template for short summary
        long_template: Template for long summary
        verbose: Print progress

    Returns:
        ClusterSummary with short and long summaries
    """
    if verbose:
        print(f"\nSummarizing cluster {cluster_id} ({len(chunks)} chunks)...")

    # Select representative chunks (first N for now, could use more sophisticated selection)
    sample_chunks = chunks[:max_chunks]

    # Combine chunks for prompt
    chunks_text = "\n\n---\n\n".join(f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(sample_chunks))

    # Generate short summary
    short_prompt = short_template.format(chunks=chunks_text)
    short_summary = model.generate_summary(
        short_prompt,
        max_length=50,
        min_length=10,
    )

    if verbose:
        print(f"  Short: {short_summary[:100]}...")

    # Generate long summary
    long_prompt = long_template.format(chunks=chunks_text)
    long_summary = model.generate_summary(
        long_prompt,
        max_length=200,
        min_length=50,
    )

    if verbose:
        print(f"  Long: {long_summary[:100]}...")

    return ClusterSummary(
        cluster_id=cluster_id,
        short_summary=short_summary,
        long_summary=long_summary,
        num_chunks=len(chunks),
        sample_chunks=sample_chunks,
        metadata={
            "max_chunks_used": max_chunks,
            "actual_chunks_used": len(sample_chunks),
        }
    )


def summarize_all_clusters(
    chunks: List,
    cluster_labels: np.ndarray,
    model_name: str = DEFAULT_MODEL_NAME,
    generate_project_summary: bool = True,
    max_chunks_per_cluster: int = 10,
    device: Optional[str] = None,
    load_in_8bit: bool = False,
    verbose: bool = True,
) -> SummaryResult:
    """
    Generate summaries for all clusters and optionally a project overview.

    Args:
        chunks: List of Chunk objects (from chunk subsystem)
        cluster_labels: Cluster assignment for each chunk (from cluster subsystem)
        model_name: HuggingFace model name to use
        generate_project_summary: Whether to generate overall project summary
        max_chunks_per_cluster: Max chunks to use for each cluster summary
        device: Device to run model on
        load_in_8bit: Use 8-bit quantization to save memory
        verbose: Print progress

    Returns:
        SummaryResult with all cluster summaries and optional project summary
    """
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARIZATION")
        print("=" * 70)
        print(f"\nModel: {model_name}")
        print(f"Device: {device or 'auto'}")
        print(f"Total chunks: {len(chunks)}")

    # Initialize model
    model = SummaryModel(
        model_name=model_name,
        device=device,
        load_in_8bit=load_in_8bit,
    )

    # Group chunks by cluster
    cluster_chunks = {}
    for chunk, label in zip(chunks, cluster_labels):
        if label not in cluster_chunks:
            cluster_chunks[label] = []
        # Extract text from Chunk object
        chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
        cluster_chunks[label].append(chunk_text)

    # Filter out noise cluster (-1) if present
    if -1 in cluster_chunks:
        if verbose:
            print(f"\nSkipping noise cluster (-1) with {len(cluster_chunks[-1])} chunks")
        del cluster_chunks[-1]

    n_clusters = len(cluster_chunks)

    if verbose:
        print(f"\nGenerating summaries for {n_clusters} clusters...")

    # Summarize each cluster
    cluster_summaries = []
    for cluster_id in sorted(cluster_chunks.keys()):
        summary = summarize_cluster(
            chunks=cluster_chunks[cluster_id],
            cluster_id=cluster_id,
            model=model,
            max_chunks=max_chunks_per_cluster,
            verbose=verbose,
        )
        cluster_summaries.append(summary)

    # Generate project-level summary
    project_summary = None
    if generate_project_summary and cluster_summaries:
        if verbose:
            print(f"\nGenerating project overview...")

        # Combine cluster summaries
        cluster_summary_text = "\n\n".join(
            f"Cluster {cs.cluster_id} ({cs.num_chunks} chunks):\n{cs.long_summary}"
            for cs in cluster_summaries
        )

        project_prompt = DEFAULT_PROJECT_SUMMARY_TEMPLATE.format(
            cluster_summaries=cluster_summary_text
        )

        project_summary = model.generate_summary(
            project_prompt,
            max_length=300,
            min_length=100,
        )

        if verbose:
            print(f"  Project summary: {project_summary[:150]}...")

    if verbose:
        print(f"\n✓ Summarization complete")

    return SummaryResult(
        cluster_summaries=cluster_summaries,
        project_summary=project_summary,
        model_name=model_name,
        total_clusters=n_clusters,
        metadata={
            "max_chunks_per_cluster": max_chunks_per_cluster,
            "device": model.device,
            "load_in_8bit": load_in_8bit,
        }
    )


# ============================================================
# Utility Functions
# ============================================================

def print_summary_report(result: SummaryResult) -> None:
    """
    Print a formatted summary report.

    Args:
        result: SummaryResult to print
    """
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)

    print(f"\nModel: {result.model_name}")
    print(f"Total Clusters: {result.total_clusters}")

    for summary in result.cluster_summaries:
        print(f"\n{'─' * 70}")
        print(f"Cluster {summary.cluster_id} ({summary.num_chunks} chunks)")
        print(f"{'─' * 70}")
        print(f"\nShort Summary:\n  {summary.short_summary}")
        print(f"\nDetailed Summary:\n  {summary.long_summary}")

    if result.project_summary:
        print(f"\n{'═' * 70}")
        print("PROJECT OVERVIEW")
        print(f"{'═' * 70}")
        print(f"\n{result.project_summary}")

    print("\n" + "=" * 70)


def save_summary_result(result: SummaryResult, filepath: str) -> None:
    """
    Save summary result to a text file.

    Args:
        result: SummaryResult to save
        filepath: Path to output file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("DOCTOWN SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Model: {result.model_name}\n")
        f.write(f"Total Clusters: {result.total_clusters}\n\n")

        for summary in result.cluster_summaries:
            f.write("─" * 70 + "\n")
            f.write(f"Cluster {summary.cluster_id} ({summary.num_chunks} chunks)\n")
            f.write("─" * 70 + "\n\n")
            f.write(f"Short Summary:\n{summary.short_summary}\n\n")
            f.write(f"Detailed Summary:\n{summary.long_summary}\n\n")

        if result.project_summary:
            f.write("═" * 70 + "\n")
            f.write("PROJECT OVERVIEW\n")
            f.write("═" * 70 + "\n\n")
            f.write(f"{result.project_summary}\n\n")

        f.write("=" * 70 + "\n")

    print(f"✓ Saved summary report to {filepath}")


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":
    print("Testing summarize subsystem...")
    print("=" * 60)

    # Test with synthetic data
    print("\n[Setup] Creating test data")
    print("-" * 60)

    # Mock Chunk class for testing
    from dataclasses import dataclass as test_dataclass

    @test_dataclass
    class MockChunk:
        text: str

    # Create test chunks
    test_chunks = [
        MockChunk("def calculate_sum(a, b): return a + b"),
        MockChunk("def calculate_product(a, b): return a * b"),
        MockChunk("class MathOperations: pass"),
        MockChunk("def parse_json(data): import json; return json.loads(data)"),
        MockChunk("def write_file(path, content): with open(path, 'w') as f: f.write(content)"),
    ]

    # Create test cluster labels
    test_labels = np.array([0, 0, 0, 1, 1])  # 2 clusters

    print(f"✓ Created {len(test_chunks)} test chunks in {len(set(test_labels))} clusters")

    # Test summarization
    print("\n\n[Test] Generating summaries")
    print("-" * 60)

    result = summarize_all_clusters(
        chunks=test_chunks,
        cluster_labels=test_labels,
        model_name=DEFAULT_MODEL_NAME,
        generate_project_summary=True,
        max_chunks_per_cluster=5,
        verbose=True,
    )

    # Print results
    print_summary_report(result)

    # Test saving
    print("\n\n[Test] Saving summary")
    print("-" * 60)

    test_file = "/tmp/test_summary.txt"
    save_summary_result(result, test_file)

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
