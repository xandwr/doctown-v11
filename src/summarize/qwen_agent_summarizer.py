"""
qwen_agent_summarizer.py
Enhanced summarizer using qwen-agent for agentic capabilities.

This module demonstrates how to use qwen-agent to create an intelligent
summarization system that could be extended to handle ANY file format,
not just text/code files.

Future capabilities:
- Automatic file format detection and parsing
- Multi-modal analysis (images, PDFs, etc.)
- Tool-based processing pipelines
- RAG-enhanced summarization
"""

import json5
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from qwen_agent.agents import Assistant
from qwen_agent.llm import get_chat_model


# ============================================================
# Configuration
# ============================================================

@dataclass
class QwenAgentConfig:
    """Configuration for Qwen Agent summarizer."""
    # Model configuration
    model: str = "Qwen2.5-3B-Instruct"
    model_server: Optional[str] = None  # e.g., "http://localhost:8000/v1" for vLLM/Ollama
    api_key: str = "EMPTY"  # Use "EMPTY" for local models

    # Generation parameters
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 500

    # Agent behavior
    enable_tools: bool = True
    enable_rag: bool = False
    verbose: bool = True


# ============================================================
# Qwen Agent Summarizer
# ============================================================

class QwenAgentSummarizer:
    """
    Enhanced summarizer using Qwen Agent framework.

    This provides:
    - Better prompt handling via chat templates
    - Tool-calling capabilities for complex analysis
    - Potential for multi-modal inputs
    - RAG integration for context-aware summaries
    """

    def __init__(self, config: Optional[QwenAgentConfig] = None):
        """
        Initialize the Qwen Agent summarizer.

        Args:
            config: Configuration object
        """
        self.config = config or QwenAgentConfig()

        # Build LLM configuration
        llm_cfg = {
            'model': self.config.model,
            'generate_cfg': {
                'top_p': self.config.top_p,
                'temperature': self.config.temperature,
                'max_tokens': self.config.max_tokens,
            }
        }

        # Add model server config if provided (for local vLLM/Ollama)
        if self.config.model_server:
            llm_cfg['model_server'] = self.config.model_server
            llm_cfg['api_key'] = self.config.api_key

        # Create assistant agent
        system_instruction = """You are an expert technical documentation analyst.
Your task is to analyze code/text chunks and generate clear, concise summaries.

When generating summaries:
1. Identify the main purpose and functionality
2. Highlight key components or patterns
3. Keep summaries factual and technical
4. Avoid speculation or assumptions
5. Use clear, professional language"""

        tools = []  # Can add custom tools here

        self.agent = Assistant(
            llm=llm_cfg,
            system_message=system_instruction,
            function_list=tools,
        )

        if self.config.verbose:
            print(f"✓ Qwen Agent initialized with model: {self.config.model}")

    def generate_summary(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a summary using the Qwen Agent.

        Args:
            prompt: The summarization prompt
            context: Optional context dictionary

        Returns:
            Generated summary text
        """
        messages = [{'role': 'user', 'content': prompt}]

        # Run the agent
        response = []
        for response_chunk in self.agent.run(messages=messages):
            response = response_chunk

        # Extract text from response
        if response:
            # Response is a list of message dictionaries
            assistant_messages = [msg for msg in response if msg.get('role') == 'assistant']
            if assistant_messages:
                content = assistant_messages[-1].get('content', '')
                return content.strip()

        return ""

    def summarize_cluster(
        self,
        chunks: List[str],
        cluster_id: int,
        max_chunks: int = 10,
    ) -> Dict[str, str]:
        """
        Generate both short and long summaries for a cluster.

        Args:
            chunks: List of text chunks
            cluster_id: Cluster identifier
            max_chunks: Maximum chunks to include

        Returns:
            Dictionary with 'short' and 'long' summary keys
        """
        # Select representative chunks
        sample_chunks = chunks[:max_chunks]
        chunks_text = "\n\n---\n\n".join(
            f"Chunk {i+1}:\n{chunk}"
            for i, chunk in enumerate(sample_chunks)
        )

        # Generate short summary
        short_prompt = f"""Analyze the following code/text chunks and provide a concise 1-2 sentence summary of the main topic or purpose:

{chunks_text}

Summary:"""

        short_summary = self.generate_summary(short_prompt)

        # Generate long summary
        long_prompt = f"""Analyze the following code/text chunks and provide a detailed summary covering:
1. Main purpose and functionality
2. Key components or concepts
3. Important patterns or relationships

{chunks_text}

Detailed Summary:"""

        long_summary = self.generate_summary(long_prompt)

        if self.config.verbose:
            print(f"\nCluster {cluster_id}:")
            print(f"  Short: {short_summary[:100]}...")
            print(f"  Long: {long_summary[:100]}...")

        return {
            'short': short_summary,
            'long': long_summary,
        }


# ============================================================
# Advanced Tool-Based Summarizer (Future Enhancement)
# ============================================================

class AdvancedAgenticSummarizer(QwenAgentSummarizer):
    """
    Advanced version with custom tools for:
    - Multi-format file parsing (PDF, DOCX, images, etc.)
    - Code analysis and AST parsing
    - Data extraction from structured formats
    - Context-aware summarization with RAG

    This demonstrates the vision for expanding doctown beyond just text/code
    to handle ANY file format and generate tailored documentation.
    """

    def __init__(self, config: Optional[QwenAgentConfig] = None):
        super().__init__(config)

        # TODO: Register custom tools
        # - PDF parser tool
        # - Image analysis tool (OCR, vision models)
        # - Structured data analyzer (JSON, CSV, Excel)
        # - Code AST analyzer
        # - Web content fetcher

        if self.config.verbose:
            print("  (Advanced tools not yet implemented)")

    def analyze_file(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """
        Analyze any file type and extract relevant information.

        This is a placeholder for future multi-format support.

        Args:
            file_path: Path to file
            file_type: Type of file (pdf, docx, image, etc.)

        Returns:
            Analysis results dictionary
        """
        # Future implementation would:
        # 1. Detect file type
        # 2. Route to appropriate tool
        # 3. Extract structured information
        # 4. Generate contextual summary

        raise NotImplementedError(
            "Multi-format analysis not yet implemented. "
            "This is a placeholder for future agentic capabilities."
        )


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("QWEN AGENT SUMMARIZER TEST")
    print("=" * 70)

    # Example 1: Basic usage with local model via vLLM/Ollama
    config = QwenAgentConfig(
        model="Qwen2.5-3B-Instruct",
        # Uncomment if using vLLM/Ollama:
        # model_server="http://localhost:8000/v1",
        temperature=0.2,
        verbose=True,
    )

    summarizer = QwenAgentSummarizer(config)

    # Test with sample chunks
    test_chunks = [
        "def embed_chunks(chunks, model_name='all-MiniLM-L6-v2'):\n    model = SentenceTransformer(model_name)\n    return model.encode(chunks)",
        "class EmbeddingModel:\n    def __init__(self, model_name):\n        self.model = SentenceTransformer(model_name)",
        "embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)",
    ]

    print("\nGenerating cluster summary...")
    result = summarizer.summarize_cluster(
        chunks=test_chunks,
        cluster_id=0,
        max_chunks=3,
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nShort Summary:\n{result['short']}")
    print(f"\nLong Summary:\n{result['long']}")

    print("\n" + "=" * 70)
    print("✓ Test complete!")
    print("=" * 70)
