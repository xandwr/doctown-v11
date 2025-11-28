#!/usr/bin/env python3
# example.py
# Doctown v11 – Example usage of cluster subsystem with embed subsystem

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from embed import embed_chunks
from cluster import cluster_embeddings, find_optimal_k


def example_basic_clustering():
    """Example 1: Basic KMeans clustering of text chunks."""
    print("\n" + "=" * 70)
    print("Example 1: Basic KMeans Clustering")
    print("=" * 70)

    # Sample text chunks about different topics
    chunks = [
        # Topic 1: Python programming
        "Python is a high-level programming language.",
        "Python supports object-oriented programming.",
        "Python has a rich ecosystem of libraries.",

        # Topic 2: Machine learning
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are used in deep learning.",
        "Training models requires large datasets.",

        # Topic 3: Web development
        "HTML is the markup language for web pages.",
        "CSS is used for styling web pages.",
        "JavaScript adds interactivity to websites.",

        # Topic 4: Databases
        "SQL is used for database queries.",
        "NoSQL databases are schema-flexible.",
        "Database indexing improves query performance.",
    ]

    print(f"\nEmbedding {len(chunks)} text chunks...")

    # Step 1: Generate embeddings
    embeddings = embed_chunks(
        chunks,
        model_name="google/embeddinggemma-300m",
        show_progress=True,
    )

    # Step 2: Cluster embeddings
    result = cluster_embeddings(
        embeddings,
        algorithm="kmeans",
        n_clusters=4,  # We know there are 4 topics
    )

    # Step 3: Display results
    print("\n" + "-" * 70)
    print("Clustering Results")
    print("-" * 70)

    cluster_sizes = result.get_cluster_sizes()
    for cluster_id in sorted(cluster_sizes.keys()):
        print(f"\nCluster {cluster_id} ({cluster_sizes[cluster_id]} items):")
        members = result.get_cluster_members(cluster_id)
        for idx in members:
            print(f"  - {chunks[idx]}")


def example_optimal_k():
    """Example 2: Find optimal number of clusters."""
    print("\n" + "=" * 70)
    print("Example 2: Finding Optimal K")
    print("=" * 70)

    # Sample chunks (unknown number of topics)
    chunks = [
        "The sun rises in the east.",
        "Stars shine brightly at night.",
        "The moon orbits the earth.",
        "Apples are red or green.",
        "Bananas are yellow fruits.",
        "Oranges are citrus fruits.",
        "Dogs are loyal pets.",
        "Cats are independent animals.",
        "Birds can fly in the sky.",
    ]

    print(f"\nEmbedding {len(chunks)} text chunks...")

    # Generate embeddings
    embeddings = embed_chunks(
        chunks,
        model_name="google/embeddinggemma-300m",
        show_progress=True,
    )

    # Find optimal k
    optimal_k, scores = find_optimal_k(
        embeddings,
        k_range=range(2, 6),
        metric="silhouette",
    )

    # Cluster with optimal k
    result = cluster_embeddings(
        embeddings,
        algorithm="kmeans",
        n_clusters=optimal_k,
    )

    # Display results
    print("\n" + "-" * 70)
    print(f"Clustering with optimal k={optimal_k}")
    print("-" * 70)

    cluster_sizes = result.get_cluster_sizes()
    for cluster_id in sorted(cluster_sizes.keys()):
        print(f"\nCluster {cluster_id} ({cluster_sizes[cluster_id]} items):")
        members = result.get_cluster_members(cluster_id)
        for idx in members:
            print(f"  - {chunks[idx]}")


def example_hdbscan():
    """Example 3: HDBSCAN clustering (automatic cluster detection)."""
    print("\n" + "=" * 70)
    print("Example 3: HDBSCAN Clustering (Automatic)")
    print("=" * 70)

    chunks = [
        # Large cluster: Programming
        "Functions are reusable blocks of code.",
        "Variables store data in memory.",
        "Loops allow repeated execution.",
        "Conditionals enable branching logic.",
        "Arrays store collections of items.",

        # Medium cluster: Food
        "Pizza is a popular Italian dish.",
        "Sushi is traditional Japanese cuisine.",
        "Tacos are a Mexican food.",

        # Small cluster or noise
        "The weather is nice today.",
        "Random text that doesn't fit.",
    ]

    print(f"\nEmbedding {len(chunks)} text chunks...")

    # Generate embeddings
    embeddings = embed_chunks(
        chunks,
        model_name="google/embeddinggemma-300m",
        show_progress=True,
    )

    # HDBSCAN clustering
    result = cluster_embeddings(
        embeddings,
        algorithm="hdbscan",
        min_cluster_size=3,  # Minimum 3 items per cluster
    )

    # Display results
    print("\n" + "-" * 70)
    print("Clustering Results")
    print("-" * 70)

    cluster_sizes = result.get_cluster_sizes()
    for cluster_id in sorted(cluster_sizes.keys()):
        if cluster_id == -1:
            print(f"\nNoise ({cluster_sizes[cluster_id]} items):")
        else:
            print(f"\nCluster {cluster_id} ({cluster_sizes[cluster_id]} items):")

        members = result.get_cluster_members(cluster_id)
        for idx in members:
            print(f"  - {chunks[idx]}")


def example_integration():
    """Example 4: Full pipeline integration with embed subsystem."""
    print("\n" + "=" * 70)
    print("Example 4: Full Pipeline Integration")
    print("=" * 70)

    # Simulate chunks from previous pipeline stages
    chunks = [
        "def calculate_sum(a, b): return a + b",
        "def calculate_product(a, b): return a * b",
        "class User: def __init__(self, name): self.name = name",
        "class Database: def connect(self): pass",
        "import numpy as np",
        "import pandas as pd",
        "SELECT * FROM users WHERE active = true",
        "UPDATE products SET price = price * 1.1",
    ]

    print(f"\nProcessing {len(chunks)} code chunks through full pipeline...")

    # Step 1: Embed
    print("\n[1/3] Embedding...")
    embeddings = embed_chunks(
        chunks,
        model_name="google/embeddinggemma-300m",
        normalize=True,  # Important for cosine distance
        show_progress=True,
    )

    # Step 2: Find optimal clusters
    print("\n[2/3] Finding optimal number of clusters...")
    optimal_k, _ = find_optimal_k(
        embeddings,
        k_range=range(2, 5),
        metric="silhouette",
        verbose=True,
    )

    # Step 3: Cluster
    print(f"\n[3/3] Clustering with k={optimal_k}...")
    result = cluster_embeddings(
        embeddings,
        algorithm="kmeans",
        n_clusters=optimal_k,
        verbose=True,
    )

    # Display final results
    print("\n" + "-" * 70)
    print("Final Clusters")
    print("-" * 70)

    for cluster_id in sorted(result.get_cluster_sizes().keys()):
        members = result.get_cluster_members(cluster_id)
        print(f"\nCluster {cluster_id}: {len(members)} chunks")
        for idx in members:
            print(f"  [{idx}] {chunks[idx][:60]}...")

    # Save results
    print("\n" + "-" * 70)
    print("Saving Results")
    print("-" * 70)

    from cluster import save_cluster_result
    save_cluster_result(result, "/tmp/doctown_clusters.npz")

    print("\n✓ Pipeline complete!")


def main():
    """Run all examples."""
    print("\n" * 2)
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "DOCTOWN v11 - CLUSTER EXAMPLES" + " " * 23 + "║")
    print("╚" + "═" * 68 + "╝")

    examples = [
        ("Basic KMeans Clustering", example_basic_clustering),
        ("Finding Optimal K", example_optimal_k),
        ("HDBSCAN Clustering", example_hdbscan),
        ("Full Pipeline Integration", example_integration),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning all examples...")

    for name, func in examples:
        try:
            func()
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user")
            break
        except Exception as e:
            print(f"\n⚠ Example '{name}' failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("✓ All examples completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
