# cluster.py
# Doctown v11 – Cluster subsystem: Group embeddings into coherent topic clusters

import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score


# ============================================================
# Configuration
# ============================================================

# Default clustering algorithm
DEFAULT_ALGORITHM: Literal["kmeans", "hdbscan"] = "kmeans"

# Default number of clusters for KMeans
DEFAULT_N_CLUSTERS = 5

# KMeans parameters
DEFAULT_MAX_ITER = 200
DEFAULT_N_INIT = "auto"
DEFAULT_RANDOM_STATE = 42

# HDBSCAN parameters
DEFAULT_MIN_CLUSTER_SIZE = 5
DEFAULT_MIN_SAMPLES = 3


# ============================================================
# Cluster Results
# ============================================================

@dataclass
class ClusterResult:
    """
    Result of clustering operation.

    Attributes:
        labels: Cluster assignment for each embedding (shape: N)
                -1 indicates noise points (for HDBSCAN)
        centroids: Cluster centroids (shape: K, D) where K is number of clusters
                   None for HDBSCAN
        n_clusters: Number of clusters found
        algorithm: Algorithm used ("kmeans" or "hdbscan")
        silhouette: Silhouette score (quality metric, -1 to 1, higher is better)
                    None if less than 2 clusters or computation fails
        inertia: Within-cluster sum of squares (KMeans only, lower is better)
                 None for HDBSCAN
    """
    labels: np.ndarray
    centroids: Optional[np.ndarray]
    n_clusters: int
    algorithm: str
    silhouette: Optional[float] = None
    inertia: Optional[float] = None

    def __repr__(self) -> str:
        parts = [
            f"ClusterResult(algorithm={self.algorithm}",
            f"n_clusters={self.n_clusters}",
        ]
        if self.silhouette is not None:
            parts.append(f"silhouette={self.silhouette:.3f}")
        if self.inertia is not None:
            parts.append(f"inertia={self.inertia:.1f}")
        parts.append(f"labels={self.labels.shape})")
        return ", ".join(parts)

    def get_cluster_sizes(self) -> dict[int, int]:
        """
        Get the size of each cluster.

        Returns:
            Dictionary mapping cluster_id -> count
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def get_cluster_members(self, cluster_id: int) -> np.ndarray:
        """
        Get indices of all members of a cluster.

        Args:
            cluster_id: ID of the cluster

        Returns:
            Array of indices belonging to this cluster
        """
        return np.where(self.labels == cluster_id)[0]


# ============================================================
# KMeans Clustering
# ============================================================

def cluster_kmeans(
    embeddings: np.ndarray,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    max_iter: int = DEFAULT_MAX_ITER,
    n_init: str = DEFAULT_N_INIT,
    random_state: int = DEFAULT_RANDOM_STATE,
    verbose: bool = True,
) -> ClusterResult:
    """
    Cluster embeddings using KMeans with cosine distance support.

    The embeddings should be L2-normalized for cosine distance.
    If using embed.embed_chunks() with normalize=True, they already are.

    Args:
        embeddings: Normalized embeddings of shape (N, D)
        n_clusters: Number of clusters to create
        max_iter: Maximum number of iterations
        n_init: Number of times the k-means algorithm will be run
                with different centroid seeds ('auto' recommended)
        random_state: Random seed for reproducibility
        verbose: Print clustering information

    Returns:
        ClusterResult with labels and centroids
    """
    if embeddings.shape[0] < n_clusters:
        raise ValueError(
            f"Number of samples ({embeddings.shape[0]}) must be >= "
            f"n_clusters ({n_clusters})"
        )

    if verbose:
        print(f"\nClustering {embeddings.shape[0]} embeddings into {n_clusters} clusters...")
        print(f"Using KMeans with cosine distance (normalized embeddings)")

    # Normalize embeddings for cosine distance
    # If already normalized (from embed subsystem), this is a no-op
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normed = embeddings / norms

    # KMeans clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        verbose=0,
    )

    labels = kmeans.fit_predict(normed)
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_

    # Compute silhouette score if we have enough clusters
    silhouette = None
    if n_clusters > 1 and n_clusters < embeddings.shape[0]:
        try:
            silhouette = silhouette_score(normed, labels, metric='cosine')
            if verbose:
                print(f"✓ Silhouette score: {silhouette:.3f} (range: -1 to 1, higher is better)")
        except Exception as e:
            if verbose:
                print(f"⚠ Could not compute silhouette score: {e}")

    if verbose:
        print(f"✓ Clustering complete")
        print(f"  Inertia: {inertia:.2f}")

        # Print cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Cluster sizes:")
        for cluster_id, count in zip(unique, counts):
            print(f"    Cluster {cluster_id}: {count} items")

    return ClusterResult(
        labels=labels,
        centroids=centroids,
        n_clusters=n_clusters,
        algorithm="kmeans",
        silhouette=silhouette,
        inertia=inertia,
    )


# ============================================================
# HDBSCAN Clustering
# ============================================================

def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    metric: str = "cosine",
    verbose: bool = True,
) -> ClusterResult:
    """
    Cluster embeddings using HDBSCAN (density-based clustering).

    HDBSCAN automatically determines the number of clusters and can
    mark points as noise (label -1).

    Args:
        embeddings: Embeddings of shape (N, D)
        min_cluster_size: Minimum size of clusters
        min_samples: Number of samples in a neighborhood for a point
                     to be considered a core point
        metric: Distance metric (default: "cosine")
        verbose: Print clustering information

    Returns:
        ClusterResult with labels (no centroids for HDBSCAN)
    """
    if verbose:
        print(f"\nClustering {embeddings.shape[0]} embeddings with HDBSCAN...")
        print(f"Parameters: min_cluster_size={min_cluster_size}, "
              f"min_samples={min_samples}, metric={metric}")

    # HDBSCAN clustering
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )

    labels = clusterer.fit_predict(embeddings)

    # Count actual clusters (excluding noise label -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    # Compute silhouette score if we have enough clusters
    silhouette = None
    if n_clusters > 1:
        # Only compute on non-noise points
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) > n_clusters:
            try:
                silhouette = silhouette_score(
                    embeddings[non_noise_mask],
                    labels[non_noise_mask],
                    metric=metric
                )
                if verbose:
                    print(f"✓ Silhouette score: {silhouette:.3f} (on non-noise points)")
            except Exception as e:
                if verbose:
                    print(f"⚠ Could not compute silhouette score: {e}")

    if verbose:
        print(f"✓ Clustering complete")
        print(f"  Clusters found: {n_clusters}")
        print(f"  Noise points: {n_noise}")

        # Print cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Cluster sizes:")
        for cluster_id, count in zip(unique, counts):
            if cluster_id == -1:
                print(f"    Noise: {count} items")
            else:
                print(f"    Cluster {cluster_id}: {count} items")

    return ClusterResult(
        labels=labels,
        centroids=None,  # HDBSCAN doesn't produce centroids
        n_clusters=n_clusters,
        algorithm="hdbscan",
        silhouette=silhouette,
        inertia=None,
    )


# ============================================================
# Main Clustering Function
# ============================================================

def cluster_embeddings(
    embeddings: np.ndarray,
    algorithm: Literal["kmeans", "hdbscan"] = DEFAULT_ALGORITHM,
    n_clusters: Optional[int] = None,
    **kwargs
) -> ClusterResult:
    """
    Cluster embeddings using the specified algorithm.

    This is the main function to use from other subsystems.

    Args:
        embeddings: Embeddings array of shape (N, D)
        algorithm: Clustering algorithm ("kmeans" or "hdbscan")
        n_clusters: Number of clusters (KMeans only)
                    If None, uses DEFAULT_N_CLUSTERS
        **kwargs: Additional arguments passed to the clustering function

    Returns:
        ClusterResult with labels and metadata

    Examples:
        # KMeans with 5 clusters
        result = cluster_embeddings(embeddings, algorithm="kmeans", n_clusters=5)

        # HDBSCAN with automatic cluster detection
        result = cluster_embeddings(embeddings, algorithm="hdbscan")
    """
    if embeddings.shape[0] == 0:
        raise ValueError("Cannot cluster empty embeddings array")

    if algorithm == "kmeans":
        if n_clusters is None:
            n_clusters = DEFAULT_N_CLUSTERS
        return cluster_kmeans(embeddings, n_clusters=n_clusters, **kwargs)

    elif algorithm == "hdbscan":
        return cluster_hdbscan(embeddings, **kwargs)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'kmeans' or 'hdbscan'")


# ============================================================
# Utility Functions
# ============================================================

def save_cluster_result(result: ClusterResult, filepath: str) -> None:
    """
    Save cluster result to disk.

    Args:
        result: ClusterResult to save
        filepath: Path to save file (.npz format)
    """
    data = {
        "labels": result.labels,
        "algorithm": np.array([result.algorithm]),  # Store as array for npz
        "n_clusters": np.array([result.n_clusters]),
    }

    if result.centroids is not None:
        data["centroids"] = result.centroids

    if result.silhouette is not None:
        data["silhouette"] = np.array([result.silhouette])

    if result.inertia is not None:
        data["inertia"] = np.array([result.inertia])

    np.savez(filepath, **data)
    print(f"✓ Saved cluster result to {filepath}")


def load_cluster_result(filepath: str) -> ClusterResult:
    """
    Load cluster result from disk.

    Args:
        filepath: Path to .npz file

    Returns:
        ClusterResult
    """
    data = np.load(filepath, allow_pickle=True)

    result = ClusterResult(
        labels=data["labels"],
        centroids=data.get("centroids", None),
        n_clusters=int(data["n_clusters"][0]),
        algorithm=str(data["algorithm"][0]),
        silhouette=float(data["silhouette"][0]) if "silhouette" in data else None,
        inertia=float(data["inertia"][0]) if "inertia" in data else None,
    )

    print(f"✓ Loaded cluster result from {filepath}")
    print(f"  {result}")

    return result


def find_optimal_k(
    embeddings: np.ndarray,
    k_range: range = range(2, 11),
    metric: Literal["silhouette", "inertia"] = "silhouette",
    verbose: bool = True,
) -> tuple[int, list[float]]:
    """
    Find optimal number of clusters by trying different values of k.

    Args:
        embeddings: Embeddings array of shape (N, D)
        k_range: Range of k values to try
        metric: Metric to optimize ("silhouette" or "inertia")
                silhouette: higher is better
                inertia: lower is better
        verbose: Print progress

    Returns:
        Tuple of (optimal_k, scores_list)
    """
    if verbose:
        print(f"\nFinding optimal k using {metric} score...")
        print(f"Testing k in range {list(k_range)}")

    scores = []
    for k in k_range:
        if verbose:
            print(f"  Testing k={k}...", end=" ")

        result = cluster_kmeans(
            embeddings,
            n_clusters=k,
            verbose=False,
        )

        if metric == "silhouette":
            score = result.silhouette if result.silhouette is not None else -1
        else:  # inertia
            score = -result.inertia if result.inertia is not None else float('inf')

        scores.append(score)

        if verbose:
            print(f"{metric}={score:.3f}")

    # Find best k
    optimal_idx = np.argmax(scores)
    optimal_k = list(k_range)[optimal_idx]

    if verbose:
        print(f"\n✓ Optimal k: {optimal_k} ({metric}={scores[optimal_idx]:.3f})")

    return optimal_k, scores


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":
    print("Testing cluster subsystem...")
    print("=" * 60)

    # Generate synthetic embeddings for testing
    print("\n[Setup] Generating synthetic embeddings")
    print("-" * 60)

    np.random.seed(42)

    # Create 3 clusters of embeddings
    n_per_cluster = 20
    dim = 128

    # Cluster 1: around (1, 0, 0, ...)
    cluster1 = np.random.randn(n_per_cluster, dim) * 0.1
    cluster1[:, 0] += 1

    # Cluster 2: around (0, 1, 0, ...)
    cluster2 = np.random.randn(n_per_cluster, dim) * 0.1
    cluster2[:, 1] += 1

    # Cluster 3: around (0, 0, 1, ...)
    cluster3 = np.random.randn(n_per_cluster, dim) * 0.1
    cluster3[:, 2] += 1

    # Combine and normalize
    embeddings = np.vstack([cluster1, cluster2, cluster3])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    print(f"✓ Generated {embeddings.shape[0]} embeddings with {embeddings.shape[1]} dimensions")
    print(f"  True clusters: 3 clusters of {n_per_cluster} items each")

    # Test 1: KMeans clustering
    print("\n\n[Test 1] KMeans clustering")
    print("-" * 60)

    result_kmeans = cluster_embeddings(
        embeddings,
        algorithm="kmeans",
        n_clusters=3,
    )

    print(f"\nResult: {result_kmeans}")
    print(f"Cluster sizes: {result_kmeans.get_cluster_sizes()}")

    # Test 2: HDBSCAN clustering
    print("\n\n[Test 2] HDBSCAN clustering")
    print("-" * 60)

    result_hdbscan = cluster_embeddings(
        embeddings,
        algorithm="hdbscan",
        min_cluster_size=5,
    )

    print(f"\nResult: {result_hdbscan}")
    print(f"Cluster sizes: {result_hdbscan.get_cluster_sizes()}")

    # Test 3: Get cluster members
    print("\n\n[Test 3] Get cluster members")
    print("-" * 60)

    cluster_0_members = result_kmeans.get_cluster_members(0)
    print(f"Cluster 0 members (first 10): {cluster_0_members[:10]}")
    print(f"Total members in cluster 0: {len(cluster_0_members)}")

    # Test 4: Save and load
    print("\n\n[Test 4] Save and load cluster result")
    print("-" * 60)

    test_file = "/tmp/test_clusters.npz"
    save_cluster_result(result_kmeans, test_file)
    loaded = load_cluster_result(test_file)

    assert np.array_equal(result_kmeans.labels, loaded.labels), "Labels don't match!"
    assert result_kmeans.n_clusters == loaded.n_clusters, "n_clusters doesn't match!"
    print("✓ Save/load test passed")

    # Test 5: Find optimal k
    print("\n\n[Test 5] Find optimal k")
    print("-" * 60)

    optimal_k, scores = find_optimal_k(
        embeddings,
        k_range=range(2, 6),
        metric="silhouette",
    )

    print(f"Silhouette scores: {[f'{s:.3f}' for s in scores]}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
