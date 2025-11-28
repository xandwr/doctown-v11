# cluster/__init__.py
# Doctown v11 – Cluster subsystem

from .cluster import (
    # Main functions
    cluster_embeddings,
    cluster_kmeans,
    cluster_hdbscan,

    # Result class
    ClusterResult,

    # Utilities
    save_cluster_result,
    load_cluster_result,
    find_optimal_k,

    # Constants
    DEFAULT_ALGORITHM,
    DEFAULT_N_CLUSTERS,
)

__all__ = [
    # Main functions
    "cluster_embeddings",
    "cluster_kmeans",
    "cluster_hdbscan",

    # Result class
    "ClusterResult",

    # Utilities
    "save_cluster_result",
    "load_cluster_result",
    "find_optimal_k",

    # Constants
    "DEFAULT_ALGORITHM",
    "DEFAULT_N_CLUSTERS",
]
