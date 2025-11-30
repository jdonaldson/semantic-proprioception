#!/usr/bin/env python3
"""
Bootstrapped LSH: Learn hash functions from dense bucket structure

Two-pass approach:
1. Random LSH finds dense buckets (cheap)
2. Extract semantic prototypes from dense buckets
3. Use prototypes as learned hash directions

This exploits the natural structure of embeddings without expensive PCA on all data.
"""

import numpy as np
from typing import List, Literal
from krapivin_hash_rs import PyLSHIndex


class BootstrappedKrapivinLSH:
    """
    Two-pass LSH that learns hash functions from data structure

    Example:
        >>> learner = BootstrappedKrapivinLSH(embedding_dim=384)
        >>> learner.fit(embeddings, method='contrastive', min_density=5)
        >>> hash_value = learner.hash(new_embedding)
    """

    def __init__(self, embedding_dim: int = 384, seed: int = 12345):
        """
        Initialize bootstrapped LSH

        Args:
            embedding_dim: Dimensionality of embeddings
            seed: Random seed for reproducibility
        """
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.hash_directions = None
        self.method = None

    def fit(
        self,
        embeddings: List[np.ndarray],
        method: Literal['contrastive', 'centroids', 'lda', 'spherical'] = 'contrastive',
        min_density: int = 5,
        bootstrap_bits: int = 6
    ):
        """
        Learn hash functions from embedding structure

        Args:
            embeddings: List of embedding vectors (normalized)
            method: Learning method:
                - 'contrastive': Dense bucket centroids (fastest)
                - 'centroids': K-means on centroids (more hash functions)
                - 'lda': Linear discriminant analysis (best separation)
                - 'spherical': Spherical k-means (best for normalized)
            min_density: Minimum bucket size to consider "dense"
            bootstrap_bits: Bits for initial random LSH (lower = denser buckets)

        Returns:
            self (for chaining)
        """
        embeddings = np.array(embeddings)
        n_embeddings = len(embeddings)

        print("=== Bootstrapped LSH Learning ===\n")

        # PASS 1: Random LSH to find structure
        print(f"Pass 1: Bootstrap with random LSH ({bootstrap_bits} bits)...")
        bootstrap = PyLSHIndex(
            seed=self.seed,
            num_bits=bootstrap_bits,
            embedding_dim=self.embedding_dim,
            capacity=n_embeddings,
            delta=0.3
        )

        for i, emb in enumerate(embeddings):
            bootstrap.add_embedding(emb.tolist(), "bootstrap", i)

        dense_buckets = bootstrap.get_dense_buckets(min_count=min_density)
        n_dense = len(dense_buckets)
        n_in_dense = sum(count for _, count in dense_buckets)

        print(f"  Found {n_dense} dense buckets")
        print(f"  Items in dense buckets: {n_in_dense}/{n_embeddings} ({n_in_dense/n_embeddings:.1%})")

        if n_dense == 0:
            print("  ⚠️  No dense buckets found! Using random LSH instead.")
            self.hash_directions = None
            self.method = 'random'
            return self

        # PASS 2: Learn from dense buckets
        print(f"\nPass 2: Learning hash functions (method={method})...")

        if method == 'contrastive':
            self.hash_directions = self._learn_contrastive(
                embeddings, bootstrap, min_density
            )
        elif method == 'centroids':
            self.hash_directions = self._learn_centroids(
                embeddings, bootstrap, min_density
            )
        elif method == 'lda':
            self.hash_directions = self._learn_lda(
                embeddings, bootstrap, min_density
            )
        elif method == 'spherical':
            self.hash_directions = self._learn_spherical(
                embeddings, bootstrap, min_density
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        self.method = method

        print(f"\n✓ Learned {len(self.hash_directions)} hash directions")
        print(f"  Method: {method}")
        print(f"  Coverage: {n_in_dense} embeddings used for learning")

        return self

    def _learn_contrastive(
        self,
        embeddings: np.ndarray,
        bootstrap: PyLSHIndex,
        min_density: int
    ) -> np.ndarray:
        """
        Dense bucket centroids as hash directions (fastest)

        Each centroid represents a semantic prototype.
        """
        dense_buckets = bootstrap.get_dense_buckets(min_count=min_density)
        centroids = []

        for bucket_id, count in dense_buckets:
            refs = bootstrap.get_bucket_contents(bucket_id)
            bucket_embeddings = [embeddings[ref.row_id] for ref in refs]

            # Compute centroid
            centroid = np.mean(bucket_embeddings, axis=0)

            # Normalize (project to unit sphere)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
                centroids.append(centroid)

        print(f"  Extracted {len(centroids)} semantic prototypes (centroids)")

        return np.array(centroids)

    def _learn_centroids(
        self,
        embeddings: np.ndarray,
        bootstrap: PyLSHIndex,
        min_density: int,
        n_clusters: int = 256
    ) -> np.ndarray:
        """
        K-means clustering on dense bucket centroids

        Provides more hash functions than raw centroids.
        """
        from sklearn.cluster import KMeans

        # Get initial centroids
        centroids = self._learn_contrastive(embeddings, bootstrap, min_density)

        # Cluster if we have enough
        n_clusters = min(n_clusters, len(centroids))

        if n_clusters < len(centroids):
            print(f"  Clustering {len(centroids)} centroids → {n_clusters} hash functions")
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=self.seed)
            kmeans.fit(centroids)

            # Normalize cluster centers
            hash_dirs = kmeans.cluster_centers_
            for i in range(len(hash_dirs)):
                norm = np.linalg.norm(hash_dirs[i])
                if norm > 0:
                    hash_dirs[i] /= norm
        else:
            hash_dirs = centroids
            print(f"  Using {len(centroids)} centroids directly (< {n_clusters})")

        return hash_dirs

    def _learn_lda(
        self,
        embeddings: np.ndarray,
        bootstrap: PyLSHIndex,
        min_density: int
    ) -> np.ndarray:
        """
        Linear Discriminant Analysis on dense bucket embeddings

        Finds directions that maximally separate dense buckets.
        Best for discriminative power.
        """
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        dense_buckets = bootstrap.get_dense_buckets(min_count=min_density)

        # Build labeled dataset
        X, y = [], []
        for bucket_idx, (bucket_id, count) in enumerate(dense_buckets):
            refs = bootstrap.get_bucket_contents(bucket_id)
            for ref in refs:
                X.append(embeddings[ref.row_id])
                y.append(bucket_idx)

        X = np.array(X)
        y = np.array(y)

        print(f"  Training LDA on {len(X)} embeddings from {len(dense_buckets)} classes")

        # LDA can find at most (n_classes - 1) components
        n_components = min(len(dense_buckets) - 1, 32)

        lda = LinearDiscriminantAnalysis(n_components=n_components)
        lda.fit(X, y)

        # Extract discriminant directions
        hash_dirs = lda.scalings_.T  # (n_components, embedding_dim)

        # Normalize
        for i in range(len(hash_dirs)):
            norm = np.linalg.norm(hash_dirs[i])
            if norm > 0:
                hash_dirs[i] /= norm

        print(f"  LDA explained variance: {lda.explained_variance_ratio_.sum():.1%}")

        return hash_dirs

    def _learn_spherical(
        self,
        embeddings: np.ndarray,
        bootstrap: PyLSHIndex,
        min_density: int,
        n_clusters: int = 256,
        samples_per_bucket: int = 10
    ) -> np.ndarray:
        """
        Spherical k-means on dense bucket samples

        Designed for normalized embeddings (unit vectors).
        Uses cosine similarity instead of Euclidean distance.
        """
        dense_buckets = bootstrap.get_dense_buckets(min_count=min_density)

        # Sample representatives from dense buckets
        representatives = []

        for bucket_id, count in dense_buckets:
            refs = bootstrap.get_bucket_contents(bucket_id)
            bucket_embeddings = [embeddings[ref.row_id] for ref in refs]

            # Sample up to samples_per_bucket
            n_samples = min(samples_per_bucket, len(bucket_embeddings))
            sampled_indices = np.random.choice(
                len(bucket_embeddings), n_samples, replace=False
            )

            for idx in sampled_indices:
                representatives.append(bucket_embeddings[idx])

        representatives = np.array(representatives)
        print(f"  Sampled {len(representatives)} representatives from {len(dense_buckets)} dense buckets")

        # Spherical k-means
        n_clusters = min(n_clusters, len(representatives) // 2)
        print(f"  Running spherical k-means (k={n_clusters})...")

        hash_dirs = self._spherical_kmeans(representatives, n_clusters)

        return hash_dirs

    def _spherical_kmeans(
        self,
        X: np.ndarray,
        n_clusters: int,
        max_iter: int = 20
    ) -> np.ndarray:
        """
        Simple spherical k-means implementation

        Uses cosine similarity for assignment (not Euclidean distance).
        Centers are normalized to unit sphere.
        """
        # Initialize with random samples
        indices = np.random.choice(len(X), n_clusters, replace=False)
        centers = X[indices].copy()

        for iteration in range(max_iter):
            # Assignment: find nearest center by cosine similarity
            similarities = X @ centers.T  # (n_samples, n_clusters)
            labels = np.argmax(similarities, axis=1)

            # Update centers: mean of assigned points, projected to sphere
            new_centers = []
            for k in range(n_clusters):
                cluster_points = X[labels == k]

                if len(cluster_points) > 0:
                    # Compute mean
                    new_center = np.mean(cluster_points, axis=0)

                    # Normalize (project to unit sphere)
                    norm = np.linalg.norm(new_center)
                    if norm > 0:
                        new_center = new_center / norm

                    new_centers.append(new_center)
                else:
                    # Empty cluster - keep old center
                    new_centers.append(centers[k])

            centers = np.array(new_centers)

        return centers

    def hash(self, embedding: np.ndarray) -> int:
        """
        Hash an embedding using learned directions

        Args:
            embedding: Embedding vector (normalized)

        Returns:
            Hash value (index of nearest hash direction)
        """
        if self.hash_directions is None:
            raise ValueError("Must call fit() first to learn hash directions")

        embedding = np.array(embedding)

        # Find nearest hash direction (highest cosine similarity)
        similarities = self.hash_directions @ embedding
        return int(np.argmax(similarities))

    def hash_batch(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Hash multiple embeddings at once (faster)

        Args:
            embeddings: List of embedding vectors

        Returns:
            Array of hash values
        """
        if self.hash_directions is None:
            raise ValueError("Must call fit() first to learn hash directions")

        embeddings = np.array(embeddings)

        # Batch compute: (n_embeddings, dim) @ (dim, n_directions)^T
        similarities = embeddings @ self.hash_directions.T
        return np.argmax(similarities, axis=1)

    def save(self, path: str):
        """
        Save learned hash directions to disk

        Args:
            path: Output path (recommended extension: .khash)
                  Example: "twitter_support.lda.khash"

        File format: .khash (internally NumPy .npz) containing:
            - hash_directions: (n_functions, embedding_dim) array
            - method: Learning method used
            - embedding_dim: Dimension of embeddings
            - seed: Random seed for reproducibility
        """
        if self.hash_directions is None:
            raise ValueError("No hash directions to save (must call fit() first)")

        from pathlib import Path
        import os

        path_obj = Path(path)

        # Ensure .khash extension
        if path_obj.suffix != '.khash':
            path_obj = path_obj.with_suffix('.khash')

        # Save to temporary .npz file (np.savez adds .npz)
        temp_path = path_obj.with_suffix('')  # Remove .khash

        np.savez(
            str(temp_path),
            hash_directions=self.hash_directions,
            method=self.method,
            embedding_dim=self.embedding_dim,
            seed=self.seed
        )

        # Rename .npz to .khash
        npz_path = Path(str(temp_path) + '.npz')
        if npz_path.exists():
            npz_path.rename(path_obj)

        print(f"✓ Saved {len(self.hash_directions)} hash directions to {path_obj}")

    def load(self, path: str):
        """
        Load learned hash directions from disk

        Args:
            path: Path to .khash file

        Returns:
            self (for chaining)
        """
        data = np.load(path, allow_pickle=True)

        self.hash_directions = data['hash_directions']
        self.method = str(data['method'])
        self.embedding_dim = int(data['embedding_dim'])
        self.seed = int(data['seed'])

        print(f"✓ Loaded {len(self.hash_directions)} hash directions ({self.method})")

        return self


def compare_lsh_quality(
    embeddings: List[np.ndarray],
    random_index: PyLSHIndex,
    learned_lsh: BootstrappedKrapivinLSH,
    min_density: int = 5
):
    """
    Compare random vs learned LSH quality

    Metrics:
    - Number of dense buckets
    - Average bucket density
    - Coverage (% of embeddings in dense buckets)
    """
    embeddings = np.array(embeddings)

    print("=== LSH Quality Comparison ===\n")

    # Random LSH stats
    random_dense = random_index.get_dense_buckets(min_count=min_density)
    random_n_dense = len(random_dense)
    random_coverage = sum(count for _, count in random_dense)

    print(f"Random LSH ({random_index.stats().num_buckets} buckets):")
    print(f"  Dense buckets (≥{min_density}): {random_n_dense}")
    print(f"  Coverage: {random_coverage}/{len(embeddings)} ({random_coverage/len(embeddings):.1%})")

    if random_n_dense > 0:
        avg_density = random_coverage / random_n_dense
        print(f"  Avg dense bucket size: {avg_density:.1f}")

    # Learned LSH stats
    if learned_lsh.hash_directions is not None:
        print(f"\nLearned LSH ({len(learned_lsh.hash_directions)} hash functions):")

        # Hash all embeddings
        learned_hashes = learned_lsh.hash_batch(embeddings)

        # Count bucket sizes
        from collections import Counter
        bucket_sizes = Counter(learned_hashes)

        # Find dense buckets
        learned_dense = [(bid, size) for bid, size in bucket_sizes.items()
                         if size >= min_density]
        learned_n_dense = len(learned_dense)
        learned_coverage = sum(size for _, size in learned_dense)

        print(f"  Dense buckets (≥{min_density}): {learned_n_dense}")
        print(f"  Coverage: {learned_coverage}/{len(embeddings)} ({learned_coverage/len(embeddings):.1%})")

        if learned_n_dense > 0:
            avg_density = learned_coverage / learned_n_dense
            print(f"  Avg dense bucket size: {avg_density:.1f}")

        # Improvement
        print(f"\n✓ Improvement:")
        if random_n_dense > 0:
            density_improvement = (learned_coverage / learned_n_dense) / (random_coverage / random_n_dense)
            print(f"  Avg density: {density_improvement:.2f}x better")

        coverage_improvement = learned_coverage / random_coverage
        print(f"  Coverage: {coverage_improvement:.2f}x better")
