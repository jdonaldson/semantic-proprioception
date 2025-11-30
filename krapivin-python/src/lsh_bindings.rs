//! Python bindings for LSH index

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyIOError};
use pyo3::types::PyList;
use krapivin_lsh::{LSHIndex, FileRef, index_parquet_files};
use std::path::PathBuf;

/// File reference returned from LSH queries
#[pyclass]
#[derive(Clone)]
pub struct PyFileRef {
    #[pyo3(get)]
    pub file_path: String,
    #[pyo3(get)]
    pub row_id: u64,
}

#[pymethods]
impl PyFileRef {
    fn __repr__(&self) -> String {
        format!("FileRef(file='{}', row={})", self.file_path, self.row_id)
    }

    fn __str__(&self) -> String {
        format!("{}:{}", self.file_path, self.row_id)
    }
}

impl From<FileRef> for PyFileRef {
    fn from(fr: FileRef) -> Self {
        PyFileRef {
            file_path: fr.file_path,
            row_id: fr.row_id,
        }
    }
}

impl From<&FileRef> for PyFileRef {
    fn from(fr: &FileRef) -> Self {
        PyFileRef {
            file_path: fr.file_path.clone(),
            row_id: fr.row_id,
        }
    }
}

/// LSH index statistics
#[pyclass]
#[derive(Clone)]
pub struct PyIndexStats {
    #[pyo3(get)]
    pub num_buckets: usize,
    #[pyo3(get)]
    pub num_files: usize,
    #[pyo3(get)]
    pub load_factor: f64,
    #[pyo3(get)]
    pub level_densities: Vec<f64>,
}

#[pymethods]
impl PyIndexStats {
    fn __repr__(&self) -> String {
        format!(
            "IndexStats(buckets={}, files={}, load={:.2})",
            self.num_buckets, self.num_files, self.load_factor
        )
    }
}

/// LSH Index for embedding vectors
///
/// Enables fast approximate nearest neighbor search across multiple Parquet files
/// containing embedding vectors from language models.
///
/// # Example
///
/// ```python
/// from krapivin_hash_rs import PyLSHIndex
///
/// # Create index
/// index = PyLSHIndex(
///     seed=12345,
///     num_bits=16,
///     embedding_dim=384,
///     capacity=10000,
///     delta=0.3
/// )
///
/// # Index Parquet files
/// count = index.add_parquet_files(
///     ["embeddings1.parquet", "embeddings2.parquet"],
///     column="embedding"
/// )
///
/// # Query
/// results = index.query([0.1] * 384)
///
/// # Save/load
/// index.save("index.krapivin")
/// index2 = PyLSHIndex.load("index.krapivin")
/// ```
#[pyclass]
pub struct PyLSHIndex {
    index: LSHIndex,
}

#[pymethods]
impl PyLSHIndex {
    /// Create new LSH index
    ///
    /// # Arguments
    /// * `seed` - Fixed seed for LSH hyperplanes (use same seed to merge indexes)
    /// * `num_bits` - Number of LSH hash bits (typically 16-64)
    /// * `embedding_dim` - Dimensionality of embedding vectors
    /// * `capacity` - Hash table capacity
    /// * `delta` - Empty fraction parameter (0 < δ < 1, default 0.3)
    #[new]
    #[pyo3(signature = (seed, num_bits, embedding_dim, capacity, delta=0.3))]
    fn new(
        seed: u64,
        num_bits: usize,
        embedding_dim: usize,
        capacity: usize,
        delta: f64,
    ) -> PyResult<Self> {
        if delta <= 0.0 || delta >= 1.0 {
            return Err(PyValueError::new_err("delta must be in (0, 1)"));
        }
        Ok(PyLSHIndex {
            index: LSHIndex::new(seed, num_bits, embedding_dim, capacity, delta),
        })
    }

    /// Add a single embedding to the index
    ///
    /// # Arguments
    /// * `embedding` - Embedding vector (must match embedding_dim)
    /// * `file_path` - Path to source file
    /// * `row_id` - Row ID in source file
    fn add_embedding(
        &mut self,
        embedding: Vec<f32>,
        file_path: String,
        row_id: u64,
    ) -> PyResult<()> {
        let file_ref = FileRef { file_path, row_id };
        self.index.add_embedding(&embedding, file_ref);
        Ok(())
    }

    /// Index embeddings from Parquet files
    ///
    /// # Arguments
    /// * `file_paths` - List of Parquet file paths
    /// * `column` - Name of column containing embeddings
    ///
    /// # Returns
    /// Total number of embeddings indexed
    fn add_parquet_files(
        &mut self,
        file_paths: Vec<String>,
        column: String,
    ) -> PyResult<usize> {
        let paths: Vec<PathBuf> = file_paths.iter().map(PathBuf::from).collect();

        index_parquet_files(&mut self.index, &paths, &column)
            .map_err(|e| PyIOError::new_err(format!("Failed to index Parquet files: {}", e)))
    }

    /// Query for similar embeddings
    ///
    /// # Arguments
    /// * `embedding` - Query embedding vector
    ///
    /// # Returns
    /// List of FileRef objects in the same LSH bucket, or None if bucket is empty
    fn query(&self, embedding: Vec<f32>) -> PyResult<Option<Vec<PyFileRef>>> {
        let results = self.index.query(&embedding);
        Ok(results.map(|refs| refs.iter().map(PyFileRef::from).collect()))
    }

    /// Get index statistics
    fn stats(&self) -> PyResult<PyIndexStats> {
        let stats = self.index.stats();
        Ok(PyIndexStats {
            num_buckets: stats.num_buckets,
            num_files: stats.num_files,
            load_factor: stats.load_factor,
            level_densities: stats.level_densities,
        })
    }

    /// Get list of indexed files
    fn indexed_files(&self) -> PyResult<Vec<String>> {
        Ok(self.index.indexed_files().to_vec())
    }

    /// Save index to disk
    ///
    /// # Arguments
    /// * `path` - Path to save .krapivin file
    fn save(&self, path: String) -> PyResult<()> {
        self.index
            .save(&path)
            .map_err(|e| PyIOError::new_err(format!("Failed to save index: {}", e)))
    }

    /// Load index from disk
    ///
    /// # Arguments
    /// * `path` - Path to .krapivin file
    ///
    /// # Returns
    /// New PyLSHIndex instance
    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        let index = LSHIndex::load(&path)
            .map_err(|e| PyIOError::new_err(format!("Failed to load index: {}", e)))?;
        Ok(PyLSHIndex { index })
    }

    fn __repr__(&self) -> String {
        let stats = self.index.stats();
        format!(
            "PyLSHIndex(buckets={}, files={}, load={:.2})",
            stats.num_buckets, stats.num_files, stats.load_factor
        )
    }

    fn __str__(&self) -> String {
        let stats = self.index.stats();
        format!(
            "LSH Index: {} buckets, {} files indexed, {:.1}% load",
            stats.num_buckets,
            stats.num_files,
            stats.load_factor * 100.0
        )
    }

    /// Get contents of a specific bucket
    ///
    /// # Arguments
    /// * `bucket_id` - LSH bucket ID
    ///
    /// # Returns
    /// List of FileRef objects in this bucket, or None if bucket is empty
    fn get_bucket_contents(&self, bucket_id: u64) -> PyResult<Option<Vec<PyFileRef>>> {
        Ok(self.index.get_bucket(bucket_id)
            .map(|refs| refs.iter().map(PyFileRef::from).collect()))
    }

    /// Get dense buckets above a threshold
    ///
    /// # Arguments
    /// * `min_count` - Minimum number of items in bucket
    ///
    /// # Returns
    /// List of (bucket_id, count) tuples for buckets with >= min_count items
    fn get_dense_buckets(&self, min_count: usize) -> PyResult<Vec<(u64, usize)>> {
        Ok(self.index.dense_buckets(min_count))
    }

    /// Get bucket size distribution
    ///
    /// # Returns
    /// Dictionary mapping bucket_size -> count_of_buckets_with_that_size
    fn bucket_size_histogram(&self) -> PyResult<std::collections::HashMap<usize, usize>> {
        Ok(self.index.bucket_size_histogram())
    }

    /// Iterate over all buckets
    ///
    /// # Returns
    /// List of (bucket_id, count) tuples for all non-empty buckets
    fn all_buckets(&self) -> PyResult<Vec<(u64, usize)>> {
        Ok(self.index.iter_buckets()
            .map(|(bucket_id, refs)| (bucket_id, refs.len()))
            .collect())
    }

    /// Export all index data for Parquet serialization
    ///
    /// # Returns
    /// List of (bucket_id, file_path, row_id) tuples for all indexed embeddings
    fn export_data(&self) -> PyResult<Vec<(u64, String, u64)>> {
        let mut records = Vec::new();

        for (bucket_id, refs) in self.index.iter_buckets() {
            for file_ref in refs {
                records.push((
                    bucket_id,
                    file_ref.file_path.clone(),
                    file_ref.row_id,
                ));
            }
        }

        Ok(records)
    }

    // =========================================================================
    // Multi-Resolution LSH Methods
    // =========================================================================

    /// Truncate a hash to a coarser resolution
    ///
    /// Multi-resolution LSH: store at full resolution, query at any coarser level.
    /// This is an O(1) bit operation.
    ///
    /// # Arguments
    /// * `hash` - Full resolution hash
    /// * `target_bits` - Desired resolution (must be <= num_bits)
    ///
    /// # Returns
    /// Hash truncated to target_bits
    ///
    /// # Example
    /// ```python
    /// hash_32 = index.hash_embedding(embedding)  # 32-bit
    /// hash_16 = index.truncate_hash(hash_32, 16)  # Coarser 16-bit
    /// hash_12 = index.truncate_hash(hash_32, 12)  # Even coarser 12-bit
    /// ```
    fn truncate_hash(&self, hash: u64, target_bits: usize) -> PyResult<u64> {
        Ok(self.index.lsh_hasher().truncate_hash(hash, target_bits))
    }

    /// Hash an embedding to get its bucket ID
    ///
    /// # Arguments
    /// * `embedding` - Embedding vector
    ///
    /// # Returns
    /// Bucket ID (at full resolution)
    fn hash_embedding(&self, embedding: Vec<f32>) -> PyResult<u64> {
        Ok(self.index.lsh_hasher().hash(&embedding))
    }

    /// Hash an embedding at a specific resolution
    ///
    /// More efficient than hash_embedding() + truncate_hash() when you only
    /// need the coarse resolution.
    ///
    /// # Arguments
    /// * `embedding` - Embedding vector
    /// * `resolution_bits` - Number of bits to compute
    ///
    /// # Returns
    /// Hash at the specified resolution
    fn hash_at_resolution(&self, embedding: Vec<f32>, resolution_bits: usize) -> PyResult<u64> {
        Ok(self.index.lsh_hasher().hash_at_resolution(&embedding, resolution_bits))
    }

    /// Calculate optimal query bits for a corpus size
    ///
    /// Multi-resolution strategy: aim for ~target_density docs per bucket.
    ///
    /// # Arguments
    /// * `corpus_size` - Number of documents in the corpus
    /// * `target_density` - Target average documents per bucket (default: 10)
    ///
    /// # Returns
    /// Optimal number of bits
    ///
    /// # Example
    /// ```python
    /// # 8.8M docs with target 10 docs/bucket → 20 bits
    /// optimal = index.optimal_query_bits(8_800_000, 10)
    /// ```
    #[pyo3(signature = (corpus_size, target_density=10))]
    fn optimal_query_bits(&self, corpus_size: usize, target_density: usize) -> PyResult<usize> {
        Ok(self.index.lsh_hasher().optimal_query_bits(corpus_size, target_density))
    }

    /// Get multi-resolution hash for an embedding
    ///
    /// Returns bucket IDs at multiple resolutions, enabling O(1) density
    /// queries at any scale.
    ///
    /// # Arguments
    /// * `embedding` - Embedding vector
    /// * `resolutions` - List of bit resolutions to compute (e.g., [12, 16, 20, 24])
    ///
    /// # Returns
    /// List of (resolution_bits, bucket_id) tuples
    ///
    /// # Example
    /// ```python
    /// hashes = index.multi_resolution_hash(embedding, [12, 16, 20, 24])
    /// # Returns: [(12, 3421), (16, 54321), (20, 870234), (24, 13923857)]
    /// ```
    fn multi_resolution_hash(&self, embedding: Vec<f32>, resolutions: Vec<usize>) -> PyResult<Vec<(usize, u64)>> {
        Ok(self.index.lsh_hasher().multi_resolution_hash(&embedding, &resolutions))
    }

    /// Get the number of LSH bits used by this index
    #[getter]
    fn num_bits(&self) -> PyResult<usize> {
        Ok(self.index.lsh_hasher().num_bits())
    }

    /// Get the number of hierarchical levels in the hash table
    #[getter]
    fn num_levels(&self) -> PyResult<usize> {
        Ok(self.index.stats().level_densities.len())
    }

    /// Get the load factor of the hash table
    #[getter]
    fn load_factor(&self) -> PyResult<f64> {
        Ok(self.index.stats().load_factor)
    }

    // =========================================================================
    // Hamming Distance and Multi-Probe Methods
    // =========================================================================

    /// Compute Hamming distance between two bucket IDs
    ///
    /// # Arguments
    /// * `a` - First bucket ID
    /// * `b` - Second bucket ID
    ///
    /// # Returns
    /// Number of differing bits
    ///
    /// # Example
    /// ```python
    /// dist = index.hamming_distance(0b1010, 0b1001)  # 2 bits differ
    /// ```
    fn hamming_distance(&self, a: u64, b: u64) -> PyResult<u32> {
        Ok(self.index.lsh_hasher().hamming_distance(a, b))
    }

    /// Flip a specific bit in a hash
    ///
    /// # Arguments
    /// * `hash` - Original hash
    /// * `bit_index` - Bit position to flip (0 = LSB)
    ///
    /// # Returns
    /// Hash with the specified bit flipped
    fn flip_bit(&self, hash: u64, bit_index: usize) -> PyResult<u64> {
        Ok(self.index.lsh_hasher().flip_bit(hash, bit_index))
    }

    /// Get all bucket IDs at exact Hamming distance from given bucket
    ///
    /// # Computational Cost (for n-bit hash)
    ///
    /// | Distance | Neighbors | Formula    | 16-bit | 32-bit |
    /// |----------|-----------|------------|--------|--------|
    /// | 1        | n         | C(n,1)     | 16     | 32     |
    /// | 2        | n(n-1)/2  | C(n,2)     | 120    | 496    |
    /// | 3        | n³/6      | C(n,3)     | 560    | 4,960  |
    ///
    /// Distance 3+ grows combinatorially. For most multi-probe applications,
    /// distance 1-2 provides good recall without excessive computation.
    ///
    /// # Arguments
    /// * `bucket_id` - Source bucket ID
    /// * `distance` - Exact Hamming distance (0-3 supported)
    ///
    /// # Returns
    /// List of neighbor bucket IDs at exactly the specified Hamming distance
    ///
    /// # Example
    /// ```python
    /// d1 = index.hamming_neighbors(bucket_id, 1)  # 16 neighbors for 16-bit
    /// d2 = index.hamming_neighbors(bucket_id, 2)  # 120 neighbors for 16-bit
    /// d3 = index.hamming_neighbors(bucket_id, 3)  # 560 neighbors for 16-bit
    /// ```
    fn hamming_neighbors(&self, bucket_id: u64, distance: usize) -> PyResult<Vec<u64>> {
        Ok(self.index.lsh_hasher().hamming_neighbors(bucket_id, distance))
    }

    /// Get all bucket IDs up to a given Hamming distance (cumulative)
    ///
    /// Returns all neighbors at distances 0 through max_distance, inclusive.
    /// This is useful for multi-probe LSH when you want to search nearby buckets.
    ///
    /// # Cumulative Cost (for n-bit hash)
    ///
    /// | max_distance | Total Buckets | 16-bit  | 32-bit  |
    /// |--------------|---------------|---------|---------|
    /// | 0            | 1             | 1       | 1       |
    /// | 1            | 1 + n         | 17      | 33      |
    /// | 2            | 1 + n + C(n,2)| 137     | 529     |
    /// | 3            | + C(n,3)      | 697     | 5,489   |
    ///
    /// # Arguments
    /// * `bucket_id` - Source bucket ID
    /// * `max_distance` - Maximum Hamming distance (0-3 supported)
    ///
    /// # Returns
    /// List of (neighbor_bucket_id, hamming_distance) tuples
    ///
    /// # Example
    /// ```python
    /// ball = index.hamming_ball(bucket_id, 2)
    /// # Returns: bucket (dist 0) + 16 at dist 1 + 120 at dist 2 = 137 total
    /// for neighbor_id, dist in ball:
    ///     print(f"Bucket {neighbor_id} at distance {dist}")
    /// ```
    fn hamming_ball(&self, bucket_id: u64, max_distance: usize) -> PyResult<Vec<(u64, u32)>> {
        Ok(self.index.lsh_hasher().hamming_ball(bucket_id, max_distance))
    }

    /// Multi-probe query: get bucket contents from bucket + Hamming neighbors
    ///
    /// This is the key multi-probe LSH operation. It queries the exact bucket
    /// and all buckets within the specified Hamming distance, returning all
    /// matching file references.
    ///
    /// # Arguments
    /// * `embedding` - Query embedding vector
    /// * `max_distance` - Maximum Hamming distance to probe (default: 1)
    ///
    /// # Returns
    /// List of (FileRef, hamming_distance) tuples, where distance indicates
    /// which bucket the result came from (0 = exact match, 1 = 1-bit neighbor, etc.)
    ///
    /// # Example
    /// ```python
    /// results = index.query_multiprobe(embedding, max_distance=2)
    /// for file_ref, dist in results:
    ///     print(f"{file_ref} at Hamming distance {dist}")
    /// ```
    #[pyo3(signature = (embedding, max_distance=1))]
    fn query_multiprobe(&self, embedding: Vec<f32>, max_distance: usize) -> PyResult<Vec<(PyFileRef, u32)>> {
        let bucket_id = self.index.lsh_hasher().hash(&embedding);
        let neighbors = self.index.lsh_hasher().hamming_ball(bucket_id, max_distance);

        let mut results = Vec::new();
        for (neighbor_bucket_id, distance) in neighbors {
            if let Some(refs) = self.index.get_bucket(neighbor_bucket_id) {
                for file_ref in refs {
                    results.push((PyFileRef::from(file_ref), distance));
                }
            }
        }

        Ok(results)
    }
}
