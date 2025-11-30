//! LSH index implementation using Krapivin hash table

use crate::file_ref::FileRef;
use crate::lsh::LSHHasher;
use krapivin_core::KrapivinHashTable;

/// LSH index storing bucket → file references mapping
pub struct LSHIndex {
    /// Krapivin hash table: bucket_id → Vec<FileRef>
    pub(crate) table: KrapivinHashTable<u64, Vec<FileRef>>,

    /// LSH hasher
    pub(crate) lsh_hasher: LSHHasher,

    /// List of indexed files
    pub(crate) indexed_files: Vec<String>,
}

impl LSHIndex {
    /// Create new LSH index
    ///
    /// # Arguments
    /// * `lsh_seed` - Fixed seed for LSH hyperplanes
    /// * `num_bits` - Number of LSH hash bits
    /// * `embedding_dim` - Embedding dimensionality
    /// * `capacity` - Hash table capacity
    /// * `delta` - Empty fraction parameter (0 < δ < 1)
    pub fn new(
        lsh_seed: u64,
        num_bits: usize,
        embedding_dim: usize,
        capacity: usize,
        delta: f64,
    ) -> Self {
        LSHIndex {
            table: KrapivinHashTable::new(capacity, delta, 20),
            lsh_hasher: LSHHasher::new(lsh_seed, num_bits, embedding_dim),
            indexed_files: Vec::new(),
        }
    }

    /// Add embedding to index
    ///
    /// # Arguments
    /// * `embedding` - Embedding vector
    /// * `file_ref` - Reference to file and row
    pub fn add_embedding(&mut self, embedding: &[f32], file_ref: FileRef) {
        let bucket_id = self.lsh_hasher.hash(embedding);

        // Get existing file refs or create new vec
        if let Some(refs) = self.table.get_mut(&bucket_id, bucket_id) {
            let old_count = refs.len();
            refs.push(file_ref);
            let new_count = refs.len();
            // Update O(1) density tracking
            self.table.update_bucket_count(bucket_id, old_count, new_count);
        } else {
            self.table.insert(bucket_id, vec![file_ref], bucket_id);
            // New bucket with 1 item
            self.table.update_bucket_count(bucket_id, 0, 1);
        }
    }

    /// Query for similar embeddings
    ///
    /// Returns file references in the same LSH bucket
    pub fn query(&self, embedding: &[f32]) -> Option<Vec<FileRef>> {
        let bucket_id = self.lsh_hasher.hash(embedding);
        self.table.get(&bucket_id, bucket_id).cloned()
    }

    /// Add a file to indexed list
    pub fn track_file(&mut self, file_path: String) {
        if !self.indexed_files.contains(&file_path) {
            self.indexed_files.push(file_path);
        }
    }

    /// Remove all entries from a specific file
    pub fn remove_file(&mut self, file_path: &str) {
        // Iterate through all buckets and remove matching file refs
        // Note: This is O(n) - could be optimized with reverse index
        for (_, refs) in self.table.iter() {
            // Can't modify during iteration - would need different approach
            // This is a placeholder for the actual implementation
        }

        self.indexed_files.retain(|f| f != file_path);
    }

    /// Get indexed files
    pub fn indexed_files(&self) -> &[String] {
        &self.indexed_files
    }

    /// Get table statistics
    pub fn stats(&self) -> IndexStats {
        IndexStats {
            num_buckets: self.table.len(),
            num_files: self.indexed_files.len(),
            load_factor: self.table.load_factor(),
            level_densities: self.table.density_by_level(),
        }
    }

    /// Get LSH hasher (for external use)
    pub fn lsh_hasher(&self) -> &LSHHasher {
        &self.lsh_hasher
    }

    /// Iterate over all buckets
    ///
    /// # Returns
    /// Iterator of (bucket_id, file_refs)
    pub fn iter_buckets(&self) -> impl Iterator<Item = (u64, &Vec<FileRef>)> + '_ {
        self.table.iter().map(|(bucket_id, refs)| (*bucket_id, refs))
    }

    /// Get contents of a specific bucket
    ///
    /// # Arguments
    /// * `bucket_id` - LSH bucket ID
    ///
    /// # Returns
    /// File references in this bucket, or None if bucket is empty
    pub fn get_bucket(&self, bucket_id: u64) -> Option<&Vec<FileRef>> {
        self.table.get(&bucket_id, bucket_id)
    }

    /// Get dense buckets above a threshold
    ///
    /// O(k) where k = number of distinct counts >= min_count (effectively O(1) for practical use)
    ///
    /// # Arguments
    /// * `min_count` - Minimum number of items in bucket
    ///
    /// # Returns
    /// Vector of (bucket_id, count) for buckets with >= min_count items
    pub fn dense_buckets(&self, min_count: usize) -> Vec<(u64, usize)> {
        // Use O(1) density query from KrapivinHashTable
        self.table.dense_buckets(min_count)
    }

    /// Get bucket size distribution
    ///
    /// # Returns
    /// Map of bucket_size -> count_of_buckets_with_that_size
    pub fn bucket_size_histogram(&self) -> std::collections::HashMap<usize, usize> {
        use std::collections::HashMap;

        let mut histogram = HashMap::new();
        for (_, refs) in self.iter_buckets() {
            *histogram.entry(refs.len()).or_insert(0) += 1;
        }
        histogram
    }

    // === Semantic Proprioception Features ===

    /// Get singleton buckets (exactly 1 item) - potential outliers/anomalies
    /// These represent semantically isolated points in the embedding space
    pub fn singleton_buckets(&self) -> Vec<u64> {
        self.table.singleton_buckets()
    }

    /// Get buckets within a count range [min, max]
    /// Useful for finding "medium density" buckets
    pub fn buckets_in_range(&self, min_count: usize, max_count: usize) -> Vec<(u64, usize)> {
        self.table.buckets_in_range(min_count, max_count)
    }

    /// Compute Hamming distance between two bucket IDs
    /// Lower distance = more semantically similar (in LSH space)
    #[inline]
    pub fn hamming_distance(bucket_a: u64, bucket_b: u64) -> u32 {
        (bucket_a ^ bucket_b).count_ones()
    }

    /// Estimate semantic distance between two buckets
    /// Returns a value between 0.0 (identical) and 1.0 (maximally different)
    /// Based on normalized Hamming distance of LSH hashes
    pub fn semantic_distance(&self, bucket_a: u64, bucket_b: u64) -> f64 {
        let num_bits = self.lsh_hasher.num_bits();
        let hamming = Self::hamming_distance(bucket_a, bucket_b);
        hamming as f64 / num_bits as f64
    }

    /// Get Hamming neighbors of a bucket (buckets differing by exactly 1 bit)
    /// These represent the closest semantic neighbors in LSH space
    pub fn hamming_neighbors(&self, bucket_id: u64) -> Vec<u64> {
        let num_bits = self.lsh_hasher.num_bits();
        (0..num_bits)
            .map(|bit| bucket_id ^ (1u64 << bit))
            .collect()
    }

    /// Find populated Hamming neighbors of a bucket
    /// Returns (neighbor_bucket_id, count) for neighbors that have items
    pub fn populated_neighbors(&self, bucket_id: u64) -> Vec<(u64, usize)> {
        self.hamming_neighbors(bucket_id)
            .into_iter()
            .filter_map(|neighbor_id| {
                let count = self.table.bucket_count(&neighbor_id);
                if count > 0 {
                    Some((neighbor_id, count))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Compute density gradient for a bucket
    /// Returns (bucket_id, gradient) where gradient > 0 means neighbors are denser,
    /// gradient < 0 means this bucket is denser than its neighbors
    /// Helps identify semantic attractors vs. isolated clusters
    pub fn density_gradient(&self, bucket_id: u64) -> f64 {
        let self_count = self.table.bucket_count(&bucket_id) as f64;
        let neighbors = self.populated_neighbors(bucket_id);

        if neighbors.is_empty() {
            return 0.0; // No neighbors to compare
        }

        let avg_neighbor_count: f64 = neighbors.iter()
            .map(|(_, count)| *count as f64)
            .sum::<f64>() / neighbors.len() as f64;

        avg_neighbor_count - self_count
    }

    /// Find buckets that are semantic attractors (denser than their neighbors)
    /// These represent strong semantic themes
    pub fn semantic_attractors(&self, min_count: usize) -> Vec<(u64, usize, f64)> {
        self.dense_buckets(min_count)
            .into_iter()
            .map(|(bucket_id, count)| {
                let gradient = self.density_gradient(bucket_id);
                (bucket_id, count, gradient)
            })
            .filter(|(_, _, gradient)| *gradient < 0.0) // Denser than neighbors
            .collect()
    }

    /// Find semantic bridges - medium-density buckets connecting dense regions
    /// These often represent transitional or hybrid concepts
    pub fn semantic_bridges(&self, min_count: usize, max_count: usize) -> Vec<(u64, usize, usize)> {
        self.buckets_in_range(min_count, max_count)
            .into_iter()
            .map(|(bucket_id, count)| {
                let dense_neighbors = self.populated_neighbors(bucket_id)
                    .into_iter()
                    .filter(|(_, n_count)| *n_count > count)
                    .count();
                (bucket_id, count, dense_neighbors)
            })
            .filter(|(_, _, dense_neighbors)| *dense_neighbors >= 2) // Connected to multiple dense regions
            .collect()
    }
}

/// Index statistics
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub num_buckets: usize,
    pub num_files: usize,
    pub load_factor: f64,
    pub level_densities: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_creation() {
        let index = LSHIndex::new(12345, 16, 128, 1000, 0.3);
        assert_eq!(index.indexed_files().len(), 0);
    }

    #[test]
    fn test_add_and_query() {
        let mut index = LSHIndex::new(12345, 16, 128, 1000, 0.3);

        let embedding = vec![0.5f32; 128];
        let file_ref = FileRef::new("test.parquet".to_string(), 42);

        index.add_embedding(&embedding, file_ref.clone());

        let results = index.query(&embedding);
        assert!(results.is_some());

        let refs = results.unwrap();
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0], file_ref);
    }

    #[test]
    fn test_multiple_refs_same_bucket() {
        let mut index = LSHIndex::new(12345, 16, 128, 1000, 0.3);

        let embedding = vec![0.5f32; 128];
        let file_ref1 = FileRef::new("file1.parquet".to_string(), 10);
        let file_ref2 = FileRef::new("file2.parquet".to_string(), 20);

        index.add_embedding(&embedding, file_ref1.clone());
        index.add_embedding(&embedding, file_ref2.clone());

        let results = index.query(&embedding).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_track_file() {
        let mut index = LSHIndex::new(12345, 16, 128, 1000, 0.3);

        index.track_file("file1.parquet".to_string());
        index.track_file("file2.parquet".to_string());
        index.track_file("file1.parquet".to_string());  // Duplicate

        assert_eq!(index.indexed_files().len(), 2);
    }

    #[test]
    fn test_iter_buckets() {
        let mut index = LSHIndex::new(12345, 16, 128, 1000, 0.3);

        // Add some embeddings
        for i in 0..10 {
            let embedding = vec![i as f32 * 0.1; 128];
            let file_ref = FileRef::new(format!("file{}.parquet", i), i as u64);
            index.add_embedding(&embedding, file_ref);
        }

        // Iterate buckets
        let buckets: Vec<_> = index.iter_buckets().collect();
        assert!(buckets.len() > 0);

        // Check total items
        let total_items: usize = buckets.iter().map(|(_, refs)| refs.len()).sum();
        assert_eq!(total_items, 10);
    }

    #[test]
    fn test_get_bucket() {
        let mut index = LSHIndex::new(12345, 16, 128, 1000, 0.3);

        let embedding = vec![0.5f32; 128];
        let file_ref = FileRef::new("test.parquet".to_string(), 42);
        index.add_embedding(&embedding, file_ref.clone());

        // Get the bucket this embedding went to
        let bucket_id = index.lsh_hasher().hash(&embedding);
        let refs = index.get_bucket(bucket_id).unwrap();

        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].file_path, "test.parquet");
        assert_eq!(refs[0].row_id, 42);
    }

    #[test]
    fn test_dense_buckets() {
        let mut index = LSHIndex::new(12345, 16, 128, 1000, 0.3);

        // Create one dense bucket (many similar embeddings)
        let embedding = vec![0.5f32; 128];
        for i in 0..10 {
            let file_ref = FileRef::new("test.parquet".to_string(), i);
            index.add_embedding(&embedding, file_ref);
        }

        // Create sparse buckets (different embeddings)
        for i in 0..5 {
            let different_embedding = vec![i as f32; 128];
            let file_ref = FileRef::new("test.parquet".to_string(), 100 + i);
            index.add_embedding(&different_embedding, file_ref);
        }

        // Find dense buckets (at least 5 items)
        let dense = index.dense_buckets(5);

        // We should have at least one dense bucket
        assert!(dense.len() >= 1);

        // The densest bucket should have at least 10 items
        let max_count = dense.iter().map(|(_, count)| *count).max().unwrap_or(0);
        assert!(max_count >= 10);
    }

    #[test]
    fn test_bucket_size_histogram() {
        let mut index = LSHIndex::new(12345, 16, 128, 1000, 0.3);

        // Add 11 total items (different embeddings to ensure different buckets)
        for i in 0..11 {
            let embedding = vec![i as f32 * 10.0; 128];
            let file_ref = FileRef::new("test.parquet".to_string(), i as u64);
            index.add_embedding(&embedding, file_ref);
        }

        let histogram = index.bucket_size_histogram();

        // Verify histogram has entries
        assert!(!histogram.is_empty());

        // Verify total items across all buckets = 11
        let total_items: usize = histogram.iter()
            .map(|(size, count)| size * count)
            .sum();
        assert_eq!(total_items, 11);
    }

    #[test]
    fn test_hamming_distance() {
        // Same bucket = 0 distance
        assert_eq!(LSHIndex::hamming_distance(0b1010, 0b1010), 0);

        // One bit different = 1
        assert_eq!(LSHIndex::hamming_distance(0b1010, 0b1011), 1);

        // All bits different (4 bits)
        assert_eq!(LSHIndex::hamming_distance(0b0000, 0b1111), 4);
    }

    #[test]
    fn test_semantic_distance() {
        let index = LSHIndex::new(12345, 16, 128, 1000, 0.3);

        // Same bucket = 0.0 distance
        assert_eq!(index.semantic_distance(100, 100), 0.0);

        // Different buckets = some distance > 0
        let dist = index.semantic_distance(0b0000_0000_0000_0000, 0b1111_1111_1111_1111);
        assert_eq!(dist, 1.0); // All 16 bits different = max distance
    }

    #[test]
    fn test_hamming_neighbors() {
        let index = LSHIndex::new(12345, 8, 128, 1000, 0.3); // 8 bits

        let neighbors = index.hamming_neighbors(0b00000000);

        // Should have 8 neighbors (one for each bit flip)
        assert_eq!(neighbors.len(), 8);

        // Each neighbor should differ by exactly one bit
        for neighbor in &neighbors {
            assert_eq!(LSHIndex::hamming_distance(0b00000000, *neighbor), 1);
        }
    }

    #[test]
    fn test_singleton_buckets() {
        let mut index = LSHIndex::new(12345, 16, 128, 1000, 0.3);

        // Add multiple items to one bucket (same embedding)
        let dense_embedding = vec![0.5f32; 128];
        for i in 0..5 {
            let file_ref = FileRef::new("test.parquet".to_string(), i);
            index.add_embedding(&dense_embedding, file_ref);
        }

        // Add single items to other buckets (different embeddings)
        for i in 0..3 {
            let sparse_embedding = vec![i as f32 * 100.0; 128];
            let file_ref = FileRef::new("test.parquet".to_string(), 100 + i);
            index.add_embedding(&sparse_embedding, file_ref);
        }

        let singletons = index.singleton_buckets();

        // Should have some singleton buckets (exact count depends on LSH collisions)
        // The sparse embeddings should mostly create singletons
        assert!(!singletons.is_empty());
    }

    #[test]
    fn test_density_gradient() {
        let mut index = LSHIndex::new(12345, 8, 128, 1000, 0.3);

        // Create a dense bucket
        let embedding = vec![0.5f32; 128];
        for i in 0..10 {
            let file_ref = FileRef::new("test.parquet".to_string(), i);
            index.add_embedding(&embedding, file_ref);
        }

        let bucket_id = index.lsh_hasher().hash(&embedding);
        let gradient = index.density_gradient(bucket_id);

        // Gradient should be calculable (may be 0 if no populated neighbors)
        // This test just verifies the function runs without panic
        assert!(gradient.is_finite());
    }
}
