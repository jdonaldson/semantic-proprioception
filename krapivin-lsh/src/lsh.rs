//! LSH hash function implementation
//!
//! Uses random hyperplanes for locality-sensitive hashing
//! Fixed seed ensures consistent bucket IDs across all files

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// LSH hasher with fixed hyperplanes
pub struct LSHHasher {
    /// Random hyperplanes (num_bits × embedding_dim)
    pub(crate) hyperplanes: Vec<Vec<f32>>,

    /// Number of hash bits (bucket ID size)
    pub(crate) num_bits: usize,

    /// Embedding dimensionality
    pub(crate) embedding_dim: usize,

    /// Seed used for generation
    pub(crate) seed: u64,
}

impl LSHHasher {
    /// Create new LSH hasher with fixed seed
    ///
    /// # Arguments
    /// * `seed` - Fixed seed for deterministic hyperplanes
    /// * `num_bits` - Number of hash bits (typically 16-64)
    /// * `embedding_dim` - Dimensionality of embedding vectors
    pub fn new(seed: u64, num_bits: usize, embedding_dim: usize) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Generate random hyperplanes
        let mut hyperplanes = Vec::with_capacity(num_bits);
        for _ in 0..num_bits {
            let hyperplane: Vec<f32> = (0..embedding_dim)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            hyperplanes.push(hyperplane);
        }

        LSHHasher {
            hyperplanes,
            num_bits,
            embedding_dim,
            seed,
        }
    }

    /// Hash an embedding vector to a bucket ID
    ///
    /// Returns u64 bucket ID where each bit represents sign of dot product
    /// with one hyperplane
    pub fn hash(&self, embedding: &[f32]) -> u64 {
        assert_eq!(
            embedding.len(),
            self.embedding_dim,
            "Embedding dimension mismatch"
        );

        let mut bucket_id = 0u64;

        for (bit_idx, hyperplane) in self.hyperplanes.iter().enumerate() {
            // Compute dot product
            let dot_product: f32 = embedding
                .iter()
                .zip(hyperplane.iter())
                .map(|(a, b)| a * b)
                .sum();

            // Set bit if dot product is positive
            if dot_product > 0.0 {
                bucket_id |= 1 << bit_idx;
            }
        }

        bucket_id
    }

    /// Batch hash multiple embeddings
    pub fn hash_batch(&self, embeddings: &[&[f32]]) -> Vec<u64> {
        embeddings.iter().map(|emb| self.hash(emb)).collect()
    }

    /// Get seed
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Get number of bits
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    // =========================================================================
    // Multi-Resolution LSH Support
    // =========================================================================

    /// Truncate a hash to a coarser resolution
    ///
    /// Multi-resolution LSH: store at full resolution, query at any coarser level.
    /// This is an O(1) bit-shift operation.
    ///
    /// # Arguments
    /// * `hash` - Full resolution hash (up to num_bits)
    /// * `target_bits` - Desired resolution (must be <= num_bits)
    ///
    /// # Returns
    /// Hash truncated to target_bits (keeping the most significant bits)
    ///
    /// # Example
    /// ```
    /// // 32-bit hash: 0b11001010_10110011_01010101_11110000
    /// // Truncated to 16-bit: 0b11001010_10110011
    /// let hash_32 = hasher.hash(&embedding);
    /// let hash_16 = hasher.truncate_hash(hash_32, 16);
    /// ```
    pub fn truncate_hash(&self, hash: u64, target_bits: usize) -> u64 {
        if target_bits >= self.num_bits {
            return hash;
        }
        // Keep the lower target_bits (LSH stores in lower bits)
        hash & ((1u64 << target_bits) - 1)
    }

    /// Hash an embedding at a specific resolution
    ///
    /// More efficient than hash() + truncate_hash() when you only need
    /// the coarse resolution, as it skips computing unnecessary hyperplane
    /// dot products.
    ///
    /// # Arguments
    /// * `embedding` - Embedding vector
    /// * `resolution_bits` - Number of bits to compute (must be <= num_bits)
    ///
    /// # Returns
    /// Hash computed at the specified resolution
    pub fn hash_at_resolution(&self, embedding: &[f32], resolution_bits: usize) -> u64 {
        assert_eq!(
            embedding.len(),
            self.embedding_dim,
            "Embedding dimension mismatch"
        );

        let bits_to_compute = resolution_bits.min(self.num_bits);
        let mut bucket_id = 0u64;

        for (bit_idx, hyperplane) in self.hyperplanes.iter().take(bits_to_compute).enumerate() {
            let dot_product: f32 = embedding
                .iter()
                .zip(hyperplane.iter())
                .map(|(a, b)| a * b)
                .sum();

            if dot_product > 0.0 {
                bucket_id |= 1 << bit_idx;
            }
        }

        bucket_id
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
    /// Optimal number of bits (clamped to [8, num_bits])
    ///
    /// # Example
    /// ```
    /// // 8.8M docs with target 10 docs/bucket → 20 bits
    /// let optimal = hasher.optimal_query_bits(8_800_000, 10);
    /// assert_eq!(optimal, 20);
    /// ```
    pub fn optimal_query_bits(&self, corpus_size: usize, target_density: usize) -> usize {
        if corpus_size == 0 || target_density == 0 {
            return 8;
        }

        // bits = log2(corpus_size / target_density)
        let ratio = corpus_size as f64 / target_density as f64;
        let bits = (ratio.log2().ceil() as usize).max(8);

        bits.min(self.num_bits)
    }

    /// Get multi-resolution density profile for an embedding
    ///
    /// Returns bucket IDs at multiple resolutions, enabling O(1) density
    /// queries at any scale.
    ///
    /// # Arguments
    /// * `embedding` - Embedding vector
    /// * `resolutions` - List of bit resolutions to compute
    ///
    /// # Returns
    /// Vec of (resolution_bits, bucket_id) pairs
    pub fn multi_resolution_hash(&self, embedding: &[f32], resolutions: &[usize]) -> Vec<(usize, u64)> {
        // Compute full hash once
        let full_hash = self.hash(embedding);

        // Truncate to each resolution
        resolutions
            .iter()
            .map(|&bits| {
                let truncated = self.truncate_hash(full_hash, bits);
                (bits, truncated)
            })
            .collect()
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
    /// Number of differing bits (0 to num_bits)
    ///
    /// # Example
    /// ```
    /// let dist = hasher.hamming_distance(0b1010, 0b1001); // 2 bits differ
    /// assert_eq!(dist, 2);
    /// ```
    pub fn hamming_distance(&self, a: u64, b: u64) -> u32 {
        (a ^ b).count_ones()
    }

    /// Flip a specific bit in a hash
    ///
    /// # Arguments
    /// * `hash` - Original hash
    /// * `bit_index` - Bit position to flip (0 = LSB)
    ///
    /// # Returns
    /// Hash with the specified bit flipped
    pub fn flip_bit(&self, hash: u64, bit_index: usize) -> u64 {
        if bit_index >= self.num_bits {
            return hash;
        }
        hash ^ (1u64 << bit_index)
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
    /// * `distance` - Exact Hamming distance (1, 2, or 3 supported)
    ///
    /// # Returns
    /// Vec of neighbor bucket IDs at exactly the specified Hamming distance
    ///
    /// # Example
    /// ```
    /// // For 16-bit hash
    /// let d1 = hasher.hamming_neighbors(bucket_id, 1);  // 16 neighbors
    /// let d2 = hasher.hamming_neighbors(bucket_id, 2);  // 120 neighbors
    /// let d3 = hasher.hamming_neighbors(bucket_id, 3);  // 560 neighbors
    /// ```
    pub fn hamming_neighbors(&self, bucket_id: u64, distance: usize) -> Vec<u64> {
        match distance {
            0 => vec![bucket_id],
            1 => {
                // O(n): flip each bit once
                (0..self.num_bits)
                    .map(|bit| bucket_id ^ (1u64 << bit))
                    .collect()
            }
            2 => {
                // O(n²): all pairs of bit flips
                let mut neighbors = Vec::with_capacity(self.num_bits * (self.num_bits - 1) / 2);
                for bit1 in 0..self.num_bits {
                    for bit2 in (bit1 + 1)..self.num_bits {
                        neighbors.push(bucket_id ^ (1u64 << bit1) ^ (1u64 << bit2));
                    }
                }
                neighbors
            }
            3 => {
                // O(n³): all triples of bit flips
                let n = self.num_bits;
                let capacity = n * (n - 1) * (n - 2) / 6;
                let mut neighbors = Vec::with_capacity(capacity);
                for bit1 in 0..n {
                    for bit2 in (bit1 + 1)..n {
                        for bit3 in (bit2 + 1)..n {
                            neighbors.push(
                                bucket_id ^ (1u64 << bit1) ^ (1u64 << bit2) ^ (1u64 << bit3)
                            );
                        }
                    }
                }
                neighbors
            }
            _ => {
                // Distance 4+ not supported - returns empty
                // Use hamming_ball for cumulative distances
                Vec::new()
            }
        }
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
    /// Vec of (neighbor_bucket_id, hamming_distance) tuples
    pub fn hamming_ball(&self, bucket_id: u64, max_distance: usize) -> Vec<(u64, u32)> {
        let mut result = vec![(bucket_id, 0)]; // Include self at distance 0

        for d in 1..=max_distance.min(3) {
            for neighbor in self.hamming_neighbors(bucket_id, d) {
                result.push((neighbor, d as u32));
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsh_deterministic() {
        let hasher1 = LSHHasher::new(12345, 16, 128);
        let hasher2 = LSHHasher::new(12345, 16, 128);

        let embedding = vec![0.5f32; 128];

        let hash1 = hasher1.hash(&embedding);
        let hash2 = hasher2.hash(&embedding);

        assert_eq!(hash1, hash2, "Same seed should produce same hash");
    }

    #[test]
    fn test_lsh_different_seeds() {
        let hasher1 = LSHHasher::new(12345, 16, 128);
        let hasher2 = LSHHasher::new(54321, 16, 128);

        let embedding = vec![0.5f32; 128];

        let hash1 = hasher1.hash(&embedding);
        let hash2 = hasher2.hash(&embedding);

        // Different seeds should (usually) produce different hashes
        // Note: There's a tiny chance they're equal by random chance
        assert_ne!(hash1, hash2, "Different seeds should produce different hashes");
    }

    #[test]
    fn test_similar_vectors_same_bucket() {
        let hasher = LSHHasher::new(12345, 16, 128);

        let embedding1 = vec![1.0f32; 128];
        let mut embedding2 = vec![1.0f32; 128];
        embedding2[0] = 1.01; // Tiny perturbation

        let hash1 = hasher.hash(&embedding1);
        let hash2 = hasher.hash(&embedding2);

        // Similar vectors should have similar (often identical) hashes
        // Count different bits
        let diff_bits = (hash1 ^ hash2).count_ones();
        assert!(diff_bits <= 3, "Similar vectors should have few bit differences");
    }

    #[test]
    fn test_batch_hash() {
        let hasher = LSHHasher::new(12345, 16, 128);

        let emb1 = vec![0.5f32; 128];
        let emb2 = vec![0.6f32; 128];
        let emb3 = vec![0.7f32; 128];

        let hashes = hasher.hash_batch(&[&emb1, &emb2, &emb3]);

        assert_eq!(hashes.len(), 3);
        assert_eq!(hashes[0], hasher.hash(&emb1));
        assert_eq!(hashes[1], hasher.hash(&emb2));
        assert_eq!(hashes[2], hasher.hash(&emb3));
    }
}
