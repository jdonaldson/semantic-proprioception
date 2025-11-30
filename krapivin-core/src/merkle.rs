//! Merkle tree integration for verifiable hash tables
//!
//! Provides:
//! - Per-level Merkle roots
//! - O(1) deduplication (compare roots)
//! - Integrity verification

use blake3::Hasher;

/// Merkle root (32-byte BLAKE3 hash)
pub type MerkleRoot = [u8; 32];

/// Merkle tree for a collection of entries
#[derive(Clone, Debug)]
pub struct MerkleTree {
    /// Root hash
    root: MerkleRoot,

    /// Number of entries hashed
    count: usize,
}

impl MerkleTree {
    /// Create empty Merkle tree
    pub fn empty() -> Self {
        MerkleTree {
            root: [0u8; 32],
            count: 0,
        }
    }

    /// Compute Merkle root from byte slices (order-independent XOR-based)
    pub fn from_hashes<I>(hashes: I) -> Self
    where
        I: IntoIterator<Item = MerkleRoot>,
    {
        let mut root = [0u8; 32];
        let mut count = 0;

        for hash in hashes {
            // XOR is commutative - order doesn't matter
            for (i, byte) in hash.iter().enumerate() {
                root[i] ^= byte;
            }
            count += 1;
        }

        MerkleTree { root, count }
    }

    /// Hash a single entry
    pub fn hash_entry<T: AsRef<[u8]>>(data: T) -> MerkleRoot {
        let mut hasher = Hasher::new();
        hasher.update(data.as_ref());
        *hasher.finalize().as_bytes()
    }

    /// Get root hash
    pub fn root(&self) -> &MerkleRoot {
        &self.root
    }

    /// Get entry count
    pub fn count(&self) -> usize {
        self.count
    }

    /// Check if two trees are identical (O(1))
    pub fn is_identical(&self, other: &Self) -> bool {
        self.root == other.root && self.count == other.count
    }
}

impl Default for MerkleTree {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tree() {
        let tree = MerkleTree::empty();
        assert_eq!(tree.count(), 0);
        assert_eq!(tree.root(), &[0u8; 32]);
    }

    #[test]
    fn test_single_entry() {
        let hash1 = MerkleTree::hash_entry(b"test");
        let tree = MerkleTree::from_hashes(vec![hash1]);
        assert_eq!(tree.count(), 1);
        assert_eq!(tree.root(), &hash1);
    }

    #[test]
    fn test_xor_commutative() {
        let hash1 = MerkleTree::hash_entry(b"entry1");
        let hash2 = MerkleTree::hash_entry(b"entry2");

        let tree1 = MerkleTree::from_hashes(vec![hash1, hash2]);
        let tree2 = MerkleTree::from_hashes(vec![hash2, hash1]);  // Reversed order

        assert_eq!(tree1.root(), tree2.root(), "XOR should be order-independent");
    }

    #[test]
    fn test_is_identical() {
        let hash1 = MerkleTree::hash_entry(b"data");
        let tree1 = MerkleTree::from_hashes(vec![hash1]);
        let tree2 = MerkleTree::from_hashes(vec![hash1]);

        assert!(tree1.is_identical(&tree2));
    }
}
