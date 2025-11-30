//! Generic Krapivin hash table implementation
//!
//! Provides:
//! - O(log² δ⁻¹) probe complexity
//! - Hierarchical levels encoding density
//! - Single hash call with arithmetic probing
//! - Tag metadata for composability
//! - O(1) density queries via count-indexed structure

use crate::tag::Tag;
use crate::merkle::{MerkleTree, MerkleRoot};
use std::collections::{HashMap, BTreeMap, HashSet};
use std::hash::Hash;
use std::fmt::Debug;

/// Entry in the hash table
#[derive(Clone, Debug)]
pub struct Entry<K, V> {
    pub key: K,
    pub value: V,
    pub tag: Tag,
}

/// Krapivin hash table with hierarchical levels
pub struct KrapivinHashTable<K, V> {
    /// Hierarchical levels (α levels total)
    pub(crate) levels: Vec<Vec<Option<Entry<K, V>>>>,

    /// Current number of entries
    pub(crate) size: usize,

    /// Total capacity across all levels
    pub(crate) capacity: usize,

    /// Empty fraction parameter (δ)
    pub(crate) delta: f64,

    /// Probing parameter (β)
    pub(crate) beta: usize,

    /// Number of levels (α)
    pub(crate) alpha: usize,

    /// Merkle tree per level (optional)
    pub(crate) level_merkle_trees: Vec<MerkleTree>,

    /// Overall table Merkle root
    pub(crate) table_merkle_root: Option<MerkleRoot>,

    // === O(1) Density Query Support ===

    /// Bucket counts: key → count of items in that bucket
    /// Updated on insert/delete
    pub(crate) bucket_counts: HashMap<K, usize>,

    /// Count-indexed buckets: count → set of keys with that count
    /// Enables O(1) density threshold queries
    pub(crate) count_to_buckets: BTreeMap<usize, HashSet<K>>,
}

impl<K, V> KrapivinHashTable<K, V>
where
    K: Clone + Eq + Hash + Debug,
    V: Clone + Debug,
{
    /// Create new Krapivin hash table
    ///
    /// # Arguments
    /// * `capacity` - Total capacity across all levels
    /// * `delta` - Empty fraction (0 < δ < 1), controls number of levels
    /// * `beta` - Probing parameter (block size)
    pub fn new(capacity: usize, delta: f64, beta: usize) -> Self {
        assert!(delta > 0.0 && delta < 1.0, "delta must be in (0, 1)");
        assert!(capacity > 0, "capacity must be positive");

        // Calculate number of levels: α = O(log δ⁻¹)
        let alpha = ((1.0 / delta).log2() as usize).max(1);

        // Create hierarchical levels with geometric sizing
        let mut levels = Vec::new();
        let mut remaining = capacity;

        for i in 0..alpha {
            let level_size = remaining.min(capacity / (1 << (alpha - i - 1)));
            if level_size > 0 {
                levels.push(vec![None; level_size]);
                remaining = remaining.saturating_sub(level_size);
            }
        }

        // Overflow level for any remaining capacity
        if remaining > 0 {
            levels.push(vec![None; remaining]);
        }

        let num_levels = levels.len();

        KrapivinHashTable {
            levels,
            size: 0,
            capacity,
            delta,
            beta,
            alpha,
            level_merkle_trees: vec![MerkleTree::empty(); num_levels],
            table_merkle_root: None,
            bucket_counts: HashMap::new(),
            count_to_buckets: BTreeMap::new(),
        }
    }

    // === O(1) Density Query Helpers ===

    /// Update bucket count tracking when a bucket's size changes
    /// Call this after modifying a bucket's contents
    ///
    /// # Arguments
    /// * `bucket_key` - The bucket's key
    /// * `old_count` - Previous item count (0 if new bucket)
    /// * `new_count` - New item count
    pub fn update_bucket_count(&mut self, bucket_key: K, old_count: usize, new_count: usize) {
        // Remove from old count set
        if old_count > 0 {
            if let Some(bucket_set) = self.count_to_buckets.get_mut(&old_count) {
                bucket_set.remove(&bucket_key);
                if bucket_set.is_empty() {
                    self.count_to_buckets.remove(&old_count);
                }
            }
        }

        // Add to new count set
        if new_count > 0 {
            self.bucket_counts.insert(bucket_key.clone(), new_count);
            self.count_to_buckets
                .entry(new_count)
                .or_insert_with(HashSet::new)
                .insert(bucket_key);
        } else {
            self.bucket_counts.remove(&bucket_key);
        }
    }

    /// Get all buckets with count >= min_count
    /// O(k) where k = number of distinct counts >= min_count (typically small)
    pub fn dense_buckets(&self, min_count: usize) -> Vec<(K, usize)> {
        self.count_to_buckets
            .range(min_count..)
            .flat_map(|(count, bucket_set)| {
                bucket_set.iter().map(move |key| (key.clone(), *count))
            })
            .collect()
    }

    /// Get count for a specific bucket
    pub fn bucket_count(&self, bucket_key: &K) -> usize {
        self.bucket_counts.get(bucket_key).copied().unwrap_or(0)
    }

    /// Get number of buckets at each count level
    /// Returns Vec of (count, num_buckets_with_that_count)
    pub fn count_histogram(&self) -> Vec<(usize, usize)> {
        self.count_to_buckets
            .iter()
            .map(|(count, buckets)| (*count, buckets.len()))
            .collect()
    }

    /// Get singleton buckets (exactly 1 item) - potential outliers/anomalies
    /// O(1) lookup
    pub fn singleton_buckets(&self) -> Vec<K> {
        self.count_to_buckets
            .get(&1)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get buckets within a count range [min, max]
    /// Useful for finding "medium density" buckets
    pub fn buckets_in_range(&self, min_count: usize, max_count: usize) -> Vec<(K, usize)> {
        self.count_to_buckets
            .range(min_count..=max_count)
            .flat_map(|(count, bucket_set)| {
                bucket_set.iter().map(move |key| (key.clone(), *count))
            })
            .collect()
    }

    /// Generate probe sequence from a hash value
    ///
    /// Single hash call + arithmetic offsets (no rehashing!)
    #[inline(always)]
    fn probe_sequence(&self, hash: u64) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.levels.iter().enumerate().flat_map(move |(level_idx, level)| {
            let level_size = level.len();
            (0..20).filter_map(move |j| {
                if level_size == 0 {
                    None
                } else {
                    // Quadratic probing with level offset
                    let offset = (level_idx as u64) * 1000 + (j * j);
                    let slot = ((hash.wrapping_add(offset)) % level_size as u64) as usize;
                    Some((level_idx, slot))
                }
            })
        })
    }

    /// Insert key-value pair
    ///
    /// Returns true if inserted, false if table is full
    pub fn insert(&mut self, key: K, value: V, hash: u64) -> bool {
        if self.size >= self.capacity {
            return false;
        }

        // Collect probe sequence to avoid borrow checker issues
        let probes: Vec<(usize, usize)> = self.probe_sequence(hash).collect();

        for (level_idx, slot) in probes {
            // Check if slot is empty or has same key (update case)
            let should_insert = match &self.levels[level_idx][slot] {
                None => true,
                Some(entry) => entry.key == key,
            };

            if should_insert {
                let was_empty = self.levels[level_idx][slot].is_none();

                self.levels[level_idx][slot] = Some(Entry {
                    key,
                    value,
                    tag: Tag::from_hash(hash, level_idx as u8),
                });

                if was_empty {
                    self.size += 1;
                }

                return true;
            }
        }

        false
    }

    /// Get value by key
    pub fn get(&self, key: &K, hash: u64) -> Option<&V> {
        for (level_idx, slot) in self.probe_sequence(hash) {
            match &self.levels[level_idx][slot] {
                Some(entry) => {
                    if &entry.key == key {
                        return Some(&entry.value);
                    }
                    // Continue probing (collision)
                }
                None => {
                    // Empty slot - key not present
                    return None;
                }
            }
        }

        None
    }

    /// Get mutable value by key
    pub fn get_mut(&mut self, key: &K, hash: u64) -> Option<&mut V> {
        // Collect probe sequence to avoid borrow checker issues
        let probes: Vec<(usize, usize)> = self.probe_sequence(hash).collect();

        // Find location without holding any mutable references
        let mut location = None;
        for (level_idx, slot) in probes {
            if let Some(entry) = &self.levels[level_idx][slot] {
                if &entry.key == key {
                    location = Some((level_idx, slot));
                    break;
                }
            } else {
                // Empty slot - key not present
                break;
            }
        }

        // Now do single mutable access
        location.and_then(|(level_idx, slot)| {
            self.levels[level_idx][slot]
                .as_mut()
                .map(|entry| &mut entry.value)
        })
    }

    /// Get value with tag metadata
    pub fn get_with_tag(&self, key: &K, hash: u64) -> Option<(&V, Tag)> {
        for (level_idx, slot) in self.probe_sequence(hash) {
            match &self.levels[level_idx][slot] {
                Some(entry) => {
                    if &entry.key == key {
                        return Some((&entry.value, entry.tag));
                    }
                }
                None => return None,
            }
        }

        None
    }

    /// Compute density for each level (O(n) scan)
    pub fn density_by_level(&self) -> Vec<f64> {
        self.levels
            .iter()
            .map(|level| {
                if level.is_empty() {
                    0.0
                } else {
                    let filled = level.iter().filter(|e| e.is_some()).count();
                    filled as f64 / level.len() as f64
                }
            })
            .collect()
    }

    /// Get current size
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get load factor
    pub fn load_factor(&self) -> f64 {
        self.size as f64 / self.capacity as f64
    }

    /// Get number of levels
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Iterate over all entries
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.levels.iter().flat_map(|level| {
            level.iter().filter_map(|entry_opt| {
                entry_opt.as_ref().map(|entry| (&entry.key, &entry.value))
            })
        })
    }

    /// Recompute Merkle trees (call after bulk operations)
    pub fn recompute_merkle_trees(&mut self)
    where
        K: AsRef<[u8]>,
        V: AsRef<[u8]>,
    {
        // Compute per-level Merkle roots
        for (level_idx, level) in self.levels.iter().enumerate() {
            let entry_hashes: Vec<MerkleRoot> = level
                .iter()
                .filter_map(|entry_opt| {
                    entry_opt.as_ref().map(|entry| {
                        // Hash key || value
                        let mut data = Vec::new();
                        data.extend_from_slice(entry.key.as_ref());
                        data.extend_from_slice(entry.value.as_ref());
                        MerkleTree::hash_entry(&data)
                    })
                })
                .collect();

            self.level_merkle_trees[level_idx] = MerkleTree::from_hashes(entry_hashes);
        }

        // Compute table root from level roots
        let level_roots: Vec<MerkleRoot> = self.level_merkle_trees
            .iter()
            .map(|tree| *tree.root())
            .collect();

        let table_tree = MerkleTree::from_hashes(level_roots);
        self.table_merkle_root = Some(*table_tree.root());
    }

    /// Get table Merkle root
    pub fn merkle_root(&self) -> Option<&MerkleRoot> {
        self.table_merkle_root.as_ref()
    }

    /// Get level Merkle roots
    pub fn level_merkle_roots(&self) -> &[MerkleTree] {
        &self.level_merkle_trees
    }

    /// Check if identical to another table (O(1) with Merkle trees)
    pub fn is_identical(&self, other: &Self) -> bool {
        match (self.merkle_root(), other.merkle_root()) {
            (Some(root1), Some(root2)) => root1 == root2,
            _ => false,
        }
    }

    /// Get reference to levels (for serialization)
    pub fn levels(&self) -> &[Vec<Option<Entry<K, V>>>] {
        &self.levels
    }

    /// Get alpha parameter
    pub fn alpha(&self) -> usize {
        self.alpha
    }

    /// Get delta parameter
    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// Get beta parameter
    pub fn beta(&self) -> usize {
        self.beta
    }
}

impl<K, V> Default for KrapivinHashTable<K, V>
where
    K: Clone + Eq + Hash + Debug,
    V: Clone + Debug,
{
    fn default() -> Self {
        Self::new(1024, 0.3, 20)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_hash(key: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn test_new() {
        let ht: KrapivinHashTable<String, String> = KrapivinHashTable::new(1024, 0.3, 20);
        assert_eq!(ht.len(), 0);
        assert_eq!(ht.capacity(), 1024);
        assert!(ht.num_levels() > 0);
    }

    #[test]
    fn test_insert_and_get() {
        let mut ht = KrapivinHashTable::new(1024, 0.3, 20);

        let key = "test_key".to_string();
        let value = "test_value".to_string();
        let hash = test_hash(&key);

        assert!(ht.insert(key.clone(), value.clone(), hash));
        assert_eq!(ht.len(), 1);

        let retrieved = ht.get(&key, hash);
        assert_eq!(retrieved, Some(&value));
    }

    #[test]
    fn test_update() {
        let mut ht = KrapivinHashTable::new(1024, 0.3, 20);

        let key = "key".to_string();
        let hash = test_hash(&key);

        ht.insert(key.clone(), "value1".to_string(), hash);
        assert_eq!(ht.len(), 1);

        ht.insert(key.clone(), "value2".to_string(), hash);
        assert_eq!(ht.len(), 1);  // Size shouldn't increase

        assert_eq!(ht.get(&key, hash), Some(&"value2".to_string()));
    }

    #[test]
    fn test_density() {
        let mut ht = KrapivinHashTable::new(100, 0.3, 20);

        for i in 0..50 {
            let key = format!("key{}", i);
            let hash = test_hash(&key);
            ht.insert(key, format!("value{}", i), hash);
        }

        let densities = ht.density_by_level();
        assert!(!densities.is_empty());
        assert!(densities.iter().any(|d| *d > 0.0));
    }

    #[test]
    fn test_iter() {
        let mut ht = KrapivinHashTable::new(100, 0.3, 20);

        let keys = vec!["a", "b", "c"];
        for key in &keys {
            let hash = test_hash(key);
            ht.insert(key.to_string(), format!("val_{}", key), hash);
        }

        let collected: Vec<_> = ht.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(collected.len(), 3);
        for key in keys {
            assert!(collected.contains(&key));
        }
    }
}
