//! Integration tests for krapivin-core

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn test_hash(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod integration_tests {
    use crate::KrapivinHashTable;
    use super::test_hash;

    #[test]
    fn test_basic_workflow() {
        let mut ht = KrapivinHashTable::new(1000, 0.3, 20);

        // Insert some entries
        for i in 0..100 {
            let key = format!("key{}", i);
            let value = format!("value{}", i);
            let hash = test_hash(&key);
            assert!(ht.insert(key, value, hash));
        }

        assert_eq!(ht.len(), 100);
        assert!(ht.load_factor() < 0.2);

        // Verify densities
        let densities = ht.density_by_level();
        assert!(!densities.is_empty());
    }

    #[test]
    fn test_o1_density_queries() {
        let mut ht: KrapivinHashTable<u64, Vec<String>> = KrapivinHashTable::new(1000, 0.3, 20);

        // Create buckets with different counts
        // Bucket 1: 10 items
        let bucket1: u64 = 1;
        ht.insert(bucket1, vec![], bucket1);
        for i in 0..10 {
            let old_count = ht.bucket_count(&bucket1);
            if let Some(items) = ht.get_mut(&bucket1, bucket1) {
                items.push(format!("item{}", i));
            }
            ht.update_bucket_count(bucket1, old_count, old_count + 1);
        }

        // Bucket 2: 5 items
        let bucket2: u64 = 2;
        ht.insert(bucket2, vec![], bucket2);
        for i in 0..5 {
            let old_count = ht.bucket_count(&bucket2);
            if let Some(items) = ht.get_mut(&bucket2, bucket2) {
                items.push(format!("item{}", i));
            }
            ht.update_bucket_count(bucket2, old_count, old_count + 1);
        }

        // Bucket 3: 3 items
        let bucket3: u64 = 3;
        ht.insert(bucket3, vec![], bucket3);
        for i in 0..3 {
            let old_count = ht.bucket_count(&bucket3);
            if let Some(items) = ht.get_mut(&bucket3, bucket3) {
                items.push(format!("item{}", i));
            }
            ht.update_bucket_count(bucket3, old_count, old_count + 1);
        }

        // Test O(1) density queries
        let dense_5 = ht.dense_buckets(5);
        assert_eq!(dense_5.len(), 2); // bucket1 (10) and bucket2 (5)

        let dense_8 = ht.dense_buckets(8);
        assert_eq!(dense_8.len(), 1); // only bucket1 (10)

        let dense_3 = ht.dense_buckets(3);
        assert_eq!(dense_3.len(), 3); // all buckets

        // Verify counts
        assert_eq!(ht.bucket_count(&bucket1), 10);
        assert_eq!(ht.bucket_count(&bucket2), 5);
        assert_eq!(ht.bucket_count(&bucket3), 3);

        // Test histogram
        let histogram = ht.count_histogram();
        assert!(!histogram.is_empty());
    }
}
