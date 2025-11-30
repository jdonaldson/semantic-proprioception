//! Protobuf serialization/deserialization for LSH index

use crate::index::LSHIndex;
use crate::file_ref::FileRef;
use prost::Message;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

// Include generated protobuf code
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/semantic_proprioception.rs"));
}

impl LSHIndex {
    /// Save index to protobuf file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let proto_index = self.to_proto();
        let mut file = File::create(path)?;
        let bytes = proto_index.encode_to_vec();
        file.write_all(&bytes)?;
        Ok(())
    }

    /// Load index from protobuf file
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        let proto_index = proto::KrapivinIndex::decode(&bytes[..])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Self::from_proto(proto_index)
    }

    /// Convert to protobuf message
    fn to_proto(&self) -> proto::KrapivinIndex {
        use std::time::{SystemTime, UNIX_EPOCH};

        // Extract configuration
        let lsh_seed = self.lsh_hasher.seed;
        let lsh_num_hyperplanes = self.lsh_hasher.num_bits as u32;
        let embedding_dim = self.lsh_hasher.embedding_dim as u32;

        // Convert hash table levels
        let mut levels = Vec::new();
        for (level_idx, level) in self.table.levels().iter().enumerate() {
            let mut buckets = Vec::new();
            let mut num_entries = 0u64;

            for (slot_idx, entry_opt) in level.iter().enumerate() {
                if let Some(entry) = entry_opt {
                    num_entries += 1;

                    // Convert file references
                    let refs: Vec<proto::FileRef> = entry.value.iter()
                        .map(|fr| proto::FileRef {
                            file_path: fr.file_path.clone(),
                            row_id: fr.row_id,
                            embedding_hash: vec![], // Optional, skip for now
                        })
                        .collect();

                    buckets.push(proto::Bucket {
                        bucket_id: entry.key,
                        fingerprint: entry.tag.fingerprint as u32,
                        level: entry.tag.layer as u32,
                        refs,
                        bucket_merkle_root: vec![], // Optional
                    });
                }
            }

            let load_factor = if level.len() > 0 {
                num_entries as f64 / level.len() as f64
            } else {
                0.0
            };

            levels.push(proto::Level {
                level_index: level_idx as u32,
                buckets,
                num_entries,
                load_factor,
            });
        }

        // Merkle roots
        let table_merkle_root = self.table.merkle_root()
            .map(|root| root.to_vec())
            .unwrap_or_default();

        let level_merkle_roots: Vec<Vec<u8>> = self.table.level_merkle_roots()
            .iter()
            .map(|tree| tree.root().to_vec())
            .collect();

        // Metadata
        let total_vectors = self.table.len() as u64;
        let timestamp_modified = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        proto::KrapivinIndex {
            lsh_seed,
            lsh_num_hyperplanes,
            embedding_dim,
            alpha: self.table.alpha() as u32,
            delta: self.table.delta(),
            beta: self.table.beta() as u32,
            levels,
            indexed_files: self.indexed_files.clone(),
            table_merkle_root,
            level_merkle_roots,
            total_vectors,
            timestamp_created: timestamp_modified, // Set on first save
            timestamp_modified,
            version: "1.0.0".to_string(),
        }
    }

    /// Construct from protobuf message
    fn from_proto(proto: proto::KrapivinIndex) -> std::io::Result<Self> {
        use crate::lsh::LSHHasher;
        use krapivin_core::KrapivinHashTable;

        // Reconstruct LSH hasher
        let lsh_hasher = LSHHasher::new(
            proto.lsh_seed,
            proto.lsh_num_hyperplanes as usize,
            proto.embedding_dim as usize,
        );

        // Reconstruct hash table
        let capacity: usize = proto.levels.iter()
            .map(|level| level.buckets.len())
            .sum();

        // Ensure minimum capacity of 1
        let capacity = capacity.max(1);

        let mut table = KrapivinHashTable::new(
            capacity,
            proto.delta,
            proto.beta as usize,
        );

        // Restore entries
        for level_proto in proto.levels {
            for bucket_proto in level_proto.buckets {
                let bucket_id = bucket_proto.bucket_id;
                let refs: Vec<FileRef> = bucket_proto.refs.iter()
                    .map(|fr_proto| FileRef {
                        file_path: fr_proto.file_path.clone(),
                        row_id: fr_proto.row_id,
                    })
                    .collect();

                table.insert(bucket_id, refs, bucket_id);
            }
        }

        Ok(LSHIndex {
            table,
            lsh_hasher,
            indexed_files: proto.indexed_files,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_round_trip_serialization() {
        let mut index = LSHIndex::new(12345, 16, 128, 1000, 0.3);

        // Add some test data
        let embedding = vec![0.1f32; 128];
        let file_ref = FileRef {
            file_path: "test.parquet".to_string(),
            row_id: 42,
        };
        index.add_embedding(&embedding, file_ref.clone());
        index.track_file("test.parquet".to_string());

        // Save to temp file
        let temp_file = NamedTempFile::new().unwrap();
        index.save(temp_file.path()).unwrap();

        // Load back
        let loaded = LSHIndex::load(temp_file.path()).unwrap();

        // Verify data integrity
        assert_eq!(loaded.indexed_files, index.indexed_files);
        assert_eq!(loaded.lsh_hasher.seed, index.lsh_hasher.seed);
        assert_eq!(loaded.lsh_hasher.num_bits, index.lsh_hasher.num_bits);
        assert_eq!(loaded.table.len(), index.table.len());

        // Verify query works
        let results = loaded.query(&embedding);
        assert!(results.is_some());
        let refs = results.unwrap();
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].file_path, "test.parquet");
        assert_eq!(refs[0].row_id, 42);
    }

    #[test]
    fn test_empty_index_serialization() {
        let index = LSHIndex::new(12345, 16, 128, 1000, 0.3);

        let temp_file = NamedTempFile::new().unwrap();
        index.save(temp_file.path()).unwrap();

        let loaded = LSHIndex::load(temp_file.path()).unwrap();
        assert_eq!(loaded.indexed_files.len(), 0);
        assert_eq!(loaded.table.len(), 0);
    }

    #[test]
    fn test_multiple_files_serialization() {
        let mut index = LSHIndex::new(12345, 16, 128, 1000, 0.3);

        // Add embeddings from multiple files
        for file_idx in 0..5 {
            let file_path = format!("file_{}.parquet", file_idx);
            index.track_file(file_path.clone());

            for row_idx in 0..10 {
                let embedding = vec![file_idx as f32 * 0.1; 128];
                let file_ref = FileRef {
                    file_path: file_path.clone(),
                    row_id: row_idx,
                };
                index.add_embedding(&embedding, file_ref);
            }
        }

        let temp_file = NamedTempFile::new().unwrap();
        index.save(temp_file.path()).unwrap();

        let loaded = LSHIndex::load(temp_file.path()).unwrap();
        assert_eq!(loaded.indexed_files.len(), 5);
        assert_eq!(loaded.table.len(), index.table.len());
    }
}
