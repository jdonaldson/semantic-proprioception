//! LSH (Locality-Sensitive Hashing) integration for Krapivin hash tables
//!
//! Provides:
//! - LSH hash function with fixed seed
//! - File reference storage
//! - Parquet integration
//! - Index serialization (protobuf)

pub mod lsh;
pub mod file_ref;
pub mod index;
pub mod serialization;
pub mod parquet;

pub use lsh::LSHHasher;
pub use file_ref::FileRef;
pub use index::LSHIndex;
pub use parquet::{index_parquet_file, index_parquet_files, ParquetError};

// Re-export protobuf types
pub use serialization::proto;
