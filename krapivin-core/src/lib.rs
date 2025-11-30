//! Krapivin Hash Table - Optimal hash table with O(log² δ⁻¹) probe complexity
//!
//! Core library providing:
//! - Hierarchical levels for density encoding
//! - Tag metadata for composability
//! - Single hash call with arithmetic probing
//! - Merkle tree integration for verification

pub mod hashtable;
pub mod tag;
pub mod merkle;

pub use hashtable::KrapivinHashTable;
pub use tag::Tag;
pub use merkle::MerkleTree;

#[cfg(test)]
mod tests;
