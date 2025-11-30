//! Tag metadata for composability
//!
//! Tags enable:
//! - Fast collision detection (fingerprint)
//! - Density encoding (layer)
//! - Composable merge operations

use std::fmt;

/// Tag metadata attached to each entry
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tag {
    /// 8-bit hash fingerprint for fast equality checks
    /// Rejects ~99% of non-matches without full key comparison
    pub fingerprint: u8,

    /// Which hierarchical level the entry is stored at
    /// Higher layer = denser bucket (more collisions)
    pub layer: u8,
}

impl Tag {
    /// Create a new tag
    pub fn new(fingerprint: u8, layer: u8) -> Self {
        Tag { fingerprint, layer }
    }

    /// Extract fingerprint from hash value
    #[inline(always)]
    pub fn from_hash(hash: u64, layer: u8) -> Self {
        Tag {
            fingerprint: (hash & 0xFF) as u8,
            layer,
        }
    }
}

impl fmt::Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tag(fp={}, layer={})", self.fingerprint, self.layer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tag_creation() {
        let tag = Tag::new(42, 3);
        assert_eq!(tag.fingerprint, 42);
        assert_eq!(tag.layer, 3);
    }

    #[test]
    fn test_from_hash() {
        let hash = 0x123456789ABCDEF0u64;
        let tag = Tag::from_hash(hash, 5);
        assert_eq!(tag.fingerprint, 0xF0);  // Last byte
        assert_eq!(tag.layer, 5);
    }
}
