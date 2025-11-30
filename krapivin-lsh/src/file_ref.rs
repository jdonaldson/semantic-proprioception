//! File reference for LSH index
//!
//! Stores pointer to embedding in Parquet file

use std::fmt;

/// Reference to an embedding vector in a Parquet file
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FileRef {
    /// Path to Parquet file
    pub file_path: String,

    /// Row index in Parquet file
    pub row_id: u64,
}

impl FileRef {
    /// Create new file reference
    pub fn new(file_path: String, row_id: u64) -> Self {
        FileRef { file_path, row_id }
    }
}

impl fmt::Display for FileRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.file_path, self.row_id)
    }
}

/// Serialize FileRef for Merkle hashing
impl AsRef<[u8]> for FileRef {
    fn as_ref(&self) -> &[u8] {
        // Simple serialization: just use file_path bytes
        // In production, might want more sophisticated approach
        self.file_path.as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_ref_creation() {
        let file_ref = FileRef::new("data.parquet".to_string(), 42);
        assert_eq!(file_ref.file_path, "data.parquet");
        assert_eq!(file_ref.row_id, 42);
    }

    #[test]
    fn test_file_ref_display() {
        let file_ref = FileRef::new("test.parquet".to_string(), 123);
        assert_eq!(format!("{}", file_ref), "test.parquet:123");
    }
}
