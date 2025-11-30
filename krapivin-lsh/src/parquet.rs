//! Polars/Parquet integration for LSH indexing
//!
//! Provides functions to:
//! - Read embeddings from Parquet files
//! - Batch-insert into LSH index
//! - Handle multiple file formats (List<Float32>, Array<Float32>)

use crate::index::LSHIndex;
use crate::file_ref::FileRef;
use polars::prelude::*;
use std::path::Path;

/// Error type for Parquet operations
#[derive(Debug)]
pub enum ParquetError {
    PolarsError(PolarsError),
    ColumnNotFound(String),
    InvalidColumnType(String),
    InvalidEmbeddingDim { expected: usize, found: usize },
}

impl From<PolarsError> for ParquetError {
    fn from(err: PolarsError) -> Self {
        ParquetError::PolarsError(err)
    }
}

impl std::fmt::Display for ParquetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParquetError::PolarsError(e) => write!(f, "Polars error: {}", e),
            ParquetError::ColumnNotFound(col) => write!(f, "Column not found: {}", col),
            ParquetError::InvalidColumnType(msg) => write!(f, "Invalid column type: {}", msg),
            ParquetError::InvalidEmbeddingDim { expected, found } => {
                write!(f, "Invalid embedding dimension: expected {}, found {}", expected, found)
            }
        }
    }
}

impl std::error::Error for ParquetError {}

/// Index embeddings from a Parquet file
///
/// # Arguments
/// * `index` - LSH index to add embeddings to
/// * `file_path` - Path to Parquet file
/// * `embedding_column` - Name of column containing embeddings
///
/// # Returns
/// Number of embeddings indexed
pub fn index_parquet_file(
    index: &mut LSHIndex,
    file_path: impl AsRef<Path>,
    embedding_column: &str,
) -> Result<usize, ParquetError> {
    let file_path = file_path.as_ref();
    let file_path_str = file_path.to_string_lossy().to_string();

    // Read Parquet file
    let df = LazyFrame::scan_parquet(file_path, Default::default())?
        .collect()?;

    // Extract embedding column (Column in Polars 0.45+)
    let embedding_col = df
        .column(embedding_column)
        .map_err(|_| ParquetError::ColumnNotFound(embedding_column.to_string()))?;

    // Convert Column to Series
    let embedding_series = embedding_col.as_materialized_series().clone();

    // Track file
    index.track_file(file_path_str.clone());

    // Process embeddings based on column type
    let num_indexed = match embedding_series.dtype() {
        DataType::List(inner) => {
            // List<Float32> format
            if !matches!(**inner, DataType::Float32) {
                return Err(ParquetError::InvalidColumnType(format!(
                    "Expected List<Float32>, got List<{:?}>",
                    inner
                )));
            }
            index_list_embeddings(index, &embedding_series, &file_path_str)?
        }
        DataType::Array(inner, size) => {
            // Array<Float32, N> format (fixed size)
            if !matches!(**inner, DataType::Float32) {
                return Err(ParquetError::InvalidColumnType(format!(
                    "Expected Array<Float32, _>, got Array<{:?}, {}>",
                    inner, size
                )));
            }
            index_array_embeddings(index, &embedding_series, &file_path_str)?
        }
        dtype => {
            return Err(ParquetError::InvalidColumnType(format!(
                "Unsupported embedding column type: {:?}. Expected List<Float32> or Array<Float32, N>",
                dtype
            )));
        }
    };

    Ok(num_indexed)
}

/// Index embeddings from List<Float32> column
fn index_list_embeddings(
    index: &mut LSHIndex,
    series: &Series,
    file_path: &str,
) -> Result<usize, ParquetError> {
    let list_chunked = series.list().map_err(|e| ParquetError::PolarsError(e))?;
    let expected_dim = index.lsh_hasher().embedding_dim;
    let mut count = 0;

    for (row_id, opt_series) in list_chunked.into_iter().enumerate() {
        if let Some(embedding_series) = opt_series {
            // Extract float values
            let floats = embedding_series
                .f32()
                .map_err(|e| ParquetError::PolarsError(e))?;

            let embedding: Vec<f32> = floats
                .into_iter()
                .filter_map(|opt_val| opt_val)
                .collect();

            // Validate dimension
            if embedding.len() != expected_dim {
                return Err(ParquetError::InvalidEmbeddingDim {
                    expected: expected_dim,
                    found: embedding.len(),
                });
            }

            // Add to index
            let file_ref = FileRef {
                file_path: file_path.to_string(),
                row_id: row_id as u64,
            };
            index.add_embedding(&embedding, file_ref);
            count += 1;
        }
    }

    Ok(count)
}

/// Index embeddings from Array<Float32, N> column
fn index_array_embeddings(
    index: &mut LSHIndex,
    series: &Series,
    file_path: &str,
) -> Result<usize, ParquetError> {
    let array_chunked = series.array().map_err(|e| ParquetError::PolarsError(e))?;
    let expected_dim = index.lsh_hasher().embedding_dim;
    let mut count = 0;

    for (row_id, opt_series) in array_chunked.into_iter().enumerate() {
        if let Some(embedding_series) = opt_series {
            // Extract float values
            let floats = embedding_series
                .f32()
                .map_err(|e| ParquetError::PolarsError(e))?;

            let embedding: Vec<f32> = floats
                .into_iter()
                .filter_map(|opt_val| opt_val)
                .collect();

            // Validate dimension
            if embedding.len() != expected_dim {
                return Err(ParquetError::InvalidEmbeddingDim {
                    expected: expected_dim,
                    found: embedding.len(),
                });
            }

            // Add to index
            let file_ref = FileRef {
                file_path: file_path.to_string(),
                row_id: row_id as u64,
            };
            index.add_embedding(&embedding, file_ref);
            count += 1;
        }
    }

    Ok(count)
}

/// Index embeddings from multiple Parquet files
///
/// # Arguments
/// * `index` - LSH index to add embeddings to
/// * `file_paths` - Paths to Parquet files
/// * `embedding_column` - Name of column containing embeddings
///
/// # Returns
/// Total number of embeddings indexed across all files
pub fn index_parquet_files<P: AsRef<Path>>(
    index: &mut LSHIndex,
    file_paths: &[P],
    embedding_column: &str,
) -> Result<usize, ParquetError> {
    let mut total_count = 0;

    for file_path in file_paths {
        let count = index_parquet_file(index, file_path, embedding_column)?;
        total_count += count;
    }

    Ok(total_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn create_test_parquet_file(num_rows: usize, embedding_dim: usize) -> NamedTempFile {
        use polars::prelude::*;

        // Create embeddings as List<Float32>
        let mut embeddings = Vec::new();
        for i in 0..num_rows {
            let embedding: Vec<f32> = (0..embedding_dim)
                .map(|j| (i as f32 * 0.1) + (j as f32 * 0.001))
                .collect();
            embeddings.push(Some(Series::new("".into(), &embedding)));
        }

        let embedding_series = Series::new("embedding".into(), &embeddings);

        // Create DataFrame
        let df = DataFrame::new(vec![embedding_series.into()]).unwrap();

        // Write to temp file
        let temp_file = NamedTempFile::new().unwrap();
        let mut file = std::fs::File::create(temp_file.path()).unwrap();
        ParquetWriter::new(&mut file)
            .finish(&mut df.clone())
            .unwrap();

        temp_file
    }

    #[test]
    fn test_index_parquet_file() {
        let temp_file = create_test_parquet_file(10, 128);
        let mut index = LSHIndex::new(12345, 16, 128, 1000, 0.3);

        let count = index_parquet_file(&mut index, temp_file.path(), "embedding").unwrap();

        assert_eq!(count, 10);
        assert_eq!(index.indexed_files().len(), 1);
    }

    #[test]
    fn test_index_multiple_files() {
        let file1 = create_test_parquet_file(5, 128);
        let file2 = create_test_parquet_file(7, 128);
        let file3 = create_test_parquet_file(3, 128);

        let mut index = LSHIndex::new(12345, 16, 128, 1000, 0.3);

        let paths = vec![file1.path(), file2.path(), file3.path()];
        let total_count = index_parquet_files(&mut index, &paths, "embedding").unwrap();

        assert_eq!(total_count, 15);
        assert_eq!(index.indexed_files().len(), 3);
    }

    #[test]
    fn test_invalid_dimension() {
        let temp_file = create_test_parquet_file(10, 64); // Wrong dimension
        let mut index = LSHIndex::new(12345, 16, 128, 1000, 0.3); // Expects 128

        let result = index_parquet_file(&mut index, temp_file.path(), "embedding");

        assert!(result.is_err());
        match result.unwrap_err() {
            ParquetError::InvalidEmbeddingDim { expected, found } => {
                assert_eq!(expected, 128);
                assert_eq!(found, 64);
            }
            _ => panic!("Expected InvalidEmbeddingDim error"),
        }
    }
}
