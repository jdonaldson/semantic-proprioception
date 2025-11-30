//! End-to-end workflow: Parquet files with embeddings → LSH index → Query
//!
//! Demonstrates:
//! - Creating Parquet files with embedding vectors
//! - Indexing multiple Parquet files
//! - Saving LSH index to disk
//! - Loading index and querying
//! - Getting file references for similar vectors

use krapivin_lsh::{LSHIndex, index_parquet_files};
use polars::prelude::*;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Krapivin LSH: Complete Parquet Workflow ===\n");

    // Configuration
    let lsh_seed = 12345;
    let num_bits = 16;
    let embedding_dim = 384; // Common for sentence transformers
    let capacity = 10000;
    let delta = 0.3;

    // Step 1: Create sample Parquet files with embeddings
    println!("Step 1: Creating sample Parquet files...");
    fs::create_dir_all("/tmp/krapivin_demo")?;

    let parquet_files = create_sample_parquet_files(
        "/tmp/krapivin_demo",
        3,    // 3 files
        100,  // 100 vectors per file
        embedding_dim,
    )?;

    for (i, path) in parquet_files.iter().enumerate() {
        let file_size = fs::metadata(path)?.len();
        println!("  Created {} ({} bytes)", path, file_size);
    }

    // Step 2: Build LSH index from Parquet files
    println!("\nStep 2: Building LSH index from Parquet files...");
    let mut index = LSHIndex::new(lsh_seed, num_bits, embedding_dim, capacity, delta);

    let total_indexed = index_parquet_files(&mut index, &parquet_files, "embedding")?;
    println!("  Indexed {} embeddings across {} files", total_indexed, parquet_files.len());

    // Show index statistics
    let stats = index.stats();
    println!("\nIndex statistics:");
    println!("  Buckets used: {}", stats.num_buckets);
    println!("  Load factor: {:.2}", stats.load_factor);
    println!("  Level densities: {:?}", stats.level_densities);

    // Step 3: Save index to disk
    let index_path = "/tmp/krapivin_demo/embeddings.krapivin";
    println!("\nStep 3: Saving index to {}...", index_path);
    index.save(index_path)?;

    let index_size = fs::metadata(index_path)?.len();
    println!("  Index file size: {} bytes ({:.2} KB)", index_size, index_size as f64 / 1024.0);

    // Step 4: Load index from disk
    println!("\nStep 4: Loading index from disk...");
    let loaded_index = LSHIndex::load(index_path)?;
    println!("  ✓ Index loaded successfully");

    // Step 5: Query for similar vectors
    println!("\nStep 5: Querying for similar vectors...");

    // Create a query vector (simulating a new embedding)
    let query_vector: Vec<f32> = (0..embedding_dim)
        .map(|i| 0.5 + (i as f32 * 0.001))
        .collect();

    match loaded_index.query(&query_vector) {
        Some(refs) => {
            println!("  Found {} similar vectors in LSH bucket:", refs.len());
            println!("\n  Top 10 results:");
            println!("  {:<40} {:<10}", "File", "Row ID");
            println!("  {:-<50}", "");

            for (i, file_ref) in refs.iter().take(10).enumerate() {
                let file_name = std::path::Path::new(&file_ref.file_path)
                    .file_name()
                    .and_then(|f| f.to_str())
                    .unwrap_or(&file_ref.file_path);
                println!("  {:<40} {:<10}", file_name, file_ref.row_id);
            }

            if refs.len() > 10 {
                println!("  ... and {} more", refs.len() - 10);
            }

            // Optional: Load actual embeddings from Parquet files to verify
            println!("\n  Verifying first result by loading from Parquet...");
            if let Some(first_ref) = refs.first() {
                let df = LazyFrame::scan_parquet(&first_ref.file_path, Default::default())?
                    .collect()?;
                println!("    File: {}", first_ref.file_path);
                println!("    Row: {}", first_ref.row_id);
                println!("    DataFrame shape: {:?}", df.shape());
            }
        }
        None => {
            println!("  No similar vectors found in LSH bucket");
        }
    }

    println!("\n✓ Complete workflow successful!");
    println!("\nCleanup: Files created in /tmp/krapivin_demo/");

    Ok(())
}

/// Create sample Parquet files with random embeddings
fn create_sample_parquet_files(
    dir: &str,
    num_files: usize,
    rows_per_file: usize,
    embedding_dim: usize,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut file_paths = Vec::new();

    for file_idx in 0..num_files {
        // Generate embeddings
        let mut embeddings = Vec::new();
        let mut text_ids = Vec::new();

        for row_idx in 0..rows_per_file {
            // Create embedding vector (simulating sentence-transformers output)
            let base_value = (file_idx as f32 * 0.1) + (row_idx as f32 * 0.01);
            let embedding: Vec<f32> = (0..embedding_dim)
                .map(|i| base_value + (i as f32 * 0.001))
                .collect();

            embeddings.push(Some(Series::new("".into(), &embedding)));
            text_ids.push(format!("doc_{}_chunk_{}", file_idx, row_idx));
        }

        // Create DataFrame
        let embedding_series = Series::new("embedding".into(), &embeddings);
        let text_id_series = Series::new("text_id".into(), &text_ids);

        let df = DataFrame::new(vec![text_id_series.into(), embedding_series.into()])?;

        // Write to Parquet
        let file_path = format!("{}/embeddings_{}.parquet", dir, file_idx);
        let mut file = std::fs::File::create(&file_path)?;
        ParquetWriter::new(&mut file)
            .finish(&mut df.clone())?;

        file_paths.push(file_path);
    }

    Ok(file_paths)
}
