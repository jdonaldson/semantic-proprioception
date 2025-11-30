//! Demo of LSH index serialization
//!
//! Demonstrates:
//! - Creating LSH index
//! - Adding embeddings
//! - Saving to .krapivin file
//! - Loading from disk
//! - Querying loaded index

use krapivin_lsh::{LSHIndex, FileRef};

fn main() -> std::io::Result<()> {
    println!("=== Krapivin LSH Index Serialization Demo ===\n");

    // Configuration
    let lsh_seed = 12345;
    let num_bits = 16;
    let embedding_dim = 128;
    let capacity = 1000;
    let delta = 0.3;

    // Create index
    println!("Creating LSH index...");
    let mut index = LSHIndex::new(lsh_seed, num_bits, embedding_dim, capacity, delta);

    // Add some example embeddings
    println!("Adding embeddings from 3 files...\n");
    for file_idx in 0..3 {
        let file_path = format!("data/embeddings_{}.parquet", file_idx);
        index.track_file(file_path.clone());

        for row_idx in 0..10 {
            // Generate example embedding (normally from a model)
            let embedding: Vec<f32> = (0..embedding_dim)
                .map(|i| (file_idx as f32 * 0.1) + (i as f32 * 0.001))
                .collect();

            let file_ref = FileRef {
                file_path: file_path.clone(),
                row_id: row_idx,
            };

            index.add_embedding(&embedding, file_ref);
        }
    }

    // Show statistics
    let stats = index.stats();
    println!("Index statistics:");
    println!("  Files indexed: {}", stats.num_files);
    println!("  Buckets used: {}", stats.num_buckets);
    println!("  Load factor: {:.2}", stats.load_factor);
    println!("  Level densities: {:?}", stats.level_densities);

    // Save to disk
    let index_path = "/tmp/example.krapivin";
    println!("\nSaving index to {}...", index_path);
    index.save(index_path)?;

    // Get file size
    let file_size = std::fs::metadata(index_path)?.len();
    println!("Index file size: {} bytes", file_size);

    // Load from disk
    println!("\nLoading index from disk...");
    let loaded_index = LSHIndex::load(index_path)?;

    // Verify loaded index
    let loaded_stats = loaded_index.stats();
    println!("Loaded index statistics:");
    println!("  Files indexed: {}", loaded_stats.num_files);
    println!("  Buckets used: {}", loaded_stats.num_buckets);
    println!("  Load factor: {:.2}", loaded_stats.load_factor);

    // Query with a test embedding
    println!("\nQuerying with test embedding...");
    let query_embedding: Vec<f32> = (0..embedding_dim)
        .map(|i| 0.1 + (i as f32 * 0.001))
        .collect();

    match loaded_index.query(&query_embedding) {
        Some(refs) => {
            println!("Found {} similar vectors:", refs.len());
            for (i, file_ref) in refs.iter().take(5).enumerate() {
                println!("  {}. {} (row {})", i + 1, file_ref.file_path, file_ref.row_id);
            }
            if refs.len() > 5 {
                println!("  ... and {} more", refs.len() - 5);
            }
        }
        None => println!("No similar vectors found"),
    }

    println!("\n✓ Serialization demo complete!");
    Ok(())
}
