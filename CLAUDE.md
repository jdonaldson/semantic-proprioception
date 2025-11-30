# Semantic Proprioception

Rust/Python library and paper for "Semantic Proprioception: Introspective Embedding Spaces via LSH and Krapivin Hash"

## Project Structure

- `krapivin-core/` - Core Rust hashtable implementation
- `krapivin-lsh/` - LSH hasher with Hamming distance methods
- `krapivin-python/` - Python bindings (PyO3)
- `demo/` - Demo data, visualizations, experiments (gitignored)

## Building

```bash
cargo build --release
```

## Key Features

- Hierarchical LSH with O(log² δ⁻¹) probing (Krapivin et al.)
- Count-indexed extension for O(1) density queries
- Hamming distance methods: `hamming_neighbors(distance)` for d=1,2,3
