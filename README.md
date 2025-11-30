# Semantic Proprioception Demo

**Understanding Data Through Self-Awareness**

A Streamlit demo showcasing automatic theme discovery across diverse datasets using LSH-based semantic clustering with Krapivin hash tables.

🔗 **[Live Demo](https://semantic-proprioception-demo.streamlit.app)** *(will update after deployment)*

## What is Semantic Proprioception?

Just as proprioception lets us sense our body's position, *semantic proprioception* allows data to reveal its own internal structure. This demo uses Locality-Sensitive Hashing (LSH) with Krapivin hash tables to automatically discover themes in text collections—no manual clustering, no predefined categories.

## Features

- **3 Diverse Datasets**: Twitter customer support (1,000 tweets), ArXiv research papers (1,000), Hacker News discussions (684 posts)
- **4 Embedding Models**: Compare MiniLM-L3/L6/L12 and MPNet-base
- **Automatic Theme Discovery**: Dense LSH buckets reveal common semantic patterns
- **Semantic Label Merging**: Similar themes combined using Jaccard similarity
- **Zero Runtime Costs**: All embeddings pre-computed (~24MB total)

## How It Works

1. **Embed**: Text → sentence-transformers → 384/768-dimensional vectors
2. **Hash**: Embeddings → LSH (fixed seed=12345) → bucket assignments
3. **Discover**: Dense buckets (≥5 items) → common themes
4. **Label**: LLM or keywords → semantic theme names
5. **Merge**: Similar labels → consolidated themes

## Quick Start

```bash
# Clone and run locally
git clone https://github.com/jdonaldson/semantic-proprioception-demo.git
cd semantic-proprioception-demo
pip install -r requirements.txt
streamlit run semantic_proprioception_demo.py
```

## Project Structure

```
semantic-proprioception-demo/
├── semantic_proprioception_demo.py    # Main Streamlit app
├── requirements.txt                    # Dependencies
├── semantic_proprioception_data/      # Twitter embeddings (8.6 MB)
├── arxiv_demo_data/                   # ArXiv embeddings (9.5 MB)
├── hackernews_demo_data/              # HN embeddings (5.6 MB)
└── krapivin-python/                   # LSH/Krapivin bindings
```

## Datasets

| Dataset | Items | Size | Example Themes |
|---------|-------|------|----------------|
| Twitter | 1,000 | 8.6 MB | Password resets, billing issues, tech support |
| ArXiv | 1,000 | 9.5 MB | Deep learning, quantum physics, genomics |
| Hacker News | 684 | 5.6 MB | AI/ML, startups, privacy, open source |

## Technical Stack

- **UI**: Streamlit
- **Data**: Polars (DataFrames), Apache Arrow
- **Embeddings**: sentence-transformers (HuggingFace)
- **LSH**: Custom implementation with Krapivin hash tables
- **Storage**: Parquet (compressed with zstd)

## About Krapivin Hash Tables

Traditional LSH rebuilds indexes when adding data. Krapivin hash tables enable:
- **O(1) density queries** - instant bucket size distribution
- **Incremental updates** - add/remove files without rebuilding
- **Merkle verification** - detect duplicates and verify integrity
- **Composability** - merge indexes from different sources

## References

- **Krapivin et al. (2025)**: [Optimal Hash Tables](https://arxiv.org/abs/2501.02305)
- **Indyk & Motwani (1998)**: Locality-Sensitive Hashing
- **sentence-transformers**: [HuggingFace Models](https://www.sbert.net/)

## License

MIT License

## Author

J. Justin Donaldson, Ph.D.
- Website: [jjd.io](https://jjd.io)
- GitHub: [@jdonaldson](https://github.com/jdonaldson)

## Related

This demo is part of the larger [Semantic Proprioception](https://github.com/jdonaldson/semantic-proprioception) research project exploring multi-resolution density awareness for embedding spaces.
