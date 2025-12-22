# khash Demo Data

Pre-computed embeddings and LSH indices for 4 models on 1,000 Twitter customer support tweets.

**Total size:** 8.6 MB (perfect for free hosting!)

## Files

- `models_metadata.parquet` - Model comparison stats
- `{model}_embeddings.parquet` - Tweet text + embeddings for each model
- `{model}_lsh_index.parquet` - Krapivin LSH index for each model

## Models

1. **MiniLM-L3** - Fastest (384D, 8 bits)
2. **MiniLM-L6** - Balanced (384D, 8 bits)
3. **MiniLM-L12** - Better quality (384D, 8 bits)
4. **MPNet-base** - Best quality (768D, 10 bits)

## Regenerate

```bash
python precompute_embeddings.py
```
