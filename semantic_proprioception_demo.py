#!/usr/bin/env python3
"""
Semantic Proprioception - Multi-Model LSH Theme Discovery

Understanding data's internal structure through semantic self-awareness.

Demonstrates how different embedding models discover themes across diverse datasets:
- Twitter customer support tweets
- ArXiv research papers
- Hacker News posts

Uses Krapivin hash tables with semantic label merging. All embeddings
pre-computed for free hosting.

Features:
- Compare 4 different embedding models
- 3 diverse datasets (Twitter, ArXiv, Hacker News)
- Automatic theme discovery via LSH density
- Semantic label similarity merging
- Interactive theme exploration
- Model comparison visualization
"""

import streamlit as st
import polars as pl
import numpy as np
from pathlib import Path
from numpy.linalg import norm
import plotly.graph_objects as go

# LSH parquet utilities (inlined to avoid Rust dependency for Streamlit Cloud)
def load_dense_buckets_from_parquet(path: str, min_count: int = 5) -> pl.DataFrame:
    """Load dense buckets from Parquet file"""
    path_obj = Path(path)
    if path_obj.is_dir():
        pattern = str(path_obj / "partition_*.parquet")
        df = pl.scan_parquet(pattern)
    else:
        df = pl.scan_parquet(path)

    dense = (df
        .group_by('bucket_id')
        .agg(pl.count('row_id').alias('count'))
        .filter(pl.col('count') >= min_count)
        .sort('count', descending=True)
        .collect()
    )
    return dense


def index_stats_from_parquet(path: str) -> dict:
    """Compute index statistics from Parquet file"""
    path_obj = Path(path)
    if path_obj.is_dir():
        pattern = str(path_obj / "partition_*.parquet")
        df = pl.scan_parquet(pattern)
    else:
        df = pl.scan_parquet(path)

    bucket_sizes = (df
        .group_by('bucket_id')
        .agg(pl.count('row_id').alias('size'))
        .collect()
    )

    total_entries = bucket_sizes['size'].sum()
    num_buckets = len(bucket_sizes)
    avg_bucket_size = bucket_sizes['size'].mean()

    histogram = (bucket_sizes
        .group_by('size')
        .agg(pl.count('bucket_id').alias('count'))
        .sort('size', descending=True)
    )

    estimated_capacity = 256
    load_factor = num_buckets / estimated_capacity

    return {
        'total_entries': int(total_entries),
        'num_buckets': int(num_buckets),
        'avg_bucket_size': float(avg_bucket_size),
        'load_factor': float(load_factor),
        'bucket_size_histogram': histogram
    }

# Optimal seed from research
OPTIMAL_SEED = 31


# Page config
st.set_page_config(
    page_title="Semantic Proprioception",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_model_metadata(dataset="twitter"):
    """Load metadata about available models"""
    if dataset == "twitter":
        data_dir = Path("semantic_proprioception_data")
    elif dataset == "arxiv":
        data_dir = Path("arxiv_demo_data")
    elif dataset == "hackernews":
        data_dir = Path("hackernews_demo_data")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    metadata_path = data_dir / "models_metadata.parquet"

    if not metadata_path.exists():
        st.error(f"Metadata not found for {dataset}. Run precompute script first.")
        st.stop()

    return pl.read_parquet(metadata_path)


@st.cache_data
def load_model_data(model_name, dataset="twitter"):
    """Load embeddings and LSH index for a model"""
    if dataset == "twitter":
        data_dir = Path("semantic_proprioception_data")
    elif dataset == "arxiv":
        data_dir = Path("arxiv_demo_data")
    elif dataset == "hackernews":
        data_dir = Path("hackernews_demo_data")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Load embeddings
    embeddings_path = data_dir / f"{model_name}_embeddings.parquet"
    df = pl.read_parquet(embeddings_path)

    # Load LSH index (full index with bucket_id, file_path, row_id)
    index_path = data_dir / f"{model_name}_lsh_index.parquet"
    lsh_index = pl.read_parquet(index_path)

    # Load index stats
    index_stats = index_stats_from_parquet(str(index_path))

    # Load dense buckets (returns Polars DataFrame)
    dense_buckets_df = load_dense_buckets_from_parquet(str(index_path), min_count=5)

    # Convert to list of tuples for easier use
    dense_buckets = [
        (row["bucket_id"], row["count"])
        for row in dense_buckets_df.to_dicts()
    ]

    return df, lsh_index, index_stats, dense_buckets


@st.cache_resource
def build_search_index(df, _model, density_threshold=10):
    """Build density-aware search index (cached)"""
    # Extract embeddings properly - df['embedding'] is a Series of lists
    embedding_list = df['embedding'].to_list()
    embeddings = np.array(embedding_list)
    d = embeddings.shape[1]

    # Generate LSH hyperplanes (seed 31)
    np.random.seed(OPTIMAL_SEED)
    hyperplanes = np.random.randn(16, d)

    # Compute LSH hashes for all documents
    hash_bits = (embeddings @ hyperplanes.T > 0).astype(int)
    powers = 2 ** np.arange(16)
    hashes = hash_bits @ powers

    # Build index: bucket_id -> [doc_indices]
    index = {}
    for i, hash_val in enumerate(hashes & 0xFF):  # Level 0: 8 bits
        if hash_val not in index:
            index[hash_val] = []
        index[hash_val].append(i)

    # Compute density stats
    bucket_sizes = [len(indices) for indices in index.values()]
    density_stats = {
        'num_buckets': len(index),
        'avg_size': np.mean(bucket_sizes),
        'max_size': max(bucket_sizes),
        'dense_buckets': sum(1 for s in bucket_sizes if s >= density_threshold),
        'sparse_buckets': sum(1 for s in bucket_sizes if s < density_threshold),
    }

    return {
        'embeddings': embeddings,
        'hyperplanes': hyperplanes,
        'hashes': hashes,
        'index': index,
        'density_stats': density_stats,
        'density_threshold': density_threshold,
    }


def density_aware_search(query_text, model, search_index, df, k=5):
    """
    Density-aware semantic search

    Returns: (results, search_strategy)
    - results: List of (doc_id, similarity, title) tuples
    - search_strategy: Dict with bucket info and refinement details
    """
    # Encode query
    query_embedding = model.encode([query_text], convert_to_numpy=True)[0]

    # Compute query LSH hash
    query_hash_bits = (query_embedding @ search_index['hyperplanes'].T > 0).astype(int)
    powers = 2 ** np.arange(16)
    query_hash = query_hash_bits @ powers

    # Level 0: 8-bit bucket
    bucket_id = query_hash & 0xFF

    strategy = {'bucket_id': bucket_id}

    if bucket_id not in search_index['index']:
        return [], {'bucket_id': bucket_id, 'status': 'empty'}

    candidates = search_index['index'][bucket_id]
    bucket_size = len(candidates)
    strategy['bucket_size'] = bucket_size
    strategy['original_candidates'] = bucket_size

    # Adaptive refinement based on density
    if bucket_size >= search_index['density_threshold']:
        # Dense bucket - refine using bit slicing
        level_1_hash = (query_hash >> 8) & 0x0F

        # Filter candidates by level 1 hash
        refined_candidates = []
        for idx in candidates:
            doc_level_1 = (search_index['hashes'][idx] >> 8) & 0x0F
            if doc_level_1 == level_1_hash:
                refined_candidates.append(idx)

        strategy['status'] = 'refined'
        strategy['level_1_hash'] = level_1_hash
        strategy['refined_candidates'] = len(refined_candidates)

        candidates = refined_candidates if len(refined_candidates) > 0 else candidates
    else:
        strategy['status'] = 'sparse'

    # Rank candidates by cosine similarity
    candidate_embeddings = search_index['embeddings'][candidates]
    similarities = candidate_embeddings @ query_embedding / (
        norm(candidate_embeddings, axis=1) * norm(query_embedding)
    )

    # Get top-k
    top_k_indices = np.argsort(-similarities)[:k]

    results = []
    for idx in top_k_indices:
        doc_idx = candidates[idx]
        similarity = similarities[idx]

        # Get document info (convert to dict to get scalar values)
        doc_row = df[doc_idx].to_dicts()[0]

        if 'arxiv_id' in doc_row:
            doc_id = doc_row['arxiv_id']
            title = doc_row['title']
        elif 'hn_id' in doc_row:
            doc_id = doc_row['hn_id']
            title = doc_row['title']
        else:
            doc_id = doc_row['tweet_id']
            title = doc_row['text'][:100]

        results.append((doc_id, similarity, title))

    strategy['final_candidates'] = len(candidates)

    return results, strategy


def get_bucket_tweets(bucket_id, lsh_index, df, dataset="twitter"):
    """Get all items in a specific bucket (tweets, papers, or HN posts)"""
    # Filter LSH index for this bucket
    bucket_rows = lsh_index.filter(pl.col("bucket_id") == bucket_id)

    # Get row IDs
    row_ids = bucket_rows.get_column("row_id").to_list()

    # Get items from embeddings dataframe
    items = []
    for row_id in row_ids:
        if row_id < len(df):
            if dataset == "arxiv":
                # ArXiv papers: use title + abstract
                title = df[row_id, "title"]
                abstract = df[row_id, "abstract"]
                item_id = df[row_id, "arxiv_id"]
                category = df[row_id, "category_name"]
                text = f"{title}\n\n{abstract[:200]}..."  # Title + truncated abstract
                items.append({
                    "text": text,
                    "title": title,
                    "abstract": abstract,
                    "tweet_id": item_id,  # Keep same key for compatibility
                    "category": category,
                    "row_id": row_id
                })
            elif dataset == "hackernews":
                # Hacker News posts: use title + text
                title = df[row_id, "title"]
                post_text = df[row_id, "text"]
                item_id = df[row_id, "hn_id"]
                category = df[row_id, "category_name"]
                score = df[row_id, "score"]
                by = df[row_id, "by"]
                text = f"{title}\n\n{post_text[:200]}..."  # Title + truncated text
                items.append({
                    "text": text,
                    "title": title,
                    "post_text": post_text,
                    "tweet_id": item_id,  # Keep same key for compatibility
                    "category": category,
                    "score": score,
                    "by": by,
                    "row_id": row_id
                })
            else:
                # Twitter tweets
                tweet_text = df[row_id, "text"]
                tweet_id = df[row_id, "tweet_id"]
                items.append({"text": tweet_text, "tweet_id": tweet_id, "row_id": row_id})

    return items


def generate_theme_label(tweets, bucket_id):
    """Generate a short theme label for a bucket using LLM or keywords"""

    # Try Ollama first (local, free, fast)
    try:
        import subprocess

        # Check if Ollama is available
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=2
        )

        if result.returncode == 0 and 'llama3.2' in result.stdout:
            # Use Ollama to generate label
            sample_tweets = tweets[:5]
            tweets_text = "\n".join([f"- {t['text'][:100]}" for t in sample_tweets])

            prompt = f"""Generate a 3-5 word theme label for these similar customer support tweets.
Just the label, nothing else.

Tweets:
{tweets_text}

Label:"""

            result = subprocess.run(
                ['ollama', 'run', 'llama3.2:3b', prompt],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                label = result.stdout.strip()
                # Clean up
                label = label.replace('**', '').replace('*', '').strip('"\'')
                # Take first line if multiple
                label = label.split('\n')[0]
                # Limit length
                if len(label) > 50:
                    label = label[:47] + "..."
                return label
    except:
        pass

    # Fallback: keyword extraction
    all_text = " ".join([t["text"].lower() for t in tweets[:10]])

    # Common customer support keywords
    keywords = {
        "password": 0, "reset": 0, "login": 0, "account": 0,
        "charge": 0, "refund": 0, "cancel": 0, "billing": 0,
        "payment": 0, "order": 0, "delivery": 0, "shipping": 0,
        "help": 0, "support": 0, "issue": 0, "problem": 0,
        "email": 0, "phone": 0, "app": 0, "update": 0,
        "service": 0, "wifi": 0, "internet": 0, "connection": 0,
    }

    for keyword in keywords:
        keywords[keyword] = all_text.count(keyword)

    # Get top 3 keywords
    top_keywords = sorted(keywords.items(), key=lambda x: -x[1])[:3]
    top_keywords = [k for k, v in top_keywords if v > 0]

    if top_keywords:
        return " + ".join(top_keywords[:2]).title()
    else:
        return f"Theme {bucket_id}"


@st.cache_data
def get_bucket_labels(dense_buckets, lsh_index, df, dataset="twitter"):
    """Generate labels for all dense buckets (cached)"""
    labels = {}

    with st.spinner("Generating theme labels..."):
        for bucket_id, count in dense_buckets[:10]:
            items = get_bucket_tweets(bucket_id, lsh_index, df, dataset)
            label = generate_theme_label(items, bucket_id)
            labels[bucket_id] = label

    return labels


def compute_label_embeddings(labels, df):
    """
    Compute embeddings for theme labels using the same model as the tweets.

    Since embeddings are already computed, we'll use a simple TF-IDF-like approach
    for label similarity instead of loading the full model.
    """
    from collections import Counter
    import re

    # Tokenize labels
    label_tokens = {}
    for label in labels:
        # Simple tokenization: lowercase, split on non-alphanumeric
        tokens = re.findall(r'\w+', label.lower())
        label_tokens[label] = Counter(tokens)

    return label_tokens


def label_similarity(label1_tokens, label2_tokens):
    """Compute Jaccard similarity between two label token sets"""
    tokens1 = set(label1_tokens.keys())
    tokens2 = set(label2_tokens.keys())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    return intersection / union if union > 0 else 0.0


def merge_themes_by_label(dense_buckets, theme_labels, lsh_index, df, similarity_threshold=0.5, dataset="twitter"):
    """
    Merge buckets with semantically similar labels into unified themes.

    Args:
        dense_buckets: List of (bucket_id, count) tuples
        theme_labels: Dict mapping bucket_id to label string
        lsh_index: LSH index DataFrame
        df: Embeddings DataFrame
        similarity_threshold: Merge labels with similarity >= threshold

    Returns:
        List of merged themes with merged labels
    """
    from collections import defaultdict

    # Get unique labels
    bucket_list = [(bid, count, theme_labels.get(bid, f"Theme {bid}")) for bid, count in dense_buckets[:10]]

    # Compute label token sets for similarity comparison
    unique_labels = list(set(label for _, _, label in bucket_list))
    label_tokens = compute_label_embeddings(unique_labels, df)

    # Group similar labels using hierarchical clustering approach
    label_groups = []  # Each group is a list of similar labels
    label_to_group = {}  # Map label to its group index

    for label in unique_labels:
        # Check if this label is similar to any existing group
        found_group = False

        for group_idx, group in enumerate(label_groups):
            # Check similarity with representative (first) label in group
            representative = group[0]
            sim = label_similarity(label_tokens[label], label_tokens[representative])

            if sim >= similarity_threshold:
                # Add to this group
                group.append(label)
                label_to_group[label] = group_idx
                found_group = True
                break

        if not found_group:
            # Create new group
            label_groups.append([label])
            label_to_group[label] = len(label_groups) - 1

    # Choose best label for each group (longest/most descriptive)
    group_labels = []
    for group in label_groups:
        # Sort by length (longer = more descriptive), break ties alphabetically
        best_label = max(group, key=lambda x: (len(x), x))
        group_labels.append(best_label)

    # Map original labels to merged labels
    label_mapping = {}
    for label, group_idx in label_to_group.items():
        label_mapping[label] = group_labels[group_idx]

    # Group buckets by merged label
    merged_label_to_buckets = defaultdict(list)

    for bucket_id, count, original_label in bucket_list:
        merged_label = label_mapping[original_label]
        merged_label_to_buckets[merged_label].append((bucket_id, count, original_label))

    # Create merged themes
    merged_themes = []

    for merged_label, bucket_data in merged_label_to_buckets.items():
        # Get all tweets from all buckets with this merged label
        all_tweets = []
        total_count = 0
        buckets = []
        original_labels = set()

        for bucket_id, count, original_label in bucket_data:
            items = get_bucket_tweets(bucket_id, lsh_index, df, dataset)
            all_tweets.extend(items)
            total_count += count
            buckets.append((bucket_id, count))
            original_labels.add(original_label)

        merged_themes.append({
            'label': merged_label,
            'original_labels': list(original_labels),
            'buckets': buckets,
            'total_count': total_count,
            'tweets': all_tweets,
            'is_merged': len(original_labels) > 1,
        })

    # Sort by total count (descending)
    merged_themes.sort(key=lambda x: -x['total_count'])

    return merged_themes


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_similar(query_embedding, df, top_k=10):
    """Find most similar tweets using brute force cosine similarity"""
    embeddings = np.array(df.get_column("embedding").to_list())
    query = np.array(query_embedding)

    # Compute similarities
    similarities = np.array([
        cosine_similarity(query, emb) for emb in embeddings
    ])

    # Get top-k
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "text": df[idx, "text"],
            "similarity": similarities[idx],
            "tweet_id": df[idx, "tweet_id"],
        })

    return results


def lsh_search(query_embedding, df, index_path, top_k=10):
    """Find similar tweets using LSH index (load from parquet)"""
    # This would require loading the actual index structure
    # For now, return empty to show the comparison
    # In production, we'd load the PyLSHIndex and do bucket lookup
    return []


def render_sidebar():
    """Render sidebar with model selection and info"""
    st.sidebar.title("🧠 Semantic Proprioception")
    st.sidebar.caption("Understanding Data Through Self-Awareness")
    st.sidebar.markdown("---")

    # Dataset selection
    st.sidebar.subheader("Dataset")

    dataset_options = {
        "📱 Twitter Customer Support": "twitter",
        "📚 ArXiv Research Papers": "arxiv",
        "💬 Hacker News": "hackernews"
    }

    dataset_display = st.sidebar.selectbox(
        "Choose dataset",
        options=list(dataset_options.keys()),
        help="Compare how themes emerge in different domains"
    )
    dataset = dataset_options[dataset_display]

    st.sidebar.markdown("---")

    # Load metadata
    metadata = load_model_metadata(dataset)

    # Model selection
    st.sidebar.subheader("Select Model")

    model_options = metadata.get_column("model_name").to_list()
    model_descriptions = metadata.get_column("description").to_list()

    # Format options with descriptions
    display_options = [
        f"{name}: {desc.split(' - ')[0]}"
        for name, desc in zip(model_options, model_descriptions)
    ]

    selected_display = st.sidebar.radio(
        "Embedding Model",
        display_options,
        help="Different models cluster tweets differently"
    )

    # Extract model name
    selected_model = selected_display.split(":")[0]

    # Show model info
    model_row = metadata.filter(pl.col("model_name") == selected_model).to_dicts()[0]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Info")
    st.sidebar.metric("Dimension", f"{model_row['dimension']}")
    st.sidebar.metric("LSH Bits", f"{model_row['lsh_bits']}")
    st.sidebar.metric("Buckets Used", f"{model_row['num_buckets']}")
    st.sidebar.metric("Load Factor", f"{model_row['load_factor']:.1%}")
    st.sidebar.metric("Dense Themes", f"{model_row['dense_buckets']}")

    st.sidebar.markdown("---")

    # Theme merging settings
    st.sidebar.subheader("Theme Merging")
    similarity_threshold = st.sidebar.slider(
        "Label Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Merge themes with label similarity ≥ threshold (0=exact match only, 1=merge all)"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **About this demo:**
    - All embeddings pre-computed
    - No API keys needed
    - Free to host on Streamlit Cloud
    - Compare model clustering quality
    """)

    return selected_model, model_row, similarity_threshold, dataset


def render_themes_tab(df, lsh_index, dense_buckets, similarity_threshold=0.5, dataset="twitter"):
    """Render theme discovery tab"""
    st.header("🎯 Discovered Themes")

    if not dense_buckets:
        st.info("No dense themes found with current model")
        return

    # Generate labels for themes
    theme_labels = get_bucket_labels(dense_buckets, lsh_index, df, dataset)

    # Merge themes with semantically similar labels
    merged_themes = merge_themes_by_label(
        dense_buckets,
        theme_labels,
        lsh_index,
        df,
        similarity_threshold=similarity_threshold,
        dataset=dataset
    )

    st.markdown(f"""
    Found **{len(merged_themes)}** distinct themes from {len(dense_buckets)} dense buckets.
    Buckets with similar labels are automatically merged.
    """)

    # Show top themes
    st.subheader("Top Themes by Size")

    # Show merged themes
    for i, theme in enumerate(merged_themes, 1):
        label = theme['label']
        total_count = theme['total_count']
        buckets = theme['buckets']
        tweets = theme['tweets']
        is_merged = theme.get('is_merged', False)
        original_labels = theme.get('original_labels', [label])

        # Build bucket info string
        if len(buckets) > 1:
            bucket_info = f"{len(buckets)} buckets: " + ", ".join([hex(bid) for bid, _ in buckets[:3]])
            if len(buckets) > 3:
                bucket_info += f", +{len(buckets) - 3} more"
        else:
            bucket_info = f"bucket {hex(buckets[0][0])}"

        with st.expander(f"**{label}** — {total_count} tweets ({bucket_info})", expanded=(i==1)):
            # Show merge info
            if len(buckets) > 1:
                if is_merged:
                    st.info(f"🔗 **Semantically merged theme** from {len(buckets)} LSH buckets with similar labels")

                    # Show original labels that were merged
                    if len(original_labels) > 1:
                        with st.expander("📝 Original labels merged", expanded=False):
                            for orig_label in sorted(original_labels):
                                st.markdown(f"• {orig_label}")
                else:
                    st.info(f"📊 **Merged theme** from {len(buckets)} LSH buckets with identical labels")

                # Show bucket breakdown
                cols = st.columns(min(len(buckets), 4))
                for idx, (bucket_id, count) in enumerate(buckets[:4]):
                    with cols[idx % 4]:
                        st.metric(f"Bucket {hex(bucket_id)}", f"{count} tweets")

                st.markdown("---")

            st.markdown("### Sample Tweets")

            # Show first 8 tweets from merged theme
            for j, tweet in enumerate(tweets[:8], 1):
                st.markdown(f"**{j}.** {tweet['text']}")
                st.caption(f"Tweet ID: {tweet['tweet_id']} | Row: {tweet['row_id']}")
                st.markdown("")

            if len(tweets) > 8:
                st.caption(f"... and {len(tweets) - 8} more similar tweets")


@st.cache_resource
def load_embedding_model(model_name):
    """Load sentence transformer model (cached)"""
    from sentence_transformers import SentenceTransformer

    model_ids = {
        "MiniLM-L3": "sentence-transformers/paraphrase-MiniLM-L3-v2",
        "MiniLM-L6": "sentence-transformers/all-MiniLM-L6-v2",
        "MiniLM-L12": "sentence-transformers/all-MiniLM-L12-v2",
        "MPNet-base": "sentence-transformers/all-mpnet-base-v2",
    }

    return SentenceTransformer(model_ids[model_name])


def render_search_tab(df, model_name, dataset="twitter"):
    """Render density-aware semantic search tab"""
    st.header("🔎 Density-Aware Semantic Search")

    st.markdown("""
    **Adaptive search using LSH bucket density**:
    - Sparse buckets (< 10 docs): Returns all candidates
    - Dense buckets (≥ 10 docs): Automatically refines using bit slicing

    Search uses **seed 31** (optimal from our research) with O(1) hierarchical refinement.
    """)

    # Load model
    with st.spinner(f"Loading {model_name} model..."):
        model = load_embedding_model(model_name)

    # Build search index
    with st.spinner("Building search index..."):
        search_index = build_search_index(df, model, density_threshold=10)

    # Show index stats
    with st.expander("📊 Index Statistics", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Buckets", search_index['density_stats']['num_buckets'])
            st.metric("Avg Bucket Size", f"{search_index['density_stats']['avg_size']:.1f}")
        with col2:
            st.metric("Dense Buckets", search_index['density_stats']['dense_buckets'])
            st.metric("Sparse Buckets", search_index['density_stats']['sparse_buckets'])
        with col3:
            st.metric("Max Bucket Size", search_index['density_stats']['max_size'])
            st.metric("Density Threshold", search_index['density_threshold'])

    st.markdown("---")

    # Search input
    dataset_examples = {
        "twitter": "e.g., 'password reset problem', 'billing question', 'account locked'",
        "arxiv": "e.g., 'neural networks', 'graph algorithms', 'quantum computing'",
        "hackernews": "e.g., 'programming languages', 'startup advice', 'web development'"
    }

    query = st.text_input(
        "Enter your search query",
        placeholder=dataset_examples.get(dataset, "e.g., 'password reset problem'"),
        help="The model will find semantically similar content"
    )

    if query:
        st.markdown("---")

        with st.spinner("Searching..."):
            results, strategy = density_aware_search(query, model, search_index, df, k=5)

        # Show search strategy
        if strategy['status'] == 'empty':
            st.warning(f"No documents in bucket {hex(strategy['bucket_id'])}. Try a different query.")
        else:
            # Strategy visualization
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Bucket ID", hex(strategy['bucket_id']))
                st.metric("Bucket Size", strategy['bucket_size'])

            with col2:
                if strategy['status'] == 'refined':
                    st.metric("Strategy", "🔍 Dense → Refined")
                    st.metric("Level 1 Hash", hex(strategy['level_1_hash']))
                else:
                    st.metric("Strategy", "📊 Sparse → Direct")
                    st.caption("Bucket small enough, no refinement")

            with col3:
                if strategy['status'] == 'refined':
                    st.metric("Original Candidates", strategy['original_candidates'])
                    st.metric("After Refinement", strategy['refined_candidates'])
                    reduction = (1 - strategy['refined_candidates'] / strategy['original_candidates']) * 100
                    st.caption(f"Reduced by {reduction:.0f}%")
                else:
                    st.metric("Candidates", strategy['final_candidates'])

            st.markdown("---")

            # Show results
            if results:
                st.subheader(f"Top {len(results)} Results")

                for i, (doc_id, similarity, title) in enumerate(results, 1):
                    with st.expander(f"**#{i}** — Similarity: {similarity:.3f}", expanded=(i==1)):
                        st.markdown(f"**{title}**")
                        st.caption(f"Document ID: {doc_id}")

                        # Show full content if available
                        doc_row = df.filter(
                            (pl.col('arxiv_id') == doc_id) if 'arxiv_id' in df.columns
                            else (pl.col('hn_id') == doc_id) if 'hn_id' in df.columns
                            else (pl.col('tweet_id') == doc_id)
                        )

                        if len(doc_row) > 0:
                            doc = doc_row.to_dicts()[0]

                            if 'abstract' in doc and doc['abstract']:
                                st.markdown("**Abstract:**")
                                abstract_text = str(doc['abstract'])
                                st.markdown(abstract_text[:500] + "..." if len(abstract_text) > 500 else abstract_text)
                            elif 'text' in doc and doc['text'] and str(doc['text']) != str(title):
                                st.markdown(str(doc['text']))
            else:
                st.info("No results found after refinement. Try a different query.")


@st.cache_data
def load_umap_data(model_name):
    """Load pre-computed UMAP 3D data for specific model"""
    umap_path = Path(__file__).parent / f"umap_3d_data_{model_name}.parquet"
    if umap_path.exists():
        return pl.read_parquet(umap_path)
    return None


def render_umap_tab(df, model_name, dataset="twitter"):
    """Render 3D UMAP visualization tab"""
    st.markdown("### 3D UMAP Visualization of Embedding Space")

    st.info("""
    **UMAP** (Uniform Manifold Approximation and Projection) reduces high-dimensional embeddings
    to 3D while preserving semantic structure. Points that are close in the original high-D space
    stay close in this 3D projection.

    **LSH buckets** (colored regions) partition this space. Hover over points to see their bucket assignments.
    """)

    # Load UMAP data
    umap_df = load_umap_data(model_name)

    if umap_df is None:
        st.warning("UMAP visualization not available. Run `python test_umap_visualization.py` to generate it.")
        return

    st.success(f"Showing {len(umap_df):,} points in 3D")

    # Visualization type selector
    viz_type = st.radio(
        "Visualization type:",
        ["LSH Buckets", "Multi-Probe Query"],
        horizontal=True
    )

    if viz_type == "LSH Buckets":
        # Color by bucket density (more meaningful than bucket ID)
        fig = go.Figure(data=[go.Scatter3d(
            x=umap_df['x'],
            y=umap_df['y'],
            z=umap_df['z'],
            mode='markers',
            marker=dict(
                size=3,
                color=umap_df['bucket_size'],  # Color by density
                colorscale='RdYlBu_r',  # Red (dense) → Yellow → Blue (sparse)
                showscale=True,
                colorbar=dict(title="Bucket<br>Density"),
                line=dict(width=0),
                cmin=1,  # Minimum bucket size
                cmax=umap_df['bucket_size'].max(),  # Maximum bucket size
            ),
            text=umap_df['text'],
            customdata=np.column_stack((umap_df['bucket'], umap_df['bucket_size'])),
            hovertemplate='<b>Bucket:</b> %{customdata[0]}<br>' +
                          '<b>Density:</b> %{customdata[1]} docs<br>' +
                          '<b>Text:</b> %{text}<br>' +
                          '<extra></extra>',
        )])

        fig.update_layout(
            title='3D UMAP: Colored by LSH Bucket Density',
            scene=dict(
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                zaxis_title='UMAP 3',
            ),
            height=900,
            hovermode='closest',
        )

        st.plotly_chart(fig, use_container_width=True)

        # Calculate density stats
        max_density = umap_df['bucket_size'].max()
        min_density = umap_df['bucket_size'].min()
        avg_density = umap_df['bucket_size'].mean()
        dense_threshold = 10

        num_dense = len([s for s in umap_df['bucket_size'] if s >= dense_threshold])
        pct_dense = 100 * num_dense / len(umap_df)

        st.markdown(f"""
        **Density Analysis:**
        - {len(umap_df['bucket'].unique())} unique LSH buckets
        - Bucket sizes: {min_density} to {max_density} documents (avg: {avg_density:.1f})
        - <span style="color:red">**Red regions**</span>: Dense buckets (many similar documents)
        - <span style="color:blue">**Blue regions**</span>: Sparse buckets (few similar documents)
        - {pct_dense:.1f}% of points in dense buckets (≥{dense_threshold} docs)

        **What this reveals**: Dense regions indicate common themes or topics where many tweets cluster together.
        Sparse regions represent more unique or diverse content.
        """, unsafe_allow_html=True)

    else:  # Multi-Probe Query
        st.markdown("Enter a query to see how multi-probe LSH explores the embedding space:")

        query_text = st.text_input(
            "Query:",
            value="my account was hacked",
            help="Enter a search query to visualize multi-probe LSH"
        )

        if query_text:
            # Load sentence transformer model
            from sentence_transformers import SentenceTransformer

            # Model IDs
            model_ids = {
                "MiniLM-L3": "sentence-transformers/paraphrase-MiniLM-L3-v2",
                "MiniLM-L6": "sentence-transformers/all-MiniLM-L6-v2",
                "MiniLM-L12": "sentence-transformers/all-MiniLM-L12-v2",
                "MPNet-base": "sentence-transformers/all-mpnet-base-v2",
            }

            @st.cache_resource
            def load_sentence_model(model_name):
                return SentenceTransformer(model_ids[model_name])

            model = load_sentence_model(model_name)

            # Compute query embedding and hash
            query_embedding = model.encode([query_text], convert_to_numpy=True)[0]

            # Load hyperplanes
            embedding_list = df['embedding'].to_list()
            embeddings = np.array(embedding_list)
            d = embeddings.shape[1]

            np.random.seed(OPTIMAL_SEED)
            hyperplanes = np.random.randn(16, d)

            query_hash_bits = (query_embedding @ hyperplanes.T > 0).astype(int)
            powers = 2 ** np.arange(16)
            query_hash = query_hash_bits @ powers
            query_bucket = query_hash & 0xFF

            # Generate 2-bit flip probes
            def flip_bits_2(bucket_id):
                buckets = [bucket_id]
                for bit_pos in range(8):
                    buckets.append(bucket_id ^ (1 << bit_pos))
                for bit_pos1 in range(8):
                    for bit_pos2 in range(bit_pos1 + 1, 8):
                        buckets.append(bucket_id ^ (1 << bit_pos1) ^ (1 << bit_pos2))
                return buckets

            probe_buckets = set(flip_bits_2(query_bucket))

            # Assign colors
            colors = []
            for bucket in umap_df['bucket_int']:
                if bucket == query_bucket:
                    colors.append('red')
                elif bucket in probe_buckets:
                    colors.append('orange')
                else:
                    colors.append('lightgray')

            # Create figure
            fig = go.Figure()

            # Add all points
            fig.add_trace(go.Scatter3d(
                x=umap_df['x'],
                y=umap_df['y'],
                z=umap_df['z'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=colors,
                    line=dict(width=0),
                    opacity=0.6,
                ),
                text=umap_df['text'],
                customdata=np.column_stack((umap_df['bucket'], umap_df['bucket_size'])),
                hovertemplate='<b>Bucket:</b> %{customdata[0]}<br>' +
                              '<b>Size:</b> %{customdata[1]}<br>' +
                              '<b>Text:</b> %{text}<br>' +
                              '<extra></extra>',
                name='Tweets',
            ))

            fig.update_layout(
                title=f'Multi-Probe LSH: "{query_text}"<br>' +
                      f'<span style="color:red">■</span> Query bucket (0x{query_bucket:02x}) | ' +
                      f'<span style="color:orange">■</span> Probed buckets ({len(probe_buckets)}) | ' +
                      f'<span style="color:gray">■</span> Other buckets',
                scene=dict(
                    xaxis_title='UMAP 1',
                    yaxis_title='UMAP 2',
                    zaxis_title='UMAP 3',
                ),
                height=900,
                showlegend=False,
                hovermode='closest',
            )

            st.plotly_chart(fig, use_container_width=True)

            # Stats
            query_bucket_size = len([b for b in umap_df['bucket_int'] if b == query_bucket])
            probed_bucket_sizes = [len([b for b in umap_df['bucket_int'] if b == pb]) for pb in probe_buckets]
            total_candidates = sum(probed_bucket_sizes)

            st.markdown(f"""
            **Multi-Probe Statistics:**
            - Query bucket: `0x{query_bucket:02x}` ({query_bucket_size} documents)
            - Probed buckets: {len(probe_buckets)} (2-bit flip)
            - Total candidates: {total_candidates} ({100*total_candidates/len(umap_df):.1f}% of dataset)
            - Speedup: ~{len(umap_df)/max(total_candidates, 1):.1f}× faster than brute force
            """)


def render_comparison_tab(metadata):
    """Render model comparison tab"""
    st.header("📊 Model Comparison")

    st.markdown("""
    Different embedding models produce different semantic clusterings.
    Compare how each model organizes the same customer support tweets.
    """)

    # Show comparison table
    st.subheader("Model Statistics")

    comparison_df = metadata.select([
        "model_name",
        "dimension",
        "lsh_bits",
        "num_buckets",
        "load_factor",
        "dense_buckets",
    ]).to_pandas()

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    # Insights
    st.subheader("Key Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Dimension vs Quality:**
        - Higher dimension = better semantic understanding
        - MPNet-base (768D) captures more nuance
        - MiniLM models (384D) balance speed/quality
        """)

    with col2:
        st.markdown("""
        **LSH Bucketing:**
        - More bits = more buckets = finer clustering
        - Dense buckets reveal common patterns
        - Load factor shows index efficiency
        """)

    # Chart would go here
    st.info("💡 Switch models in the sidebar to see how clustering changes")


def main():
    """Main app"""

    # Sidebar
    selected_model, model_info, similarity_threshold, dataset = render_sidebar()

    # Load data for selected model and dataset
    with st.spinner(f"Loading {selected_model} data for {dataset}..."):
        df, lsh_index, index_stats, dense_buckets = load_model_data(selected_model, dataset)

    # Main header
    st.title("🧠 Semantic Proprioception")
    st.caption("Multi-Model Theme Discovery via LSH Density Analysis")

    # Dataset-specific description
    dataset_desc = {
        "twitter": f"{len(df):,} customer support tweets from major brands",
        "arxiv": f"{len(df):,} research paper abstracts across 10 scientific fields",
        "hackernews": f"{len(df):,} Hacker News posts from top stories, discussions, and Show HN"
    }

    st.markdown(f"""
    Comparing **4 different embedding models** on {dataset_desc[dataset]}.
    Themes emerge automatically from LSH bucket density—the data reveals its own structure.

    **Currently viewing:** {selected_model} ({model_info['dimension']} dimensions) on **{dataset.title()}** dataset
    """)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Themes", "🔎 Search", "🗺️ 3D Map", "📊 Comparison"])

    with tab1:
        render_themes_tab(df, lsh_index, dense_buckets, similarity_threshold, dataset)

    with tab2:
        render_search_tab(df, selected_model, dataset)

    with tab3:
        render_umap_tab(df, selected_model, dataset)

    with tab4:
        metadata = load_model_metadata()
        render_comparison_tab(metadata)

    # Footer
    st.markdown("---")
    st.caption(f"""
    **Semantic Proprioception** — Data understanding through self-awareness |
    [Dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) |
    Krapivin LSH + sentence-transformers |
    Fixed seed for composability
    """)


if __name__ == "__main__":
    main()
