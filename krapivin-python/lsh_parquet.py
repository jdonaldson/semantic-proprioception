#!/usr/bin/env python3
"""
Parquet storage for Krapivin LSH indices

Provides efficient columnar storage and retrieval for LSH index data.
Supports both single-file and partitioned storage for scalability.
"""

import polars as pl
from pathlib import Path
from typing import Optional, List, Tuple
from krapivin_hash_rs import PyLSHIndex


def save_index_to_parquet(
    index: PyLSHIndex,
    path: str,
    compression: str = 'zstd'
) -> None:
    """
    Save LSH index to a single Parquet file

    Args:
        index: PyLSHIndex to save
        path: Output Parquet file path
        compression: Compression codec ('zstd', 'snappy', 'gzip', 'lz4', 'uncompressed')

    Example:
        >>> index = PyLSHIndex(seed=12345, num_bits=8, embedding_dim=384, capacity=10000, delta=0.3)
        >>> # ... add embeddings ...
        >>> save_index_to_parquet(index, "index.parquet")
    """
    # Export all data from Rust
    records = index.export_data()

    # Convert to Polars DataFrame
    df = pl.DataFrame({
        'bucket_id': [r[0] for r in records],
        'file_path': [r[1] for r in records],
        'row_id': [r[2] for r in records],
    })

    # Write to Parquet
    df.write_parquet(path, compression=compression)

    print(f"✓ Saved {len(records):,} entries to {path}")
    print(f"  File size: {Path(path).stat().st_size / 1024 / 1024:.1f} MB")


def save_index_partitioned(
    index: PyLSHIndex,
    base_path: str,
    num_partitions: int = 16,
    compression: str = 'zstd'
) -> None:
    """
    Save LSH index to partitioned Parquet files

    Partitions by bucket_id prefix for efficient querying.

    Args:
        index: PyLSHIndex to save
        base_path: Base directory for partition files
        num_partitions: Number of partitions (must be power of 2)
        compression: Compression codec

    Example:
        >>> save_index_partitioned(index, "index_parts/", num_partitions=16)
        >>> # Creates: index_parts/partition_00.parquet, partition_01.parquet, ...
    """
    base = Path(base_path)
    base.mkdir(exist_ok=True, parents=True)

    # Export all data
    records = index.export_data()

    # Calculate partition bits
    import math
    partition_bits = int(math.log2(num_partitions))
    partition_mask = num_partitions - 1

    # Group by partition
    partitions = {}
    for bucket_id, file_path, row_id in records:
        # Use top bits of bucket_id for partitioning
        partition_id = (bucket_id >> (64 - partition_bits)) & partition_mask

        if partition_id not in partitions:
            partitions[partition_id] = []

        partitions[partition_id].append({
            'bucket_id': bucket_id,
            'file_path': file_path,
            'row_id': row_id
        })

    # Write each partition
    total_size = 0
    for partition_id, partition_records in partitions.items():
        df = pl.DataFrame(partition_records)

        partition_path = base / f"partition_{partition_id:02x}.parquet"
        df.write_parquet(partition_path, compression=compression)

        file_size = partition_path.stat().st_size
        total_size += file_size

        print(f"  Partition {partition_id:2d}: {len(partition_records):6,} entries, "
              f"{file_size / 1024 / 1024:5.1f} MB")

    print(f"\n✓ Saved {len(records):,} entries across {len(partitions)} partitions")
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")


def load_dense_buckets_from_parquet(
    path: str,
    min_count: int = 5
) -> pl.DataFrame:
    """
    Load dense buckets from Parquet file

    Args:
        path: Parquet file path (or directory for partitioned)
        min_count: Minimum bucket size threshold

    Returns:
        Polars DataFrame with columns: bucket_id, count
        Sorted by count descending

    Example:
        >>> dense = load_dense_buckets_from_parquet("index.parquet", min_count=5)
        >>> print(dense)
        ┌───────────┬───────┐
        │ bucket_id │ count │
        │ ---       │ ---   │
        │ u64       │ u32   │
        ╞═══════════╪═══════╡
        │ 132       │ 16    │
        │ 196       │ 16    │
        │ ...       │ ...   │
        └───────────┴───────┘
    """
    path_obj = Path(path)

    if path_obj.is_dir():
        # Partitioned format
        pattern = str(path_obj / "partition_*.parquet")
        df = pl.scan_parquet(pattern)
    else:
        # Single file
        df = pl.scan_parquet(path)

    # Compute dense buckets
    dense = (df
        .group_by('bucket_id')
        .agg(pl.count('row_id').alias('count'))
        .filter(pl.col('count') >= min_count)
        .sort('count', descending=True)
        .collect()
    )

    return dense


def get_bucket_contents_from_parquet(
    path: str,
    bucket_id: int
) -> pl.DataFrame:
    """
    Load contents of a specific bucket from Parquet

    Args:
        path: Parquet file path (or directory for partitioned)
        bucket_id: Bucket to retrieve

    Returns:
        Polars DataFrame with columns: bucket_id, file_path, row_id

    Example:
        >>> contents = get_bucket_contents_from_parquet("index.parquet", 132)
        >>> print(contents)
        ┌───────────┬──────────────┬────────┐
        │ bucket_id │ file_path    │ row_id │
        │ ---       │ ---          │ ---    │
        │ u64       │ str          │ u64    │
        ╞═══════════╪══════════════╪════════╡
        │ 132       │ tweets.parq  │ 42     │
        │ 132       │ tweets.parq  │ 105    │
        │ ...       │ ...          │ ...    │
        └───────────┴──────────────┴────────┘
    """
    path_obj = Path(path)

    if path_obj.is_dir():
        # Partitioned format - determine which partition
        # For now, scan all (could optimize to target specific partition)
        pattern = str(path_obj / "partition_*.parquet")
        df = pl.scan_parquet(pattern)
    else:
        # Single file
        df = pl.scan_parquet(path)

    # Filter to bucket
    result = (df
        .filter(pl.col('bucket_id') == bucket_id)
        .collect()
    )

    return result


def index_stats_from_parquet(path: str) -> dict:
    """
    Compute index statistics from Parquet file

    Args:
        path: Parquet file path (or directory for partitioned)

    Returns:
        Dictionary with statistics:
        - total_entries: Total number of indexed embeddings
        - num_buckets: Number of non-empty buckets
        - avg_bucket_size: Average entries per bucket
        - load_factor: Estimated load factor
        - bucket_size_histogram: Distribution of bucket sizes

    Example:
        >>> stats = index_stats_from_parquet("index.parquet")
        >>> print(f"Load factor: {stats['load_factor']:.2f}")
        Load factor: 0.73
    """
    path_obj = Path(path)

    if path_obj.is_dir():
        pattern = str(path_obj / "partition_*.parquet")
        df = pl.scan_parquet(pattern)
    else:
        df = pl.scan_parquet(path)

    # Compute statistics
    bucket_sizes = (df
        .group_by('bucket_id')
        .agg(pl.count('row_id').alias('size'))
        .collect()
    )

    total_entries = bucket_sizes['size'].sum()
    num_buckets = len(bucket_sizes)
    avg_bucket_size = bucket_sizes['size'].mean()

    # Histogram
    histogram = (bucket_sizes
        .group_by('size')
        .agg(pl.count('bucket_id').alias('count'))
        .sort('size', descending=True)
    )

    # Estimate load factor (approximate - would need capacity)
    # Assume 256 buckets for now (8 bits)
    estimated_capacity = 256
    load_factor = num_buckets / estimated_capacity

    return {
        'total_entries': int(total_entries),
        'num_buckets': int(num_buckets),
        'avg_bucket_size': float(avg_bucket_size),
        'load_factor': float(load_factor),
        'bucket_size_histogram': histogram
    }


def merge_parquet_indices(
    paths: List[str],
    output_path: str,
    compression: str = 'zstd'
) -> None:
    """
    Merge multiple Parquet index files into one

    Useful for combining partial indices or streaming builds.

    Args:
        paths: List of Parquet file paths to merge
        output_path: Output merged Parquet file
        compression: Compression codec

    Example:
        >>> merge_parquet_indices(
        ...     ["index1.parquet", "index2.parquet"],
        ...     "merged.parquet"
        ... )
    """
    # Load and concatenate all files
    dfs = [pl.scan_parquet(path) for path in paths]
    merged = pl.concat(dfs).collect()

    # Write merged file
    merged.write_parquet(output_path, compression=compression)

    print(f"✓ Merged {len(paths)} files into {output_path}")
    print(f"  Total entries: {len(merged):,}")
    print(f"  File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")
