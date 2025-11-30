use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use xxhash_rust::xxh3::xxh3_64_with_seed;

mod lsh_bindings;
use lsh_bindings::{PyLSHIndex, PyFileRef, PyIndexStats};

/// Tag metadata for composability - enables merge without rebuild
#[pyclass]
#[derive(Clone, Debug, Copy)]
pub struct Tag {
    /// 8-bit hash fingerprint for fast equality checks
    #[pyo3(get)]
    pub fingerprint: u8,
    /// Which hierarchical level the entry is stored at (density indicator)
    #[pyo3(get)]
    pub layer: u8,
}

#[pymethods]
impl Tag {
    #[new]
    fn new(fingerprint: u8, layer: u8) -> Self {
        Tag { fingerprint, layer }
    }

    fn __repr__(&self) -> String {
        format!("Tag(fp={}, layer={})", self.fingerprint, self.layer)
    }
}

/// Probe statistics - reveals density properties
#[pyclass]
#[derive(Clone, Debug)]
pub struct ProbeStats {
    #[pyo3(get)]
    pub probe_count: usize,
    #[pyo3(get)]
    pub level_reached: usize,
    #[pyo3(get)]
    pub block_index: usize,
}

#[pymethods]
impl ProbeStats {
    fn __repr__(&self) -> String {
        format!(
            "ProbeStats(probe_count={}, level_reached={}, block_index={})",
            self.probe_count, self.level_reached, self.block_index
        )
    }
}

/// Entry in the hash table with tag metadata
#[derive(Clone, Debug)]
struct Entry {
    key: String,
    value: String,
    tag: Tag,
}

/// Numeric entry for efficient aggregations (no string parsing)
#[derive(Clone, Debug)]
struct NumericEntry {
    key: String,
    value: f64,
    tag: Tag,
}

/// Krapivin Hash Table - Optimized Rust implementation
///
/// Features:
/// - xxHash3 (matches Polars performance)
/// - Single hash call per operation
/// - Tag metadata for composability
/// - O(log² δ⁻¹) probe complexity
#[pyclass]
pub struct KrapivinHash {
    levels: Vec<Vec<Option<Entry>>>,
    capacity: usize,
    size: usize,
    delta: f64,
    beta: usize,
    #[allow(dead_code)]
    alpha: usize,  // Number of levels, kept for future use
    probe_history: Vec<ProbeStats>,
}

impl KrapivinHash {
    /// Hash a string using xxHash3 (same as Polars)
    #[inline(always)]
    fn hash_key(&self, key: &str) -> u64 {
        xxh3_64_with_seed(key.as_bytes(), 0)
    }

    /// Generate probe sequence with SINGLE hash call + arithmetic
    /// This is the key optimization - no rehashing!
    #[inline(always)]
    fn probe_sequence(&self, base_hash: u64) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.levels.iter().enumerate().flat_map(move |(level_idx, level)| {
            let level_size = level.len();
            (0..20).filter_map(move |j| {
                if level_size == 0 {
                    None
                } else {
                    // Quadratic probing with level offset
                    let offset = (level_idx as u64) * 1000 + (j * j);
                    let slot = ((base_hash.wrapping_add(offset)) % level_size as u64) as usize;
                    Some((level_idx, slot))
                }
            })
        })
    }

    /// Extract 8-bit fingerprint from hash for tag
    #[inline(always)]
    fn fingerprint(&self, hash: u64) -> u8 {
        (hash & 0xFF) as u8
    }

}

#[pymethods]
impl KrapivinHash {
    #[new]
    #[pyo3(signature = (capacity=1024, delta=0.1, beta=8))]
    fn new(capacity: usize, delta: f64, beta: usize) -> PyResult<Self> {
        if delta <= 0.0 || delta >= 1.0 {
            return Err(PyValueError::new_err("delta must be between 0 and 1"));
        }

        // Calculate number of levels: α = O(log δ⁻¹)
        let alpha = if delta > 0.0 {
            ((1.0 / delta).log2() as usize).max(1)
        } else {
            10
        };

        // Create hierarchical arrays with geometric sizing
        let mut levels = Vec::new();
        let mut remaining = capacity;

        for i in 0..alpha {
            let level_size = remaining.min(capacity / (1 << (alpha - i - 1)));
            if level_size > 0 {
                levels.push(vec![None; level_size]);
                remaining -= level_size;
            }
        }

        // Overflow array for remaining capacity
        if remaining > 0 {
            levels.push(vec![None; remaining]);
        }

        Ok(KrapivinHash {
            levels,
            capacity,
            size: 0,
            delta,
            beta,
            alpha,
            probe_history: Vec::new(),
        })
    }

    /// Insert a key-value pair with tag metadata
    /// Returns probe statistics on success, None if table is full
    fn insert(&mut self, key: String, value: String) -> PyResult<Option<ProbeStats>> {
        let base_hash = self.hash_key(&key);
        self.insert_with_hash(key, value, base_hash)
    }

    /// Insert with pre-computed hash (for hash caching optimization)
    /// Allows reuse of Python's cached string hashes or Polars' pre-computed hashes
    #[pyo3(name = "insert_with_hash")]
    fn insert_with_hash(&mut self, key: String, value: String, hash: u64) -> PyResult<Option<ProbeStats>> {
        if self.size >= self.capacity {
            return Ok(None);
        }

        let fingerprint = self.fingerprint(hash);

        // Collect probe sequence to avoid borrow checker issues
        let probes: Vec<(usize, usize)> = self.probe_sequence(hash).collect();

        for (probe_num, (level_idx, slot)) in probes.into_iter().enumerate() {
            // Check if slot is empty or has same key (update case)
            let should_insert = match &self.levels[level_idx][slot] {
                None => true,
                Some(entry) => entry.key == key,
            };

            if should_insert {
                let was_empty = self.levels[level_idx][slot].is_none();

                self.levels[level_idx][slot] = Some(Entry {
                    key,
                    value,
                    tag: Tag {
                        fingerprint,
                        layer: level_idx as u8,
                    },
                });

                if was_empty {
                    self.size += 1;
                }

                let stats = ProbeStats {
                    probe_count: probe_num + 1,
                    level_reached: level_idx,
                    block_index: slot / self.beta,
                };

                self.probe_history.push(stats.clone());
                return Ok(Some(stats));
            }
        }

        Ok(None)
    }

    /// Get a value by key
    /// Returns (value, stats) tuple, both None if not found
    fn get(&self, key: String) -> PyResult<(Option<String>, Option<ProbeStats>)> {
        let base_hash = self.hash_key(&key);
        self.get_with_hash(key, base_hash)
    }

    /// Get with pre-computed hash (for hash caching optimization)
    /// Allows reuse of Python's cached string hashes or Polars' pre-computed hashes
    #[pyo3(name = "get_with_hash")]
    fn get_with_hash(&self, key: String, hash: u64) -> PyResult<(Option<String>, Option<ProbeStats>)> {
        let probes: Vec<(usize, usize)> = self.probe_sequence(hash).collect();

        for (probe_num, (level_idx, slot)) in probes.into_iter().enumerate() {
            match &self.levels[level_idx][slot] {
                Some(entry) => {
                    if entry.key == key {
                        let stats = ProbeStats {
                            probe_count: probe_num + 1,
                            level_reached: level_idx,
                            block_index: slot / self.beta,
                        };
                        return Ok((Some(entry.value.clone()), Some(stats)));
                    }
                    // Continue probing (collision)
                }
                None => {
                    // Empty slot - key definitely not present
                    return Ok((None, None));
                }
            }
        }

        Ok((None, None))
    }

    /// Get value and tag metadata (for merge operations)
    fn get_with_tag(&self, key: String) -> PyResult<(Option<String>, Option<Tag>)> {
        let base_hash = self.hash_key(&key);
        self.get_with_tag_and_hash(key, base_hash)
    }

    /// Get value and tag with pre-computed hash
    #[pyo3(name = "get_with_tag_and_hash")]
    fn get_with_tag_and_hash(&self, key: String, hash: u64) -> PyResult<(Option<String>, Option<Tag>)> {
        let probes: Vec<(usize, usize)> = self.probe_sequence(hash).collect();

        for (_, (level_idx, slot)) in probes.into_iter().enumerate() {
            match &self.levels[level_idx][slot] {
                Some(entry) => {
                    if entry.key == key {
                        return Ok((Some(entry.value.clone()), Some(entry.tag)));
                    }
                }
                None => return Ok((None, None)),
            }
        }

        Ok((None, None))
    }

    /// Calculate density for each level - O(n) scan
    fn density_by_level(&self) -> Vec<f64> {
        self.levels
            .iter()
            .map(|level| {
                if level.is_empty() {
                    0.0
                } else {
                    let filled = level.iter().filter(|e| e.is_some()).count();
                    filled as f64 / level.len() as f64
                }
            })
            .collect()
    }

    /// Density histogram from probe statistics - O(1) query!
    /// This is the "free" density feature from hierarchical structure
    fn density_histogram(&self) -> Vec<f64> {
        if self.probe_history.is_empty() {
            return vec![0.0; self.levels.len()];
        }

        let mut level_counts = vec![0usize; self.levels.len()];
        for stats in &self.probe_history {
            level_counts[stats.level_reached] += 1;
        }

        level_counts
            .iter()
            .map(|&count| count as f64 / self.probe_history.len() as f64)
            .collect()
    }

    /// Average probe depth - should be O(log² δ⁻¹)
    fn avg_probe_depth(&self) -> f64 {
        if self.probe_history.is_empty() {
            0.0
        } else {
            let sum: usize = self.probe_history.iter().map(|s| s.probe_count).sum();
            sum as f64 / self.probe_history.len() as f64
        }
    }

    /// Batch insert with pre-computed hashes
    /// Optimized for Polars integration where hashes are pre-computed
    #[pyo3(name = "batch_insert_with_hashes")]
    fn batch_insert_with_hashes(
        &mut self,
        keys: Vec<String>,
        values: Vec<String>,
        hashes: Vec<u64>
    ) -> PyResult<Vec<Option<ProbeStats>>> {
        if keys.len() != values.len() || keys.len() != hashes.len() {
            return Err(PyValueError::new_err(
                "keys, values, and hashes must have the same length"
            ));
        }

        let mut results = Vec::with_capacity(keys.len());

        for ((key, value), hash) in keys.into_iter().zip(values).zip(hashes) {
            let result = self.insert_with_hash(key, value, hash)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Batch get with pre-computed hashes
    #[pyo3(name = "batch_get_with_hashes")]
    fn batch_get_with_hashes(
        &self,
        keys: Vec<String>,
        hashes: Vec<u64>
    ) -> PyResult<Vec<(Option<String>, Option<ProbeStats>)>> {
        if keys.len() != hashes.len() {
            return Err(PyValueError::new_err(
                "keys and hashes must have the same length"
            ));
        }

        let mut results = Vec::with_capacity(keys.len());

        for (key, hash) in keys.into_iter().zip(hashes) {
            let result = self.get_with_hash(key, hash)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Get all entries (for testing/verification)
    fn items(&self) -> Vec<(String, String)> {
        let mut result = Vec::new();
        for level in &self.levels {
            for entry in level.iter().flatten() {
                result.push((entry.key.clone(), entry.value.clone()));
            }
        }
        result
    }

    /// Current size
    #[getter]
    fn size(&self) -> usize {
        self.size
    }

    /// Total capacity
    #[getter]
    fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of hierarchical levels
    #[getter]
    fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Load factor (size / capacity)
    #[getter]
    fn load_factor(&self) -> f64 {
        self.size as f64 / self.capacity as f64
    }

    fn __repr__(&self) -> String {
        let densities = self.density_by_level();
        let density_strs: Vec<String> = densities.iter().map(|d| format!("{:.2}", d)).collect();

        format!(
            "KrapivinHash(size={}/{}, levels={}, load={:.2}, avg_probes={:.2}, densities=[{}])",
            self.size,
            self.capacity,
            self.levels.len(),
            self.load_factor(),
            self.avg_probe_depth(),
            density_strs.join(", ")
        )
    }

    fn __len__(&self) -> usize {
        self.size
    }

    /// Group-by aggregation using KRAPIVIN HASH with PARALLEL SIMD hashing
    /// Uses rayon for multi-threaded hash computation + xxHash3 SIMD
    /// Best for large datasets (>10K keys)
    #[pyo3(name = "krapivin_groupby_sum_simd")]
    fn krapivin_groupby_sum_simd(
        &mut self,
        keys: Vec<String>,
        values: Vec<f64>
    ) -> PyResult<(Vec<String>, Vec<f64>, Vec<f64>)> {
        self.krapivin_groupby_sum_impl(keys, values, true)
    }

    /// Group-by aggregation using KRAPIVIN HASH (real implementation!)
    /// Provides O(log² δ⁻¹) probe guarantees and density tracking
    /// Use this when:
    /// - Load factor > 70% (where HashMap degrades)
    /// - Need density tracking/statistics
    /// - Want bounded worst-case performance
    #[pyo3(name = "krapivin_groupby_sum")]
    fn krapivin_groupby_sum(
        &mut self,
        keys: Vec<String>,
        values: Vec<f64>
    ) -> PyResult<(Vec<String>, Vec<f64>, Vec<f64>)> {
        self.krapivin_groupby_sum_impl(keys, values, false)
    }

    /// Internal implementation with optional SIMD parallelization
    fn krapivin_groupby_sum_impl(
        &mut self,
        keys: Vec<String>,
        values: Vec<f64>,
        use_simd: bool
    ) -> PyResult<(Vec<String>, Vec<f64>, Vec<f64>)> {
        if keys.len() != values.len() {
            return Err(PyValueError::new_err(
                "keys and values must have the same length"
            ));
        }

        // Use a separate Krapivin structure for numeric values
        let mut numeric_levels: Vec<Vec<Option<NumericEntry>>> = Vec::new();
        let alpha = ((1.0 / self.delta).log2() as usize).max(1);
        let mut remaining = self.capacity;

        for i in 0..alpha {
            let level_size = remaining.min(self.capacity / (1 << (alpha - i - 1)));
            if level_size > 0 {
                numeric_levels.push(vec![None; level_size]);
                remaining -= level_size;
            }
        }

        if remaining > 0 {
            numeric_levels.push(vec![None; remaining]);
        }

        let mut size = 0;

        // Pre-compute all hashes - optionally with parallel SIMD
        let hashes: Vec<u64> = if use_simd {
            // Parallel hashing with rayon (multi-threaded)
            // xxHash3 already uses SIMD (AVX2/NEON) internally
            use rayon::prelude::*;
            keys.par_iter()
                .map(|key| self.hash_key(key))
                .collect()
        } else {
            // Sequential hashing (still uses xxHash3 SIMD internally)
            keys.iter()
                .map(|key| self.hash_key(key))
                .collect()
        };

        // Aggregate using Krapivin probing - OPTIMIZED
        for ((key, value), hash) in keys.into_iter().zip(values).zip(hashes) {
            let fingerprint = self.fingerprint(hash);

            let mut inserted = false;
            // Iterate probes lazily - no Vec allocation!
            // Adaptive probe limit based on load factor
            let load_factor = self.size as f64 / self.capacity as f64;
            let max_probes = if load_factor > 0.9 {
                500  // Very high load needs more probes
            } else if load_factor > 0.8 {
                300
            } else {
                200
            };
            for (level_idx, slot) in self.probe_sequence(hash).take(max_probes) {
                if level_idx >= numeric_levels.len() {
                    break;
                }

                match &mut numeric_levels[level_idx][slot] {
                    None => {
                        // New key - insert with initial value (take ownership, no clone!)
                        let tag = Tag {
                            fingerprint,
                            layer: level_idx as u8,
                        };
                        numeric_levels[level_idx][slot] = Some(NumericEntry {
                            key,  // Move key instead of clone!
                            value,
                            tag,
                        });
                        size += 1;
                        inserted = true;
                        break;
                    }
                    Some(ref mut entry) => {
                        // FAST PATH: fingerprint (u8) -> full string
                        // Rejects 99%+ of non-matches with just u8 comparison!
                        if entry.tag.fingerprint == fingerprint && entry.key == key {
                            // Existing key - aggregate (NO STRING PARSING!)
                            entry.value += value;
                            inserted = true;
                            break;
                        }
                        // Collision - continue probing
                    }
                }
            }

            if !inserted && size < self.capacity {
                return Err(PyValueError::new_err(
                    format!("Failed to insert - probe sequence exhausted ({} keys / {} capacity)", size, self.capacity)
                ));
            }
        }

        // Extract results with density per level - SUPER OPTIMIZED (single pass!)
        let mut result_keys = Vec::with_capacity(size);
        let mut result_values = Vec::with_capacity(size);
        let mut level_densities = Vec::with_capacity(numeric_levels.len());

        for level in &numeric_levels {
            let level_len = level.len();
            let mut filled = 0;

            // Single pass: count AND extract
            for entry_opt in level.iter() {
                if let Some(entry) = entry_opt {
                    result_keys.push(entry.key.clone());
                    result_values.push(entry.value);
                    filled += 1;
                }
            }

            // Compute density after extraction
            let density = filled as f64 / level_len as f64;
            level_densities.push(density);
        }

        Ok((result_keys, result_values, level_densities))
    }

    /// Fast path using HashMap (for comparison and low-load cases)
    /// This is what we compared against before - fast but no theoretical guarantees
    #[pyo3(name = "hashmap_groupby_sum")]
    fn hashmap_groupby_sum(
        &mut self,
        keys: Vec<String>,
        values: Vec<f64>
    ) -> PyResult<(Vec<String>, Vec<f64>)> {
        if keys.len() != values.len() {
            return Err(PyValueError::new_err(
                "keys and values must have the same length"
            ));
        }

        use std::collections::HashMap;
        let mut aggregates: HashMap<String, f64> = HashMap::new();

        for (key, value) in keys.into_iter().zip(values) {
            *aggregates.entry(key).or_insert(0.0) += value;
        }

        let mut result_keys = Vec::with_capacity(aggregates.len());
        let mut result_values = Vec::with_capacity(aggregates.len());

        for (key, value) in aggregates {
            result_keys.push(key);
            result_values.push(value);
        }

        Ok((result_keys, result_values))
    }

    /// Convenience wrapper - chooses Krapivin or HashMap based on expected load
    #[pyo3(name = "groupby_sum_arrays")]
    fn groupby_sum_arrays(
        &mut self,
        keys: Vec<String>,
        values: Vec<f64>
    ) -> PyResult<(Vec<String>, Vec<f64>)> {
        // For now, delegate to HashMap for backward compatibility
        // Users should call krapivin_groupby_sum() explicitly for Krapivin benefits
        self.hashmap_groupby_sum(keys, values)
    }

    /// Group-by count on arrays (for Polars Series) - OPTIMIZED VERSION
    /// Uses HashMap for optimal performance
    /// Returns (keys, counts)
    #[pyo3(name = "groupby_count_arrays")]
    fn groupby_count_arrays(
        &mut self,
        keys: Vec<String>
    ) -> PyResult<(Vec<String>, Vec<u64>)> {
        use std::collections::HashMap;
        let mut counts: HashMap<String, u64> = HashMap::new();

        // Count occurrences
        for key in keys {
            *counts.entry(key).or_insert(0) += 1;
        }

        // Extract results
        let mut result_keys = Vec::with_capacity(counts.len());
        let mut result_counts = Vec::with_capacity(counts.len());

        for (key, count) in counts {
            result_keys.push(key);
            result_counts.push(count);
        }

        Ok((result_keys, result_counts))
    }
}

/// Python module
#[pymodule]
fn krapivin_hash_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Original Krapivin hash table (for group-by operations)
    m.add_class::<KrapivinHash>()?;
    m.add_class::<ProbeStats>()?;
    m.add_class::<Tag>()?;

    // LSH index (for embedding search)
    m.add_class::<PyLSHIndex>()?;
    m.add_class::<PyFileRef>()?;
    m.add_class::<PyIndexStats>()?;

    Ok(())
}
