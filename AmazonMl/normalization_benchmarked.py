"""
Entity Resolution Pipeline — Instrumented with Benchmarking
=============================================================

This is the ORIGINAL pipeline with a non-invasive benchmarking layer.
The matching algorithm, feature engineering, candidate generation, and
ML logic are IDENTICAL to normalization.py.

The only additions are:
  - time.perf_counter() calls around every major stage
  - psutil RSS snapshots at key checkpoints
  - Candidate quality metrics computed from existing structures
  - Model quality metrics on the validation split
  - A structured benchmark report saved at the end

All existing outputs are preserved:
  - output/candidate_pairs.tsv
  - output/matching_results.tsv

New outputs:
  - output/benchmark_metrics.json
  - output/benchmark_report.txt
  - output/benchmark_history.csv  (appended per run)
"""

import os
import re
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from rapidfuzz import fuzz, distance
import xgboost as xgb
from scipy import sparse
from tqdm.auto import tqdm
import gc

from benchmark import (
    BenchmarkCollector,
    Timer,
    compute_dataset_stats,
    compute_candidate_quality,
    compute_model_quality,
    compute_blocking_efficiency,
)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PIPELINE CONFIGURATION                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

PIPELINE_VERSION = "baseline_v1"

# Algorithm parameters
TOP_K = 50          # Candidates per S1 entity
BATCH_SIZE = 5000   # S1 entities per batch during candidate retrieval

# XGBoost hyperparameters
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "random_state": 42,
    "n_jobs": -1,
}


# ==============================================================================
# STEP 1: NORMALIZATION (Memory-Optimized)
# ==============================================================================
def normalize_dataset(df, drop_raw=True):
    """
    Applies vectorized string normalization.
    - Lowercase, punctuation removal, abbreviation normalization.
    - Does NOT create address_digits or blocking_text columns
      to save ~1.5 GB on large datasets.
    - Optionally drops raw columns to further save memory.
    """
    df = df.copy()
    for col in ['business_name', 'business_address', 'country']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)

    # Lowercase & strip punctuation
    for col, new_col in [('business_name', 'clean_name'),
                         ('business_address', 'clean_address'),
                         ('country', 'clean_country')]:
        df[new_col] = (
            df[col].str.lower()
            .str.replace(r'[^\w\s]', ' ', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
        )

    # Expand abbreviations & normalize legal suffixes
    suffix_map = {
        r'\b(corporation|corp|incorporated)\b': 'inc',
        r'\b(limited|ltd|pvt ltd|private limited|private)\b': 'ltd',
        r'\b(limited liability company|llc)\b': 'llc',
        r'\b(street|st)\b': 'st',
        r'\b(road|rd)\b': 'rd',
        r'\b(avenue|ave)\b': 'ave',
        r'\b(drive|dr)\b': 'dr',
    }
    for pattern, replacement in suffix_map.items():
        df['clean_name'] = df['clean_name'].str.replace(pattern, replacement, regex=True)
        df['clean_address'] = df['clean_address'].str.replace(pattern, replacement, regex=True)

    # Use category dtype for country (saves memory on large datasets)
    df['clean_country'] = df['clean_country'].astype('category')

    # Drop raw columns to save memory (optional, keeps only clean_ versions + entity_id)
    if drop_raw:
        for col in ['business_name', 'business_address', 'country']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    return df


# ==============================================================================
# STEP 2: CANDIDATE GENERATION (Country-Partitioned Sparse TF-IDF Retrieval)
# ==============================================================================

def _build_blocking_text(df):
    """
    Construct blocking text from clean columns for vectorization.
    Does not store it as a persistent column — used only during vectorizer transform.
    Returns a pandas Series of strings.
    """
    return (
        df['clean_name'].fillna('') + ' ' +
        df['clean_address'].fillna('') + ' ' +
        df['clean_country'].astype(str).fillna('')
    )


def generate_candidates_sparse(s1, pool, top_k=50, batch_size=5000, bm=None):
    """
    Country-partitioned, sparse TF-IDF candidate generation using
    HashingVectorizer with character n-grams.

    Architecture:
        For each country present in the data:
            1. Filter S1 and pool to that country
            2. Build HashingVectorizer + TF-IDF on pool partition
            3. Process S1 entities in batches of `batch_size`
            4. Sparse dot-product → top-K candidates per S1 entity
            5. Accumulate results

    Memory: ~2-4 GB peak per partition (largest partition is US ~6.2M records).
    Handles any country (open-set) without hard-coding.

    Args:
        s1: DataFrame with Source 1 entities (must have clean_name, clean_address, clean_country, entity_id)
        pool: DataFrame with Source 2 + Source 3 entities (same columns)
        top_k: Number of candidates to retrieve per S1 entity
        batch_size: Number of S1 entities to process at once
        bm: Optional BenchmarkCollector for memory snapshots

    Returns:
        candidates: dict mapping s1_id → list of candidate pool IDs
        blocking_scores: dict mapping (s1_id, pool_id) → cosine similarity score
    """
    candidates = {}
    blocking_scores = {}

    # Determine all countries present in either S1 or pool (open-set)
    s1_countries = set(s1['clean_country'].astype(str).unique())
    pool_countries = set(pool['clean_country'].astype(str).unique())
    all_countries = sorted(s1_countries | pool_countries)

    print(f"  -> Countries detected: {all_countries}")

    # The vectorizer: character n-grams (3,4) with hashing trick
    # n_features=2^18 (262,144) gives a good tradeoff between
    # hash collisions and memory. char_wb respects word boundaries.
    vectorizer = HashingVectorizer(
        n_features=2**18,
        analyzer='char_wb',
        ngram_range=(3, 4),
        alternate_sign=False,  # all positive weights (needed for TF-IDF)
        norm=None,             # we'll apply TF-IDF normalization separately
        dtype=np.float32,      # float32 halves memory vs float64
    )

    for country in all_countries:
        s1_mask = s1['clean_country'].astype(str) == country
        pool_mask = pool['clean_country'].astype(str) == country

        s1_partition = s1[s1_mask].reset_index(drop=True)
        pool_partition = pool[pool_mask].reset_index(drop=True)

        if len(s1_partition) == 0:
            print(f"  -> Country '{country}': 0 S1 entities, skipping.")
            continue

        if len(pool_partition) == 0:
            print(f"  -> Country '{country}': {len(s1_partition):,} S1 entities, 0 pool records — no candidates possible.")
            # All S1 entities in this country get empty candidate lists
            for s1_id in s1_partition['entity_id']:
                candidates[s1_id] = []
            continue

        print(f"\n  -> Country '{country}': {len(s1_partition):,} S1 entities, {len(pool_partition):,} pool records")

        # Step 1: Vectorize pool partition
        print(f"     Vectorizing pool ({len(pool_partition):,} records)...")
        pool_texts = _build_blocking_text(pool_partition)
        pool_hashed = vectorizer.transform(pool_texts)
        del pool_texts
        gc.collect()

        # Step 2: Apply TF-IDF weighting (fit on pool, transform both)
        print(f"     Fitting TF-IDF transformer...")
        tfidf = TfidfTransformer(sublinear_tf=True, use_idf=True)
        pool_tfidf = tfidf.fit_transform(pool_hashed)
        del pool_hashed
        gc.collect()

        # Normalize pool vectors for cosine similarity (dot product of normalized = cosine)
        # Row-normalize to unit L2 norm
        pool_tfidf = _l2_normalize_sparse(pool_tfidf)

        # Pool entity IDs as a numpy array for fast indexing
        pool_ids = pool_partition['entity_id'].values

        # Memory snapshot after index construction for this country
        if bm:
            bm.mem.update_peak()

        # Step 3: Process S1 in batches
        print(f"     Retrieving candidates (batch_size={batch_size})...")
        n_batches = (len(s1_partition) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(n_batches), desc=f"     {country} batches"):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(s1_partition))
            s1_batch = s1_partition.iloc[start:end]

            # Vectorize S1 batch
            s1_texts = _build_blocking_text(s1_batch)
            s1_hashed = vectorizer.transform(s1_texts)
            s1_tfidf = tfidf.transform(s1_hashed)
            s1_tfidf = _l2_normalize_sparse(s1_tfidf)
            del s1_hashed
            del s1_texts

            # Sparse dot product: (batch_size × features) @ (features × pool_size) = (batch_size × pool_size)
            # Result is sparse: only non-zero similarities are stored
            similarity_matrix = s1_tfidf @ pool_tfidf.T

            # Extract top-K candidates per S1 entity in this batch
            s1_batch_ids = s1_batch['entity_id'].values
            _extract_topk_from_sparse(
                similarity_matrix, s1_batch_ids, pool_ids,
                top_k, candidates, blocking_scores
            )

            del s1_tfidf
            del similarity_matrix

        # Free partition-specific memory before next country
        del pool_tfidf, pool_ids, tfidf
        del s1_partition, pool_partition
        gc.collect()
        print(f"     Done with country '{country}'.")

    # Ensure every S1 entity has an entry (even if empty)
    for s1_id in s1['entity_id']:
        if s1_id not in candidates:
            candidates[s1_id] = []

    total_candidates = sum(len(v) for v in candidates.values())
    avg_candidates = total_candidates / max(1, len(candidates))
    print(f"\n  -> Candidate generation complete: {len(candidates):,} S1 entities, "
          f"{total_candidates:,} total candidates, {avg_candidates:.1f} avg per entity")

    return candidates, blocking_scores


def _l2_normalize_sparse(matrix):
    """Row-wise L2 normalization of a sparse CSR matrix (in-place where possible)."""
    matrix = matrix.tocsr()
    # Compute L2 norm per row
    norms = sparse.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0  # avoid division by zero
    # Divide each row by its norm
    inv_norms = sparse.diags(1.0 / norms)
    return inv_norms @ matrix


def _extract_topk_from_sparse(sim_matrix, s1_ids, pool_ids, top_k, candidates, blocking_scores):
    """
    Given a sparse similarity matrix (n_s1 × n_pool), extract top-K
    pool candidates for each S1 entity and populate candidates/blocking_scores.

    Uses efficient CSR row slicing — no dense conversion.
    """
    sim_csr = sim_matrix.tocsr()

    for i in range(sim_csr.shape[0]):
        s1_id = s1_ids[i]

        # Get non-zero entries for this row
        row_start = sim_csr.indptr[i]
        row_end = sim_csr.indptr[i + 1]

        if row_start == row_end:
            # No non-zero similarities
            candidates[s1_id] = []
            continue

        col_indices = sim_csr.indices[row_start:row_end]
        scores = sim_csr.data[row_start:row_end]

        # Get top-K indices within this row's non-zero entries
        if len(scores) <= top_k:
            # All non-zero entries are candidates
            top_indices = np.arange(len(scores))
        else:
            # argpartition is O(n) vs O(n log n) for full sort
            top_indices = np.argpartition(scores, -top_k)[-top_k:]

        # Sort the top-K by score (descending) for consistent ordering
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        cand_ids = []
        for idx in top_indices:
            pool_id = pool_ids[col_indices[idx]]
            score = float(scores[idx])
            cand_ids.append(pool_id)
            blocking_scores[(s1_id, pool_id)] = score

        candidates[s1_id] = cand_ids


# ==============================================================================
# STEP 3: BLOCKING RECALL MEASUREMENT
# ==============================================================================
def measure_blocking_recall(candidates, gt_dict, s1_ids, label=""):
    """
    Measures what fraction of ground-truth matches appear in the candidate sets.
    This is the recall ceiling for the entire pipeline — any match not in
    candidates is permanently lost.

    Also reports per-entity recall distribution and identifies missed matches.
    """
    total_gt = 0
    total_found = 0
    entities_with_misses = 0
    entities_checked = 0

    for s1_id in s1_ids:
        gt_matches = gt_dict.get(s1_id, set())
        if len(gt_matches) == 0:
            continue  # singleton, no matches to find

        cand_set = set(candidates.get(s1_id, []))
        found = len(gt_matches & cand_set)
        total_gt += len(gt_matches)
        total_found += found
        entities_checked += 1

        if found < len(gt_matches):
            entities_with_misses += 1

    if total_gt == 0:
        print(f"  -> Blocking Recall {label}: N/A (no ground-truth matches)")
        return 1.0

    recall = total_found / total_gt
    print(f"  -> Blocking Recall {label}: {recall:.4f} "
          f"({total_found:,}/{total_gt:,} matches found in candidates)")
    print(f"     Entities with at least one missed match: "
          f"{entities_with_misses:,}/{entities_checked:,}")
    return recall


# ==============================================================================
# STEP 4: FEATURE EXTRACTION (Memory-Optimized)
# ==============================================================================
def extract_pair_features(row_s1, row_pool, tfidf_sim):
    """
    Computes fine-grained string distance & overlap features for a pair.
    """
    name1, name2 = row_s1['clean_name'], row_pool['clean_name']
    add1, add2 = row_s1['clean_address'], row_pool['clean_address']

    # Digit overlap (Jaccard similarity of extracted numbers)
    # Computed on-the-fly instead of storing as column
    dig1 = set(re.findall(r'\d+', add1))
    dig2 = set(re.findall(r'\d+', add2))
    if not dig1 and not dig2:
        digit_overlap = 1.0
    elif not dig1 or not dig2:
        digit_overlap = 0.0
    else:
        digit_overlap = len(dig1 & dig2) / len(dig1 | dig2)

    return {
        'name_jaro_winkler': distance.JaroWinkler.normalized_similarity(name1, name2),
        'name_levenshtein': fuzz.ratio(name1, name2) / 100.0,
        'name_token_set': fuzz.token_set_ratio(name1, name2) / 100.0,
        'name_token_sort': fuzz.token_sort_ratio(name1, name2) / 100.0,
        'add_token_set': fuzz.token_set_ratio(add1, add2) / 100.0,
        'add_levenshtein': fuzz.ratio(add1, add2) / 100.0,
        'digit_overlap': digit_overlap,
        'country_match': int(row_s1['clean_country'] == row_pool['clean_country']),
        'tfidf_sim': tfidf_sim,
    }


def build_feature_dataframe(s1_df, pool_df, candidates_dict, blocking_scores):
    """
    Converts candidates dict into a feature matrix for ML scoring.

    Memory-optimized: indexes into DataFrames using .loc instead of
    converting the entire pool to a Python dict (which would triple memory).
    """
    # Index both DataFrames by entity_id for fast lookup
    s1_indexed = s1_df.set_index('entity_id')
    pool_indexed = pool_df.set_index('entity_id')

    # Pre-collect only the candidate pool IDs we actually need
    needed_pool_ids = set()
    for cand_list in candidates_dict.values():
        needed_pool_ids.update(cand_list)

    # Create a small lookup dict of only the needed pool records
    # This is O(|candidates|) not O(|pool|)
    needed_pool_ids = list(needed_pool_ids)
    pool_subset = pool_indexed.loc[pool_indexed.index.isin(needed_pool_ids)]
    pool_lookup = pool_subset.to_dict('index')
    del pool_subset
    gc.collect()

    print(f"  -> Building features for {sum(len(v) for v in candidates_dict.values()):,} pairs "
          f"(pool lookup size: {len(pool_lookup):,})...")

    rows = []
    for s1_id, cand_list in tqdm(candidates_dict.items(), desc="Feature extraction"):
        if s1_id not in s1_indexed.index:
            continue
        row_s1 = s1_indexed.loc[s1_id]
        if isinstance(row_s1, pd.DataFrame):
            row_s1 = row_s1.iloc[0]
        row_s1 = row_s1.to_dict()

        for cand_id in cand_list:
            if cand_id not in pool_lookup:
                continue
            row_pool = pool_lookup[cand_id]
            sim = blocking_scores.get((s1_id, cand_id), 0.0)

            feats = extract_pair_features(row_s1, row_pool, sim)
            feats['source1_entity_id'] = s1_id
            feats['candidate_entity_id'] = cand_id
            rows.append(feats)

    return pd.DataFrame(rows)


# ==============================================================================
# STEP 5: MODEL TRAINING & F0.5 THRESHOLD OPTIMIZATION
# ==============================================================================
def compute_macro_f05(gt_dict, pred_dict, s1_entities):
    """
    Calculates Macro-Averaged F0.5 score across all S1 entities,
    handling singletons strictly per the competition metric.
    """
    scores = []
    for s1_id in s1_entities:
        gt_set = gt_dict.get(s1_id, set())
        pred_set = pred_dict.get(s1_id, set())

        if len(gt_set) == 0:
            # Singleton handling
            score = 1.0 if len(pred_set) == 0 else 0.0
        elif len(pred_set) == 0:
            score = 0.0
        else:
            tp = len(gt_set & pred_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if (0.25 * precision + recall) > 0:
                score = (1.25 * precision * recall) / (0.25 * precision + recall)
            else:
                score = 0.0
        scores.append(score)

    return np.mean(scores)


def train_and_optimize(train_features_df, gt_dict, s1_train_ids, s1_val_ids, bm=None):
    """Trains XGBoost and finds the probability threshold that maximizes F0.5.

    Accepts an optional BenchmarkCollector (bm) for timing sub-stages.
    """
    # Label pair samples
    train_features_df['is_match'] = train_features_df.apply(
        lambda x: 1 if x['candidate_entity_id'] in gt_dict.get(x['source1_entity_id'], set()) else 0,
        axis=1
    )

    feature_cols = [c for c in train_features_df.columns
                    if c not in ['source1_entity_id', 'candidate_entity_id', 'is_match']]

    train_mask = train_features_df['source1_entity_id'].isin(s1_train_ids)
    val_mask = train_features_df['source1_entity_id'].isin(s1_val_ids)

    X_train = train_features_df.loc[train_mask, feature_cols]
    y_train = train_features_df.loc[train_mask, 'is_match']
    val_df = train_features_df.loc[val_mask].copy()

    print(f"  -> Training data: {len(X_train):,} pairs ({y_train.sum():,} positives)")
    print(f"  -> Validation data: {len(val_df):,} pairs")

    # ---------- XGBoost Training ----------
    print("  -> Training XGBoost Classifier...")
    with Timer("xgb_train") as t_train:
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train)
    if bm:
        bm.set_timer("model_training", t_train.elapsed)
        bm.mem.snapshot("after_xgb_train")

    # ---------- Validation / Threshold Optimization ----------
    print("  -> Optimizing F0.5 Macro Threshold on Validation Set...")
    with Timer("threshold_opt") as t_thresh:
        val_df['prob'] = model.predict_proba(val_df[feature_cols])[:, 1]

        best_f05 = -1.0
        best_threshold = 0.70

        # Search thresholds between 0.30 and 0.96
        for thresh in np.arange(0.30, 0.96, 0.02):
            pred_dict = {}
            filtered_val = val_df[val_df['prob'] >= thresh]

            for s1_id, group in filtered_val.groupby('source1_entity_id'):
                pred_dict[s1_id] = set(group['candidate_entity_id'])

            score = compute_macro_f05(gt_dict, pred_dict, s1_val_ids)
            if score > best_f05:
                best_f05 = score
                best_threshold = thresh

    if bm:
        bm.set_timer("threshold_optimization", t_thresh.elapsed)

    print(f"  -> Optimal Threshold: {best_threshold:.2f} | Validation Macro F0.5: {best_f05:.4f}")

    # ---- Compute model quality on validation split (for benchmark) ----
    if bm:
        # Build pred_dict at optimal threshold
        val_pred_dict: dict = {}
        filtered_val = val_df[val_df['prob'] >= best_threshold]
        for s1_id, group in filtered_val.groupby('source1_entity_id'):
            val_pred_dict[s1_id] = set(group['candidate_entity_id'])

        bm.model_quality = compute_model_quality(
            val_pred_dict, gt_dict, list(s1_val_ids)
        )

    # ---------- Retrain on full data ----------
    print("  -> Retraining model on full training data...")
    with Timer("retrain_full") as t_retrain:
        X_full = train_features_df[feature_cols]
        y_full = train_features_df['is_match']
        model.fit(X_full, y_full)
    if bm:
        bm.set_timer("model_retrain_full", t_retrain.elapsed)

    return model, best_threshold, feature_cols


# ==============================================================================
# MAIN EXECUTION PIPELINE — INSTRUMENTED
# ==============================================================================
def main():
    TRAIN_DIR = "../Dataset/student_resource/dataset/train"
    TEST_DIR = "../Dataset/student_resource/dataset/test"
    OUTPUT_DIR = "output"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║ Initialize Benchmark Collector                                  ║
    # ╚══════════════════════════════════════════════════════════════════╝
    bm = BenchmarkCollector(
        version=PIPELINE_VERSION,
        top_k=TOP_K,
        batch_size=BATCH_SIZE,
        xgb_params=XGB_PARAMS,
    )
    bm.mem.snapshot("pipeline_start")

    train_pipeline_start = time.perf_counter()

    # ---------------------------------------------------------
    # 1. LOAD & NORMALIZE TRAINING DATA
    # ---------------------------------------------------------
    print("\n--- 1. Processing Training Data ---")

    # -- Load + normalize S1 --
    with Timer() as t_load_s1:
        raw_s1 = pd.read_csv(f"{TRAIN_DIR}/train_source1.tsv", sep="\t")
    bm.set_timer("load_train_s1", t_load_s1.elapsed)

    with Timer() as t_norm_s1:
        train_s1 = normalize_dataset(raw_s1)
    bm.set_timer("norm_train_s1", t_norm_s1.elapsed)
    del raw_s1
    print(f"  -> S1: {len(train_s1):,} rows")

    # -- Load + normalize S2 --
    print("  -> Loading and normalizing S2...")
    with Timer() as t_load_s2:
        raw_s2 = pd.read_csv(f"{TRAIN_DIR}/train_source2.tsv", sep="\t")
    bm.set_timer("load_train_s2", t_load_s2.elapsed)

    with Timer() as t_norm_s2:
        train_s2 = normalize_dataset(raw_s2)
    bm.set_timer("norm_train_s2", t_norm_s2.elapsed)
    del raw_s2
    print(f"     S2: {len(train_s2):,} rows")

    # -- Load + normalize S3 --
    print("  -> Loading and normalizing S3...")
    with Timer() as t_load_s3:
        raw_s3 = pd.read_csv(f"{TRAIN_DIR}/train_source3.tsv", sep="\t")
    bm.set_timer("load_train_s3", t_load_s3.elapsed)

    with Timer() as t_norm_s3:
        train_s3 = normalize_dataset(raw_s3)
    bm.set_timer("norm_train_s3", t_norm_s3.elapsed)
    del raw_s3
    print(f"     S3: {len(train_s3):,} rows")

    # Totals
    bm.set_timer("load_train_total", t_load_s1.elapsed + t_load_s2.elapsed + t_load_s3.elapsed)
    bm.set_timer("norm_train_total", t_norm_s1.elapsed + t_norm_s2.elapsed + t_norm_s3.elapsed)

    bm.mem.snapshot("after_loading_train")

    # Throughput: normalization
    total_norm_records = len(train_s1) + len(train_s2) + len(train_s3)
    norm_total_time = t_norm_s1.elapsed + t_norm_s2.elapsed + t_norm_s3.elapsed
    bm.add_throughput("normalization_train_records_per_sec", total_norm_records, norm_total_time)

    # -- Pool construction --
    with Timer() as t_pool:
        train_pool = pd.concat([train_s2, train_s3], ignore_index=True)
    bm.set_timer("pool_construction_train", t_pool.elapsed)
    bm.add_count("train_pool_size", len(train_pool))

    # Dataset stats (computed from existing DataFrames — no copies)
    bm.dataset_stats = compute_dataset_stats(train_s1, train_s2, train_s3, train_pool)

    del train_s2, train_s3
    gc.collect()
    print(f"  -> Pool: {len(train_pool):,} rows")

    bm.mem.snapshot("after_normalization_train")

    # Load Ground Truth (vectorized — avoids slow iterrows over 2.2M rows)
    gt_df = pd.read_csv(f"{TRAIN_DIR}/train_ground_truth.tsv", sep="\t")
    gt_dict = {}
    # Vectorized: process non-null matched_entity_ids
    has_matches = gt_df['matched_entity_ids'].notna() & (gt_df['matched_entity_ids'] != '')
    for s1_id in gt_df.loc[~has_matches, 'source1_entity_id']:
        gt_dict[s1_id] = set()
    for s1_id, matched in zip(
        gt_df.loc[has_matches, 'source1_entity_id'],
        gt_df.loc[has_matches, 'matched_entity_ids']
    ):
        gt_dict[s1_id] = set(m.strip() for m in str(matched).split(',') if m.strip())
    del gt_df, has_matches
    gc.collect()
    print(f"  -> Ground truth loaded: {len(gt_dict):,} S1 entities")

    # ---------------------------------------------------------
    # 2. TRAIN CANDIDATE GENERATION
    # ---------------------------------------------------------
    print("\n--- 2. Generating Training Candidates ---")
    bm.mem.snapshot("before_candidate_gen_train")

    with Timer() as t_cand_train:
        train_candidates, train_blocking_scores = generate_candidates_sparse(
            train_s1, train_pool, top_k=TOP_K, batch_size=BATCH_SIZE, bm=bm
        )
    bm.set_timer("candidate_gen_train", t_cand_train.elapsed)
    bm.mem.snapshot("after_candidate_gen_train")

    # Candidate generation throughput
    total_cand_pairs = sum(len(v) for v in train_candidates.values())
    bm.add_count("train_candidate_pairs", total_cand_pairs)
    bm.add_throughput("candidate_gen_s1_entities_per_sec", len(train_candidates), t_cand_train.elapsed)
    bm.add_throughput("candidate_gen_pairs_per_sec", total_cand_pairs, t_cand_train.elapsed)
    bm.add_throughput("candidate_gen_pool_records_per_sec", len(train_pool), t_cand_train.elapsed)

    # ---------------------------------------------------------
    # 3. MEASURE BLOCKING RECALL
    # ---------------------------------------------------------
    print("\n--- 3. Measuring Blocking Recall ---")
    with Timer() as t_br:
        all_s1_ids = train_s1['entity_id'].unique()
        blocking_recall = measure_blocking_recall(
            train_candidates, gt_dict, all_s1_ids, label="(all training data)"
        )
    bm.set_timer("blocking_recall_train", t_br.elapsed)

    if blocking_recall < 0.90:
        print(f"  !! WARNING: Blocking recall is {blocking_recall:.4f} — consider increasing top_k (currently {TOP_K})")
    elif blocking_recall < 0.95:
        print(f"  -> Blocking recall is {blocking_recall:.4f} — acceptable but could be improved")
    else:
        print(f"  -> Blocking recall is {blocking_recall:.4f} — good!")

    # Candidate quality metrics (re-uses existing candidates + gt_dict)
    bm.candidate_quality = compute_candidate_quality(
        train_candidates, gt_dict, pool_size=len(train_pool)
    )

    # Blocking efficiency (we don't have explicit posting-list data in
    # the sparse TF-IDF approach, but we can report top-K pruning stats)
    bm.blocking_efficiency = compute_blocking_efficiency(
        train_candidates,
        n_before_topk=None,  # not tracked in current impl
        n_after_topk=total_cand_pairs,
    )

    # ---------------------------------------------------------
    # 4. FEATURE EXTRACTION
    # ---------------------------------------------------------
    print("\n--- 4. Extracting Pair Features ---")
    bm.mem.snapshot("before_feature_gen_train")

    with Timer() as t_feat_train:
        train_feat_df = build_feature_dataframe(
            train_s1, train_pool, train_candidates, train_blocking_scores
        )
    bm.set_timer("feature_gen_train", t_feat_train.elapsed)
    bm.mem.snapshot("after_feature_gen_train")

    bm.add_count("train_feature_pairs", len(train_feat_df))
    bm.add_throughput("feature_gen_pairs_per_sec", len(train_feat_df), t_feat_train.elapsed)

    # Free blocking scores (can be large)
    del train_blocking_scores
    gc.collect()

    # Train / Validation Entity Split (80/20 on S1 entities)
    with Timer() as t_split:
        s1_train_ids, s1_val_ids = train_test_split(all_s1_ids, test_size=0.20, random_state=42)
    bm.set_timer("train_val_split", t_split.elapsed)

    # Measure blocking recall on validation split specifically
    measure_blocking_recall(
        train_candidates, gt_dict, s1_val_ids, label="(validation split)"
    )
    del train_candidates
    gc.collect()

    # ---------------------------------------------------------
    # 5. TRAIN CLASSIFIER & OPTIMIZE THRESHOLD
    # ---------------------------------------------------------
    print("\n--- 5. Training Model & Optimizing F0.5 Threshold ---")
    bm.mem.snapshot("before_ml")

    with Timer() as t_ml:
        model, best_threshold, feature_cols = train_and_optimize(
            train_feat_df, gt_dict, s1_train_ids, s1_val_ids, bm=bm
        )
    bm.set_timer("ml_total", t_ml.elapsed)
    bm.mem.snapshot("after_ml")

    bm.config["best_threshold"] = best_threshold

    del train_feat_df, gt_dict
    gc.collect()

    # Record total training pipeline time
    train_pipeline_elapsed = time.perf_counter() - train_pipeline_start
    bm.set_timer("train_pipeline", train_pipeline_elapsed)

    # ---------------------------------------------------------
    # 6. FREE TRAINING DATA, LOAD & NORMALIZE TEST DATA
    # ---------------------------------------------------------
    print("\n--- 6. Processing Test Data ---")
    del train_s1, train_pool
    gc.collect()

    test_pipeline_start = time.perf_counter()

    # -- Load + normalize test S1 --
    with Timer() as t_load_ts1:
        raw_ts1 = pd.read_csv(f"{TEST_DIR}/test_source1.tsv", sep="\t")
    bm.set_timer("load_test_s1", t_load_ts1.elapsed)
    with Timer() as t_norm_ts1:
        test_s1 = normalize_dataset(raw_ts1)
    bm.set_timer("norm_test_s1", t_norm_ts1.elapsed)
    del raw_ts1
    print(f"  -> Test S1: {len(test_s1):,} rows")

    # -- Load + normalize test S2 --
    print("  -> Loading and normalizing test S2...")
    with Timer() as t_load_ts2:
        raw_ts2 = pd.read_csv(f"{TEST_DIR}/test_source2.tsv", sep="\t")
    bm.set_timer("load_test_s2", t_load_ts2.elapsed)
    with Timer() as t_norm_ts2:
        test_s2 = normalize_dataset(raw_ts2)
    bm.set_timer("norm_test_s2", t_norm_ts2.elapsed)
    del raw_ts2
    print(f"     Test S2: {len(test_s2):,} rows")

    # -- Load + normalize test S3 --
    print("  -> Loading and normalizing test S3...")
    with Timer() as t_load_ts3:
        raw_ts3 = pd.read_csv(f"{TEST_DIR}/test_source3.tsv", sep="\t")
    bm.set_timer("load_test_s3", t_load_ts3.elapsed)
    with Timer() as t_norm_ts3:
        test_s3 = normalize_dataset(raw_ts3)
    bm.set_timer("norm_test_s3", t_norm_ts3.elapsed)
    del raw_ts3
    print(f"     Test S3: {len(test_s3):,} rows")

    bm.set_timer("load_test_total", t_load_ts1.elapsed + t_load_ts2.elapsed + t_load_ts3.elapsed)
    bm.set_timer("norm_test_total", t_norm_ts1.elapsed + t_norm_ts2.elapsed + t_norm_ts3.elapsed)

    bm.mem.snapshot("after_loading_test")

    # -- Pool construction --
    with Timer() as t_pool_test:
        test_pool = pd.concat([test_s2, test_s3], ignore_index=True)
    bm.set_timer("pool_construction_test", t_pool_test.elapsed)
    bm.add_count("test_pool_size", len(test_pool))

    del test_s2, test_s3
    gc.collect()
    print(f"  -> Test Pool: {len(test_pool):,} rows")

    # ---------------------------------------------------------
    # 7. TEST CANDIDATE GENERATION
    # ---------------------------------------------------------
    print("\n--- 7. Generating Test Candidates ---")
    bm.mem.snapshot("before_candidate_gen_test")

    with Timer() as t_cand_test:
        test_candidates, test_blocking_scores = generate_candidates_sparse(
            test_s1, test_pool, top_k=TOP_K, batch_size=BATCH_SIZE, bm=bm
        )
    bm.set_timer("candidate_gen_test", t_cand_test.elapsed)
    bm.mem.snapshot("after_candidate_gen_test")

    test_cand_pairs = sum(len(v) for v in test_candidates.values())
    bm.add_count("test_candidate_pairs", test_cand_pairs)

    # Write output/candidate_pairs.tsv
    with Timer() as t_write_cand:
        cand_rows = []
        for s1_id in test_s1['entity_id']:
            cands = test_candidates.get(s1_id, [])
            cand_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_ids": ",".join(cands),
            })
        pd.DataFrame(cand_rows).to_csv(f"{OUTPUT_DIR}/candidate_pairs.tsv", sep="\t", index=False)
    bm.set_timer("write_candidate_pairs", t_write_cand.elapsed)
    print(f"  -> Saved {OUTPUT_DIR}/candidate_pairs.tsv")
    del cand_rows
    gc.collect()

    # ---------------------------------------------------------
    # 8. TEST FEATURE EXTRACTION & INFERENCE
    # ---------------------------------------------------------
    print("\n--- 8. Extracting Test Features & Predicting ---")
    bm.mem.snapshot("before_feature_gen_test")

    with Timer() as t_feat_test:
        test_feat_df = build_feature_dataframe(
            test_s1, test_pool, test_candidates, test_blocking_scores
        )
    bm.set_timer("feature_gen_test", t_feat_test.elapsed)
    bm.mem.snapshot("after_feature_gen_test")

    bm.add_count("test_feature_pairs", len(test_feat_df))
    bm.add_throughput("feature_gen_test_pairs_per_sec", len(test_feat_df), t_feat_test.elapsed)

    del test_blocking_scores
    gc.collect()

    match_dict = {s1_id: [] for s1_id in test_s1['entity_id']}
    with Timer() as t_pred:
        if not test_feat_df.empty:
            test_feat_df['prob'] = model.predict_proba(test_feat_df[feature_cols])[:, 1]

            # Filter matches passing the optimal F0.5 threshold
            matched_pairs = test_feat_df[test_feat_df['prob'] >= best_threshold]
            for s1_id, group in matched_pairs.groupby('source1_entity_id'):
                match_dict[s1_id] = list(group['candidate_entity_id'])
    bm.set_timer("prediction_test", t_pred.elapsed)
    bm.add_throughput("prediction_pairs_per_sec", len(test_feat_df), t_pred.elapsed)

    # ---------------------------------------------------------
    # 9. VALIDATE & WRITE FINAL RESULTS
    # ---------------------------------------------------------
    print("\n--- 9. Writing Final Results ---")

    # Validate: every final match must appear in candidates
    validation_errors = 0
    for s1_id, matches in match_dict.items():
        cand_set = set(test_candidates.get(s1_id, []))
        for m in matches:
            if m not in cand_set:
                validation_errors += 1
                print(f"  !! ERROR: {s1_id} match {m} NOT in candidates!")
    if validation_errors == 0:
        print("  -> Validation passed: all final matches are in candidate sets.")
    else:
        print(f"  !! {validation_errors} validation errors found!")

    # Write output/matching_results.tsv
    with Timer() as t_write_match:
        match_rows = []
        for s1_id in test_s1['entity_id']:
            matches = match_dict.get(s1_id, [])
            match_rows.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(matches),
            })
        pd.DataFrame(match_rows).to_csv(f"{OUTPUT_DIR}/matching_results.tsv", sep="\t", index=False)
    bm.set_timer("write_matching_results", t_write_match.elapsed)
    bm.set_timer("output_total", t_write_cand.elapsed + t_write_match.elapsed)
    print(f"  -> Saved {OUTPUT_DIR}/matching_results.tsv")

    # Summary stats
    total_matches = sum(len(v) for v in match_dict.values())
    entities_with_matches = sum(1 for v in match_dict.values() if len(v) > 0)
    print(f"\n  -> Summary:")
    print(f"     Test S1 entities: {len(test_s1):,}")
    print(f"     Entities with matches: {entities_with_matches:,}")
    print(f"     Total matches: {total_matches:,}")
    print(f"     Threshold used: {best_threshold:.2f}")

    bm.add_count("test_total_matches", total_matches)
    bm.add_count("test_entities_with_matches", entities_with_matches)

    # Record total test pipeline time
    test_pipeline_elapsed = time.perf_counter() - test_pipeline_start
    bm.set_timer("test_pipeline", test_pipeline_elapsed)

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║ FINALIZE & SAVE BENCHMARK                                      ║
    # ╚══════════════════════════════════════════════════════════════════╝
    bm.mem.snapshot("before_cleanup")
    gc.collect()
    bm.mem.snapshot("after_cleanup")

    bm.finalize()
    bm.save(OUTPUT_DIR)

    # Print the text report to console
    txt_path = os.path.join(OUTPUT_DIR, "benchmark_report.txt")
    with open(txt_path, "r", encoding="utf-8") as f:
        print(f.read())

    print("\n=== PIPELINE RUN COMPLETE SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
