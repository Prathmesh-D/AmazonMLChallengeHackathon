"""
Entity Resolution Pipeline — Benchmarking & Instrumentation Module
===================================================================

Provides lightweight, non-invasive instrumentation for measuring:
  - High-resolution per-stage timing
  - RSS memory at key checkpoints (via psutil)
  - Candidate-set quality (recall, reduction ratio, size distribution)
  - ML model quality (precision, recall, F1, F0.5)
  - Dataset statistics
  - Per-stage throughput
  - Structured report generation (JSON / TXT / CSV history)
  - Baseline comparison utility

Design principles:
  1. Monotonic time.perf_counter() for all timings
  2. psutil.Process().memory_info().rss for memory — no large arrays
  3. Incremental counters, streaming aggregation where possible
  4. Re-use existing structures (gt_dict, candidates) for recall — no copies
"""

import time
import json
import csv
import os
import sys
import platform
import datetime
import uuid
from collections import defaultdict
from typing import Dict, Set, List, Optional, Any

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependency: psutil  (graceful fallback)
# ---------------------------------------------------------------------------
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 1. TIMER CONTEXT MANAGER                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class Timer:
    """Lightweight monotonic timer with context-manager interface."""

    def __init__(self, label: str = ""):
        self.label = label
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start_time

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self.start_time
        return self.elapsed

    def read(self) -> float:
        """Return elapsed since start without stopping."""
        return time.perf_counter() - self.start_time


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 2. MEMORY TRACKER                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class MemoryTracker:
    """
    Lightweight RSS memory tracker using psutil.

    Stores only named checkpoints (label → rss_bytes) and a running peak.
    Does NOT accumulate per-record measurements.
    """

    def __init__(self):
        if _HAS_PSUTIL:
            self._process = psutil.Process(os.getpid())
            self._total_ram = psutil.virtual_memory().total
        else:
            self._process = None
            self._total_ram = 0
        self.checkpoints: Dict[str, int] = {}
        self.peak_rss: int = 0

    def snapshot(self, label: str) -> int:
        """Record current RSS under *label* and update peak."""
        rss = self._get_rss()
        self.checkpoints[label] = rss
        if rss > self.peak_rss:
            self.peak_rss = rss
        return rss

    def update_peak(self) -> int:
        """Update peak without storing a labelled checkpoint."""
        rss = self._get_rss()
        if rss > self.peak_rss:
            self.peak_rss = rss
        return rss

    @property
    def total_system_ram(self) -> int:
        return self._total_ram

    # --- helpers ---

    def _get_rss(self) -> int:
        if self._process:
            return self._process.memory_info().rss
        return 0

    @staticmethod
    def bytes_to_gb(b: int) -> float:
        return b / (1024 ** 3)

    def report_dict(self) -> dict:
        """Return a JSON-friendly dict of checkpoints + peak."""
        d = {}
        for label, rss in self.checkpoints.items():
            d[label] = {
                "rss_bytes": rss,
                "rss_gb": round(self.bytes_to_gb(rss), 3),
            }
        d["_peak_rss"] = {
            "rss_bytes": self.peak_rss,
            "rss_gb": round(self.bytes_to_gb(self.peak_rss), 3),
        }
        if self._total_ram:
            d["_system_ram_gb"] = round(self.bytes_to_gb(self._total_ram), 3)
            d["_peak_pct_system"] = round(
                100.0 * self.peak_rss / self._total_ram, 2
            ) if self._total_ram else None
        return d


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 3. CANDIDATE QUALITY METRICS                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def candidate_recall_at_k(
    candidates: Dict[str, list],
    gt_dict: Dict[str, Set[str]],
    k: int,
) -> float:
    """
    Compute candidate recall when only the top-*k* candidates per entity
    are considered.  Re-uses existing structures — no copies.
    """
    total_gt = 0
    found = 0
    for s1_id, gt_set in gt_dict.items():
        if not gt_set:
            continue
        cand_list = candidates.get(s1_id, [])
        cand_at_k = set(cand_list[:k])
        total_gt += len(gt_set)
        found += len(gt_set & cand_at_k)
    return found / total_gt if total_gt else 1.0


def compute_candidate_quality(
    candidates: Dict[str, list],
    gt_dict: Dict[str, Set[str]],
    pool_size: int,
) -> dict:
    """
    Comprehensive candidate-set quality report.

    Computes:
      - Overall recall, recall@K for several K values
      - Candidate-set size statistics (mean, median, P90–P99, min, max)
      - Reduction ratio
      - Per-entity recall breakdown (full / partial / missed)
    """
    # --- recall ---
    total_gt = 0
    total_found = 0
    entities_full_recall = 0
    entities_partial_recall = 0
    entities_zero_recall = 0
    entities_checked = 0

    for s1_id, gt_set in gt_dict.items():
        if not gt_set:
            continue
        cand_set = set(candidates.get(s1_id, []))
        hit = len(gt_set & cand_set)
        total_gt += len(gt_set)
        total_found += hit
        entities_checked += 1
        if hit == len(gt_set):
            entities_full_recall += 1
        elif hit > 0:
            entities_partial_recall += 1
        else:
            entities_zero_recall += 1

    overall_recall = total_found / total_gt if total_gt else 1.0

    recall_at = {}
    for k in [10, 20, 30, 50, 100]:
        recall_at[k] = candidate_recall_at_k(candidates, gt_dict, k)

    # --- size distribution ---
    sizes = np.array([len(v) for v in candidates.values()], dtype=np.int64)
    total_pairs = int(sizes.sum())
    n_s1 = len(candidates)

    size_stats = {
        "total_pairs": total_pairs,
        "avg": round(float(sizes.mean()), 2) if len(sizes) else 0,
        "median": int(np.median(sizes)) if len(sizes) else 0,
        "min": int(sizes.min()) if len(sizes) else 0,
        "max": int(sizes.max()) if len(sizes) else 0,
        "p90": int(np.percentile(sizes, 90)) if len(sizes) else 0,
        "p95": int(np.percentile(sizes, 95)) if len(sizes) else 0,
        "p99": int(np.percentile(sizes, 99)) if len(sizes) else 0,
    }

    # --- reduction ratio ---
    total_possible = n_s1 * pool_size
    reduction_ratio = (
        1.0 - (total_pairs / total_possible)
    ) if total_possible > 0 else 0.0

    return {
        "overall_recall": round(overall_recall, 6),
        "recall_at_k": {str(k): round(v, 6) for k, v in recall_at.items()},
        "entities_checked": entities_checked,
        "entities_full_recall": entities_full_recall,
        "entities_partial_recall": entities_partial_recall,
        "entities_zero_recall": entities_zero_recall,
        "total_gt_matches": total_gt,
        "total_found_in_candidates": total_found,
        "size_stats": size_stats,
        "total_possible_pairs": total_possible,
        "reduction_ratio": round(reduction_ratio, 8),
        "reduction_ratio_pct": f"{reduction_ratio * 100:.4f}%",
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 4. MODEL / MATCHING QUALITY METRICS                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def compute_model_quality(
    pred_dict: Dict[str, Set[str]],
    gt_dict: Dict[str, Set[str]],
    s1_ids: list,
) -> dict:
    """
    Compute matching quality metrics appropriate for many-to-many
    entity resolution.

    Returns micro-aggregated precision/recall/F1/F0.5 plus entity counts.
    Also returns macro-averaged F0.5 (the competition metric).
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    n_predicted_match = 0
    n_no_prediction = 0

    # For macro F0.5
    per_entity_f05 = []

    for s1_id in s1_ids:
        gt_set = gt_dict.get(s1_id, set())
        pred_set = pred_dict.get(s1_id, set())

        tp = len(gt_set & pred_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        if len(pred_set) > 0:
            n_predicted_match += 1
        else:
            n_no_prediction += 1

        # Per-entity F0.5 (competition rules: singleton with no pred → 1.0)
        if len(gt_set) == 0:
            score = 1.0 if len(pred_set) == 0 else 0.0
        elif len(pred_set) == 0:
            score = 0.0
        else:
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) > 0 else 0.0
        per_entity_f05.append(score)

    # Micro
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0.0

    macro_f05 = float(np.mean(per_entity_f05)) if per_entity_f05 else 0.0

    return {
        "micro_precision": round(precision, 6),
        "micro_recall": round(recall, 6),
        "micro_f1": round(f1, 6),
        "micro_f05": round(f05, 6),
        "macro_f05": round(macro_f05, 6),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "predicted_matches": total_tp + total_fp,
        "entities_with_prediction": n_predicted_match,
        "entities_no_prediction": n_no_prediction,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 5. DATASET STATISTICS                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def compute_dataset_stats(s1_df, s2_df, s3_df, pool_df) -> dict:
    """
    Collect lightweight dataset statistics.
    Operates on existing DataFrame columns — no copies.
    """
    import re as _re

    stats: Dict[str, Any] = {
        "n_source1": len(s1_df),
        "n_source2": len(s2_df) if s2_df is not None else None,
        "n_source3": len(s3_df) if s3_df is not None else None,
        "n_pool": len(pool_df),
    }

    # Unique countries across all sources
    all_countries: set = set()
    for df in [s1_df, pool_df]:
        col = 'clean_country' if 'clean_country' in df.columns else 'country'
        if col in df.columns:
            all_countries.update(df[col].astype(str).unique())
    stats["n_unique_countries"] = len(all_countries)
    stats["countries"] = sorted(all_countries)

    # Missing-value counts (use clean_ columns if available, else raw)
    for label, df in [("source1", s1_df), ("pool", pool_df)]:
        name_col = 'clean_name' if 'clean_name' in df.columns else 'business_name'
        addr_col = 'clean_address' if 'clean_address' in df.columns else 'business_address'
        ctry_col = 'clean_country' if 'clean_country' in df.columns else 'country'

        stats[f"{label}_missing_name"] = int(
            (df[name_col].isna() | (df[name_col].astype(str).str.strip() == '')).sum()
        ) if name_col in df.columns else None
        stats[f"{label}_missing_address"] = int(
            (df[addr_col].isna() | (df[addr_col].astype(str).str.strip() == '')).sum()
        ) if addr_col in df.columns else None
        stats[f"{label}_missing_country"] = int(
            (df[ctry_col].isna() | (df[ctry_col].astype(str).str.strip() == '')).sum()
        ) if ctry_col in df.columns else None

    # Normalized-field statistics (streaming — no copies)
    for label, df in [("source1", s1_df), ("pool", pool_df)]:
        name_col = 'clean_name' if 'clean_name' in df.columns else None
        addr_col = 'clean_address' if 'clean_address' in df.columns else None

        if name_col and name_col in df.columns:
            name_lengths = df[name_col].astype(str).str.len()
            name_tokens = df[name_col].astype(str).str.split().str.len()
            stats[f"{label}_avg_name_len"] = round(float(name_lengths.mean()), 2)
            stats[f"{label}_avg_name_tokens"] = round(float(name_tokens.mean()), 2)
            stats[f"{label}_unique_names"] = int(df[name_col].nunique())

        if addr_col and addr_col in df.columns:
            addr_lengths = df[addr_col].astype(str).str.len()
            stats[f"{label}_avg_addr_len"] = round(float(addr_lengths.mean()), 2)

    return stats


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 6. BLOCKING EFFICIENCY METRICS                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def compute_blocking_efficiency(
    candidates: Dict[str, list],
    n_before_topk: Optional[int] = None,
    n_after_topk: Optional[int] = None,
    blocking_key_counts: Optional[Dict[str, int]] = None,
) -> dict:
    """
    Blocking-specific efficiency metrics.

    *blocking_key_counts*: if the pipeline tracks how many pool records
    each blocking key maps to (posting-list sizes), pass them here.
    Otherwise this computes only candidate-level stats.
    """
    result: Dict[str, Any] = {}

    # Posting-list stats (if available)
    if blocking_key_counts:
        sizes = np.array(list(blocking_key_counts.values()), dtype=np.int64)
        result["n_unique_keys"] = len(sizes)
        result["avg_posting_list_size"] = round(float(sizes.mean()), 2)
        result["median_posting_list_size"] = int(np.median(sizes))
        result["p95_posting_list_size"] = int(np.percentile(sizes, 95))
        result["p99_posting_list_size"] = int(np.percentile(sizes, 99))
        result["max_posting_list_size"] = int(sizes.max())

    # Top-K pruning
    if n_before_topk is not None and n_after_topk is not None:
        result["candidates_before_topk"] = n_before_topk
        result["candidates_after_topk"] = n_after_topk
        removed = n_before_topk - n_after_topk
        result["candidates_removed_by_topk"] = removed
        result["topk_pruning_pct"] = (
            round(100.0 * removed / n_before_topk, 4)
            if n_before_topk > 0 else 0.0
        )

    return result


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 7. THROUGHPUT CALCULATOR                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def throughput(count: int, elapsed: float) -> float:
    """Return items/sec, guarding against division by zero."""
    return count / elapsed if elapsed > 0 else 0.0


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 8. SYSTEM / ENVIRONMENT INFO                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def system_info() -> dict:
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "os": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    if _HAS_PSUTIL:
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["total_ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    return info


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 9. BENCHMARK COLLECTOR                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class BenchmarkCollector:
    """
    Central object that accumulates all benchmark metrics for a single run.

    Usage:
        bm = BenchmarkCollector(version="baseline_v1")
        bm.set_timer("data_loading_s1", elapsed_seconds)
        bm.mem.snapshot("after_loading")
        ...
        bm.finalize()
        bm.save("output")
    """

    def __init__(self, version: str = "baseline_v1", top_k: int = 50,
                 batch_size: int = 5000, xgb_params: Optional[dict] = None):
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self.version = version
        self.timestamp = datetime.datetime.now().isoformat()
        self.system = system_info()

        # Algorithm config
        self.config = {
            "pipeline_version": version,
            "top_k": top_k,
            "batch_size": batch_size,
            "xgb_params": xgb_params or {},
        }

        # Sub-collectors
        self.mem = MemoryTracker()
        self.timers: Dict[str, float] = {}
        self.counts: Dict[str, Any] = {}

        # Will be filled by pipeline
        self.dataset_stats: dict = {}
        self.candidate_quality: dict = {}
        self.model_quality: dict = {}
        self.blocking_efficiency: dict = {}
        self.throughputs: Dict[str, float] = {}

        # Overall wall-clock
        self._pipeline_start = time.perf_counter()

    # ---- timer helpers ----

    def set_timer(self, label: str, elapsed: float):
        self.timers[label] = round(elapsed, 6)

    def add_count(self, label: str, value):
        self.counts[label] = value

    def add_throughput(self, label: str, count: int, elapsed: float):
        self.throughputs[label] = round(throughput(count, elapsed), 1)

    # ---- finalize ----

    def finalize(self):
        """Compute derived metrics and record end-to-end time."""
        total_elapsed = time.perf_counter() - self._pipeline_start
        self.timers["end_to_end"] = round(total_elapsed, 6)
        self.mem.snapshot("final")

    # ---- output ----

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "version": self.version,
            "timestamp": self.timestamp,
            "system": self.system,
            "config": self.config,
            "timers": self.timers,
            "counts": self.counts,
            "memory": self.mem.report_dict(),
            "dataset_stats": self.dataset_stats,
            "candidate_quality": self.candidate_quality,
            "model_quality": self.model_quality,
            "blocking_efficiency": self.blocking_efficiency,
            "throughputs": self.throughputs,
        }

    # ---- save to files ----

    def save(self, output_dir: str = "output"):
        os.makedirs(output_dir, exist_ok=True)

        # 1. JSON
        json_path = os.path.join(output_dir, "benchmark_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"  -> Benchmark JSON saved: {json_path}")

        # 2. Human-readable TXT
        txt_path = os.path.join(output_dir, "benchmark_report.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(self._format_text_report())
        print(f"  -> Benchmark report saved: {txt_path}")

        # 3. CSV history (append)
        csv_path = os.path.join(output_dir, "benchmark_history.csv")
        self._append_csv_history(csv_path)
        print(f"  -> Benchmark history updated: {csv_path}")

    # ---- text report ----

    def _format_text_report(self) -> str:
        sep = "=" * 68
        dash = "-" * 68
        lines = [
            sep,
            "ENTITY RESOLUTION BENCHMARK",
            sep,
            "",
            f"Run ID:       {self.run_id}",
            f"Version:      {self.version}",
            f"Timestamp:    {self.timestamp}",
            f"Python:       {self.system.get('python_version', '?').split()[0]}",
            f"OS:           {self.system.get('platform', '?')}",
            f"CPU:          {self.system.get('processor', '?')}",
            f"System RAM:   {self.system.get('total_ram_gb', '?')} GB",
            "",
        ]

        # Dataset
        ds = self.dataset_stats
        if ds:
            lines += [
                "DATASET",
                dash,
                f"Source 1 records:          {ds.get('n_source1', '?'):>14,}",
                f"Source 2 records:          {ds.get('n_source2', '?'):>14,}" if ds.get('n_source2') else "",
                f"Source 3 records:          {ds.get('n_source3', '?'):>14,}" if ds.get('n_source3') else "",
                f"Total pool:               {ds.get('n_pool', '?'):>14,}",
                f"Unique countries:          {ds.get('n_unique_countries', '?')}",
                f"Countries:                 {', '.join(ds.get('countries', []))}",
            ]
            # Missing values
            for src_label in ["source1", "pool"]:
                mn = ds.get(f"{src_label}_missing_name")
                ma = ds.get(f"{src_label}_missing_address")
                mc = ds.get(f"{src_label}_missing_country")
                if mn is not None:
                    lines.append(f"  {src_label} missing names:    {mn:>10,}")
                if ma is not None:
                    lines.append(f"  {src_label} missing addrs:    {ma:>10,}")
                if mc is not None:
                    lines.append(f"  {src_label} missing country:  {mc:>10,}")
            # Name / address stats
            for src_label in ["source1", "pool"]:
                anl = ds.get(f"{src_label}_avg_name_len")
                ant = ds.get(f"{src_label}_avg_name_tokens")
                aal = ds.get(f"{src_label}_avg_addr_len")
                un = ds.get(f"{src_label}_unique_names")
                if anl is not None:
                    lines.append(f"  {src_label} avg name len:     {anl}")
                if ant is not None:
                    lines.append(f"  {src_label} avg name tokens:  {ant}")
                if aal is not None:
                    lines.append(f"  {src_label} avg addr len:     {aal}")
                if un is not None:
                    lines.append(f"  {src_label} unique names:     {un:>10,}")
            lines.append("")

        # Runtime
        lines += ["RUNTIME", dash]
        total = self.timers.get("end_to_end", 1.0)

        # Define display order
        stage_labels = [
            ("load_train_s1", "Load Train S1"),
            ("load_train_s2", "Load Train S2"),
            ("load_train_s3", "Load Train S3"),
            ("load_train_total", "  Total data loading (train)"),
            ("norm_train_s1", "Normalize Train S1"),
            ("norm_train_s2", "Normalize Train S2"),
            ("norm_train_s3", "Normalize Train S3"),
            ("norm_train_total", "  Total normalization (train)"),
            ("pool_construction_train", "Pool construction (train)"),
            ("candidate_gen_train", "Candidate generation (train)"),
            ("blocking_recall_train", "Blocking recall measurement"),
            ("feature_gen_train", "Feature generation (train)"),
            ("train_val_split", "Train/val split"),
            ("model_training", "Model training (XGBoost)"),
            ("threshold_optimization", "Threshold optimization"),
            ("model_retrain_full", "Model retrain (full data)"),
            ("ml_total", "  Total ML"),
            ("load_test_s1", "Load Test S1"),
            ("load_test_s2", "Load Test S2"),
            ("load_test_s3", "Load Test S3"),
            ("load_test_total", "  Total data loading (test)"),
            ("norm_test_s1", "Normalize Test S1"),
            ("norm_test_s2", "Normalize Test S2"),
            ("norm_test_s3", "Normalize Test S3"),
            ("norm_test_total", "  Total normalization (test)"),
            ("pool_construction_test", "Pool construction (test)"),
            ("candidate_gen_test", "Candidate generation (test)"),
            ("feature_gen_test", "Feature generation (test)"),
            ("prediction_test", "Prediction (test)"),
            ("write_candidate_pairs", "Write candidate_pairs.tsv"),
            ("write_matching_results", "Write matching_results.tsv"),
            ("output_total", "  Total output"),
            ("train_pipeline", "--- Train pipeline total ---"),
            ("test_pipeline", "--- Test pipeline total ---"),
            ("end_to_end", "=== END-TO-END ==="),
        ]
        lines.append(f"{'Stage':<36s} {'Time (s)':>10s} {'% Total':>10s}")
        lines.append("-" * 58)
        for key, label in stage_labels:
            t = self.timers.get(key)
            if t is not None:
                pct = 100.0 * t / total if total > 0 else 0.0
                lines.append(f"{label:<36s} {t:>10.2f} {pct:>9.1f}%")
        lines.append("")

        # Memory
        mem_d = self.mem.report_dict()
        if mem_d:
            lines += ["MEMORY", dash]
            for label, info in mem_d.items():
                if label.startswith("_"):
                    continue
                gb = info.get("rss_gb", "?")
                lines.append(f"{label:<36s} {gb:>10.3f} GB")
            peak = mem_d.get("_peak_rss", {})
            lines.append(f"{'Peak RSS':<36s} {peak.get('rss_gb', '?'):>10.3f} GB")
            pct_sys = mem_d.get("_peak_pct_system")
            if pct_sys is not None:
                lines.append(f"{'Peak % system RAM':<36s} {pct_sys:>9.2f}%")
            sys_gb = mem_d.get("_system_ram_gb")
            if sys_gb:
                lines.append(f"{'Total system RAM':<36s} {sys_gb:>10.2f} GB")
            lines.append("")

        # Blocking / Candidate Quality
        cq = self.candidate_quality
        if cq:
            lines += ["BLOCKING / CANDIDATE QUALITY", dash]
            tp = cq.get("total_possible_pairs", 0)
            lines.append(f"Total possible pairs:      {tp:>20,}")
            ss = cq.get("size_stats", {})
            lines.append(f"Candidate pairs:           {ss.get('total_pairs', 0):>20,}")
            lines.append(f"Avg candidates/entity:     {ss.get('avg', 0):>20.1f}")
            lines.append(f"Median candidates/entity:  {ss.get('median', 0):>20}")
            lines.append(f"Min candidates/entity:     {ss.get('min', 0):>20}")
            lines.append(f"Max candidates/entity:     {ss.get('max', 0):>20}")
            lines.append(f"P90 candidates/entity:     {ss.get('p90', 0):>20}")
            lines.append(f"P95 candidates/entity:     {ss.get('p95', 0):>20}")
            lines.append(f"P99 candidates/entity:     {ss.get('p99', 0):>20}")
            lines.append("")
            lines.append(f"Reduction ratio:           {cq.get('reduction_ratio_pct', '?'):>20s}")
            lines.append("")
            lines.append(f"Overall candidate recall:  {cq.get('overall_recall', 0):>20.4f}")
            lines.append(f"  Entities full recall:    {cq.get('entities_full_recall', 0):>20,}")
            lines.append(f"  Entities partial recall: {cq.get('entities_partial_recall', 0):>20,}")
            lines.append(f"  Entities zero recall:    {cq.get('entities_zero_recall', 0):>20,}")
            lines.append("")
            lines.append("Candidate Recall @ K:")
            for k_str, v in cq.get("recall_at_k", {}).items():
                lines.append(f"  Recall @{k_str:>4s}:             {v:>18.4f}")
            lines.append("")

        # Blocking Efficiency
        be = self.blocking_efficiency
        if be:
            lines += ["BLOCKING EFFICIENCY", dash]
            for k, v in be.items():
                lines.append(f"{k:<36s} {v}")
            lines.append("")

        # Model quality
        mq = self.model_quality
        if mq:
            lines += ["MODEL QUALITY", dash]
            lines.append(f"Micro Precision:           {mq.get('micro_precision', 0):>20.4f}")
            lines.append(f"Micro Recall:              {mq.get('micro_recall', 0):>20.4f}")
            lines.append(f"Micro F1:                  {mq.get('micro_f1', 0):>20.4f}")
            lines.append(f"Micro F0.5:                {mq.get('micro_f05', 0):>20.4f}")
            lines.append(f"Macro F0.5 (competition):  {mq.get('macro_f05', 0):>20.4f}")
            lines.append(f"True positives:            {mq.get('true_positives', 0):>20,}")
            lines.append(f"False positives:           {mq.get('false_positives', 0):>20,}")
            lines.append(f"False negatives:           {mq.get('false_negatives', 0):>20,}")
            lines.append(f"Predicted matches:         {mq.get('predicted_matches', 0):>20,}")
            lines.append(f"Entities w/ prediction:    {mq.get('entities_with_prediction', 0):>20,}")
            lines.append(f"Entities w/o prediction:   {mq.get('entities_no_prediction', 0):>20,}")
            lines.append("")

            # Candidate recall vs model recall
            cr = self.candidate_quality.get("overall_recall")
            mr = mq.get("micro_recall")
            if cr is not None and mr is not None:
                lines.append("Recall Pipeline:")
                lines.append(f"  Candidate Recall (ceiling):  {cr:.4f}")
                lines.append(f"  Final Model Recall:          {mr:.4f}")
                gap = cr - mr
                lines.append(f"  Gap (lost by model):         {gap:.4f}")
                lines.append("")

        # Throughput
        if self.throughputs:
            lines += ["THROUGHPUT", dash]
            for label, val in self.throughputs.items():
                lines.append(f"{label:<36s} {val:>14,.1f} /s")
            lines.append("")

        # Config
        lines += ["CONFIGURATION", dash]
        for k, v in self.config.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append(sep)

        # Filter empty lines from conditional sections
        return "\n".join(l for l in lines if l is not None) + "\n"

    # ---- CSV history ----

    def _append_csv_history(self, csv_path: str):
        """Append one summary row to the benchmark history CSV."""
        cq = self.candidate_quality
        mq = self.model_quality
        ss = cq.get("size_stats", {}) if cq else {}

        row = {
            "run_id": self.run_id,
            "version": self.version,
            "timestamp": self.timestamp,
            "total_time_s": self.timers.get("end_to_end", ""),
            "train_pipeline_s": self.timers.get("train_pipeline", ""),
            "test_pipeline_s": self.timers.get("test_pipeline", ""),
            "peak_memory_gb": round(MemoryTracker.bytes_to_gb(self.mem.peak_rss), 3),
            "candidate_count": ss.get("total_pairs", ""),
            "avg_candidates": ss.get("avg", ""),
            "median_candidates": ss.get("median", ""),
            "p95_candidates": ss.get("p95", ""),
            "reduction_ratio": cq.get("reduction_ratio", "") if cq else "",
            "candidate_recall": cq.get("overall_recall", "") if cq else "",
            "recall_at_10": cq.get("recall_at_k", {}).get("10", "") if cq else "",
            "recall_at_20": cq.get("recall_at_k", {}).get("20", "") if cq else "",
            "recall_at_50": cq.get("recall_at_k", {}).get("50", "") if cq else "",
            "micro_precision": mq.get("micro_precision", "") if mq else "",
            "micro_recall": mq.get("micro_recall", "") if mq else "",
            "micro_f1": mq.get("micro_f1", "") if mq else "",
            "micro_f05": mq.get("micro_f05", "") if mq else "",
            "macro_f05": mq.get("macro_f05", "") if mq else "",
            "top_k": self.config.get("top_k", ""),
            "batch_size": self.config.get("batch_size", ""),
            "n_source1": self.dataset_stats.get("n_source1", ""),
            "n_pool": self.dataset_stats.get("n_pool", ""),
        }

        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 10. BASELINE COMPARISON UTILITY                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def compare_benchmarks(
    baseline_path: str,
    optimized_path: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Compare two benchmark JSON files and produce a human-readable diff.

    Can also be used from the CLI:
        python benchmark.py compare output/baseline.json output/optimized.json
    """
    with open(baseline_path, "r", encoding="utf-8") as f:
        base = json.load(f)
    with open(optimized_path, "r", encoding="utf-8") as f:
        opt = json.load(f)

    lines = []
    sep = "=" * 72
    dash = "-" * 72

    lines.append(sep)
    lines.append("BENCHMARK COMPARISON")
    lines.append(sep)
    lines.append(f"Baseline:   {base.get('version', '?')}  (run {base.get('run_id', '?')})")
    lines.append(f"Optimized:  {opt.get('version', '?')}  (run {opt.get('run_id', '?')})")
    lines.append("")
    lines.append(f"{'Metric':<32s} {'Baseline':>14s} {'Optimized':>14s} {'Change':>14s}")
    lines.append(dash)

    def _row(label, bval, oval, fmt=".2f", higher_is_better=None, unit=""):
        """Build one comparison row."""
        if bval is None or oval is None:
            return f"{label:<32s} {'N/A':>14s} {'N/A':>14s} {'N/A':>14s}"
        try:
            bv = float(bval)
            ov = float(oval)
        except (TypeError, ValueError):
            return f"{label:<32s} {str(bval):>14s} {str(oval):>14s} {'N/A':>14s}"

        if bv != 0:
            pct_change = 100.0 * (ov - bv) / abs(bv)
            sign = "+" if pct_change >= 0 else ""
            change_str = f"{sign}{pct_change:.1f}%"
        else:
            change_str = "N/A"

        # Direction indicator
        if higher_is_better is not None and bv != 0:
            direction = ov - bv
            if (higher_is_better and direction > 0) or (not higher_is_better and direction < 0):
                change_str += " ✓"
            elif (higher_is_better and direction < 0) or (not higher_is_better and direction > 0):
                change_str += " ✗"

        b_str = f"{bv:{fmt}}{unit}"
        o_str = f"{ov:{fmt}}{unit}"
        return f"{label:<32s} {b_str:>14s} {o_str:>14s} {change_str:>14s}"

    # Timers
    bt = base.get("timers", {})
    ot = opt.get("timers", {})
    lines.append(_row("Total runtime (s)", bt.get("end_to_end"), ot.get("end_to_end"), higher_is_better=False))
    lines.append(_row("Train pipeline (s)", bt.get("train_pipeline"), ot.get("train_pipeline"), higher_is_better=False))
    lines.append(_row("Test pipeline (s)", bt.get("test_pipeline"), ot.get("test_pipeline"), higher_is_better=False))
    lines.append(_row("Candidate gen train (s)", bt.get("candidate_gen_train"), ot.get("candidate_gen_train"), higher_is_better=False))
    lines.append(_row("Feature gen train (s)", bt.get("feature_gen_train"), ot.get("feature_gen_train"), higher_is_better=False))

    # Memory
    bm = base.get("memory", {}).get("_peak_rss", {})
    om = opt.get("memory", {}).get("_peak_rss", {})
    lines.append(_row("Peak RAM (GB)", bm.get("rss_gb"), om.get("rss_gb"), higher_is_better=False))

    # Candidate quality
    bcq = base.get("candidate_quality", {})
    ocq = opt.get("candidate_quality", {})
    bss = bcq.get("size_stats", {})
    oss = ocq.get("size_stats", {})
    lines.append(_row("Candidate pairs", bss.get("total_pairs"), oss.get("total_pairs"), ".0f", higher_is_better=False))
    lines.append(_row("Avg candidates/entity", bss.get("avg"), oss.get("avg"), higher_is_better=False))
    lines.append(_row("Candidate recall", bcq.get("overall_recall"), ocq.get("overall_recall"), ".4f", higher_is_better=True))
    lines.append(_row("Reduction ratio", bcq.get("reduction_ratio"), ocq.get("reduction_ratio"), ".6f", higher_is_better=True))

    # Recall @ K
    for k in ["10", "20", "50", "100"]:
        lines.append(_row(
            f"Recall @{k}",
            bcq.get("recall_at_k", {}).get(k),
            ocq.get("recall_at_k", {}).get(k),
            ".4f",
            higher_is_better=True
        ))

    # Model quality
    bmq = base.get("model_quality", {})
    omq = opt.get("model_quality", {})
    lines.append(_row("Micro Precision", bmq.get("micro_precision"), omq.get("micro_precision"), ".4f", higher_is_better=True))
    lines.append(_row("Micro Recall", bmq.get("micro_recall"), omq.get("micro_recall"), ".4f", higher_is_better=True))
    lines.append(_row("Micro F1", bmq.get("micro_f1"), omq.get("micro_f1"), ".4f", higher_is_better=True))
    lines.append(_row("Macro F0.5", bmq.get("macro_f05"), omq.get("macro_f05"), ".4f", higher_is_better=True))

    lines.append(dash)
    lines.append("  ✓ = improvement    ✗ = regression")
    lines.append(sep)

    report = "\n".join(lines) + "\n"

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Comparison saved: {output_path}")

    return report


def compare_from_csv(csv_path: str, version_a: str, version_b: str) -> str:
    """
    Compare two versions from the benchmark history CSV.
    Finds the latest run for each version label.
    """
    import csv as _csv

    rows_a = []
    rows_b = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            if row.get("version") == version_a:
                rows_a.append(row)
            elif row.get("version") == version_b:
                rows_b.append(row)

    if not rows_a:
        return f"No runs found for version '{version_a}'"
    if not rows_b:
        return f"No runs found for version '{version_b}'"

    a = rows_a[-1]  # latest
    b = rows_b[-1]

    sep = "=" * 72
    dash = "-" * 72
    lines = [
        sep,
        "BENCHMARK COMPARISON (from CSV history)",
        sep,
        f"Version A: {version_a}  (run {a.get('run_id', '?')})",
        f"Version B: {version_b}  (run {b.get('run_id', '?')})",
        "",
        f"{'Metric':<32s} {version_a:>14s} {version_b:>14s} {'Change':>14s}",
        dash,
    ]

    compare_keys = [
        ("total_time_s", "Total runtime (s)", False),
        ("peak_memory_gb", "Peak RAM (GB)", False),
        ("candidate_count", "Candidate pairs", False),
        ("avg_candidates", "Avg candidates", False),
        ("candidate_recall", "Candidate recall", True),
        ("recall_at_10", "Recall @10", True),
        ("recall_at_20", "Recall @20", True),
        ("recall_at_50", "Recall @50", True),
        ("micro_precision", "Precision", True),
        ("micro_recall", "Recall", True),
        ("micro_f1", "F1", True),
        ("macro_f05", "Macro F0.5", True),
    ]

    for key, label, higher_better in compare_keys:
        va = a.get(key, "")
        vb = b.get(key, "")
        try:
            fva = float(va)
            fvb = float(vb)
            if fva != 0:
                pct = 100.0 * (fvb - fva) / abs(fva)
                sign = "+" if pct >= 0 else ""
                change = f"{sign}{pct:.1f}%"
                direction = fvb - fva
                if (higher_better and direction > 0) or (not higher_better and direction < 0):
                    change += " ✓"
                elif (higher_better and direction < 0) or (not higher_better and direction > 0):
                    change += " ✗"
            else:
                change = "N/A"
            lines.append(f"{label:<32s} {fva:>14.4f} {fvb:>14.4f} {change:>14s}")
        except (ValueError, TypeError):
            lines.append(f"{label:<32s} {str(va):>14s} {str(vb):>14s} {'N/A':>14s}")

    lines += [dash, sep]
    return "\n".join(lines) + "\n"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ CLI ENTRY POINT (for comparisons)                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark comparison utility")
    sub = parser.add_subparsers(dest="command")

    # compare two JSON files
    cmp_parser = sub.add_parser("compare", help="Compare two benchmark JSON files")
    cmp_parser.add_argument("baseline", help="Path to baseline benchmark_metrics.json")
    cmp_parser.add_argument("optimized", help="Path to optimized benchmark_metrics.json")
    cmp_parser.add_argument("-o", "--output", help="Save comparison to file", default=None)

    # compare from CSV
    csv_parser = sub.add_parser("compare-csv", help="Compare two versions from CSV history")
    csv_parser.add_argument("csv_path", help="Path to benchmark_history.csv")
    csv_parser.add_argument("version_a", help="Version label A")
    csv_parser.add_argument("version_b", help="Version label B")

    args = parser.parse_args()

    if args.command == "compare":
        report = compare_benchmarks(args.baseline, args.optimized, args.output)
        print(report)
    elif args.command == "compare-csv":
        report = compare_from_csv(args.csv_path, args.version_a, args.version_b)
        print(report)
    else:
        parser.print_help()
