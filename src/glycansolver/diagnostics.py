"""
Model diagnostics for GlycanSolver results.

Provides two families of diagnostics:

1. **Residual diagnostics** (recommendation #1)
   - Error distribution: mean, median, std, skewness, kurtosis
   - Shapiro–Wilk normality test on errors
   - Systematic bias: OLS regression of error vs observed mass
   - PPM-based errors (mass-relative)
   - R² (coefficient of determination)

2. **Block usage statistics** (recommendation #4)
   - Per-block usage count (how many GOOD peaks use it)
   - Usage fraction
   - Flag blocks used by ≤2 peaks (may be over-fitted)
   - Co-occurrence matrix: how often pairs of blocks appear together
   - Block complexity distribution

Both diagnostics are computed per model when multi-model results
are available.

Output
------
* Returns a structured dict suitable for JSON serialisation / web display.
* Writes ``diagnostics_report.txt`` — a detailed human-readable report.
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class ResidualStats(NamedTuple):
    """Summary statistics for the error distribution of one model."""
    n_good: int
    n_bad: int
    mean_error: float
    median_error: float
    std_error: float
    max_error: float
    min_error: float
    skewness: float
    kurtosis: float           # excess kurtosis
    shapiro_w: float | None   # Shapiro–Wilk W statistic (None if n < 3)
    shapiro_p: float | None   # Shapiro–Wilk p-value
    # Systematic bias: error = slope × observed + intercept
    bias_slope: float
    bias_intercept: float
    bias_r2: float            # R² of the bias regression
    bias_significant: bool    # slope significantly ≠ 0 at 5 %?
    # PPM errors
    mean_ppm: float
    median_ppm: float
    max_ppm: float
    # Overall model fit
    r_squared: float          # 1 – RSS/TSS relative to observed masses
    rss: float
    tss: float


class BlockUsage(NamedTuple):
    """Usage statistics for a single building block."""
    name: str
    mass: float | None
    block_type: str           # "known" | "discovered"
    n_used: int               # number of GOOD peaks using it (count > 0)
    frac_used: float          # n_used / n_good
    mean_copies: float        # mean copy count when used
    max_copies: int
    flagged: bool             # True when n_used ≤ 2


class BlockUsageStats(NamedTuple):
    """Aggregate block-usage diagnostics for one model."""
    blocks: list[BlockUsage]
    co_occurrence: dict[tuple[str, str], int]  # (blockA, blockB) → count
    mean_blocks_per_peak: float
    max_blocks_per_peak: int
    peaks_common_only: int  # peaks using NO differential block at all


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list[float], mean: float) -> float:
    if len(vals) < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


def _skewness(vals: list[float], mean: float, std: float) -> float:
    """Sample skewness (Fisher)."""
    n = len(vals)
    if n < 3 or std < 1e-15:
        return 0.0
    m3 = sum((v - mean) ** 3 for v in vals) / n
    return m3 / (std ** 3)


def _kurtosis(vals: list[float], mean: float, std: float) -> float:
    """Excess kurtosis (Fisher). Normal = 0."""
    n = len(vals)
    if n < 4 or std < 1e-15:
        return 0.0
    m4 = sum((v - mean) ** 4 for v in vals) / n
    return m4 / (std ** 4) - 3.0


def _ols(x: list[float], y: list[float]) -> tuple[float, float, float, bool]:
    """Simple OLS: y = slope*x + intercept.

    Returns (slope, intercept, r², significant_at_5pct).
    Uses the t-test on slope with n-2 dof; approximate p via
    |t| > 2 rule for modest n.
    """
    n = len(x)
    if n < 3:
        return 0.0, _mean(y), 0.0, False

    mx, my = _mean(x), _mean(y)
    sxx = sum((xi - mx) ** 2 for xi in x)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))

    if sxx < 1e-20:
        return 0.0, my, 0.0, False

    slope = sxy / sxx
    intercept = my - slope * mx

    # R²
    ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))
    ss_tot = sum((yi - my) ** 2 for yi in y)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else 0.0

    # significance: t = slope / SE(slope)
    if n > 2 and ss_res > 0:
        se_slope = math.sqrt(ss_res / (n - 2) / sxx)
        t_stat = abs(slope / se_slope) if se_slope > 1e-20 else 0.0
        # Rough: |t| > 2.0 ≈ p < 0.05 for n ≥ 10
        significant = t_stat > 2.0
    else:
        significant = False

    return slope, intercept, max(r2, 0.0), significant


def _shapiro_wilk(vals: list[float]) -> tuple[float | None, float | None]:
    """Shapiro-Wilk test.  Uses scipy if available, else returns None."""
    if len(vals) < 3:
        return None, None
    try:
        from scipy.stats import shapiro
        w, p = shapiro(vals)
        return float(w), float(p)
    except ImportError:
        return None, None
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Results parser
# ---------------------------------------------------------------------------

def _parse_results(path: str) -> tuple[list[str], list[dict]]:
    """Read results.tsv and return (block_names, rows)."""
    if not os.path.exists(path):
        return [], []
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        headers = reader.fieldnames or []
        fixed_cols = {
            "Peak_ID", "Observed", "Reconstructed", "Reconstructed_Theoretical",
            "Error", "Status",
            "Formula", "Model", "Model_Blocks", "Model_BIC",
            "N_Models_Tested", "N_Models_Explaining",
            "Composition_Consistent",
            "Best_Model", "Best_Formula",
            "Structure", "Best_Structure",
        }
        block_names = [h for h in headers if h not in fixed_cols]
        rows = list(reader)
    return block_names, rows


def _parse_blocks(path: str) -> dict[str, dict]:
    """Read blocks.tsv → {name: {value, type}}."""
    if not os.path.exists(path):
        return {}
    info: dict[str, dict] = {}
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            name = row.get("Block", "")
            if name.lower() == "common":
                continue
            info[name] = {
                "value": float(row.get("Value", 0) or 0),
                "type": row.get("Type", "known"),
            }
    return info


def _parse_block_count(value) -> int:
    """Parse one block-count cell from results.tsv."""
    try:
        return int(float(value or 0))
    except (ValueError, TypeError):
        return 0


def _infer_model_blocks(
    model_key: str,
    model_rows: list[dict],
    block_names: list[str],
) -> list[str]:
    """Return the block columns that belong to one model.

    Subset models encode their block list directly in the ``Model`` label
    (for example ``Hex+Fuc``). Consensus-style synthetic models do not, so
    for those we infer the active block set from any positive block usage
    across the model's rows.
    """
    if not block_names:
        return []

    if model_key and model_key != "Model":
        label_parts = [part.strip() for part in model_key.split("+") if part.strip()]
        if label_parts and all(part in block_names for part in label_parts):
            return [name for name in block_names if name in label_parts]

    return [
        name
        for name in block_names
        if any(_parse_block_count(row.get(name, 0)) > 0 for row in model_rows)
    ]


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------

def compute_residual_diagnostics(
    observed: list[float],
    reconstructed: list[float],
    errors: list[float],
    statuses: list[str],
    tolerance: float | None = None,
) -> ResidualStats:
    """Compute residual diagnostics for a single model's peaks."""
    good_mask = [s.strip() == "GOOD" for s in statuses]
    n_good = sum(good_mask)
    n_bad = len(statuses) - n_good

    # Use absolute errors for the analysis
    abs_err = [abs(e) for e in errors]
    good_err = [abs_err[i] for i in range(len(abs_err)) if good_mask[i]]
    good_obs = [observed[i] for i in range(len(observed)) if good_mask[i]]
    good_rec = [reconstructed[i] for i in range(len(reconstructed)) if good_mask[i]]

    if not good_err:
        return ResidualStats(
            n_good=0, n_bad=n_bad,
            mean_error=0, median_error=0, std_error=0, max_error=0, min_error=0,
            skewness=0, kurtosis=0,
            shapiro_w=None, shapiro_p=None,
            bias_slope=0, bias_intercept=0, bias_r2=0, bias_significant=False,
            mean_ppm=0, median_ppm=0, max_ppm=0,
            r_squared=0, rss=0, tss=0,
        )

    me = _mean(good_err)
    md = _median(good_err)
    sd = _std(good_err, me)
    mx = max(good_err)
    mn = min(good_err)
    sk = _skewness(good_err, me, sd)
    ku = _kurtosis(good_err, me, sd)

    # Shapiro-Wilk on *signed* residuals (observed − reconstructed)
    signed_err = [good_obs[i] - good_rec[i] for i in range(n_good)]
    sw, sp = _shapiro_wilk(signed_err)

    # Bias regression: signed error vs observed mass
    slope, intercept, bias_r2, sig = _ols(good_obs, signed_err)

    # PPM errors
    ppm = [abs_err[i] / observed[i] * 1e6 if observed[i] > 0 else 0
           for i in range(len(observed)) if good_mask[i]]
    mean_ppm = _mean(ppm)
    med_ppm = _median(ppm)
    max_ppm = max(ppm) if ppm else 0

    # R² of the overall model fit
    rss = sum((good_obs[i] - good_rec[i]) ** 2 for i in range(n_good))
    mean_obs = _mean(good_obs)
    tss = sum((v - mean_obs) ** 2 for v in good_obs)
    r2 = 1.0 - rss / tss if tss > 1e-20 else 0.0

    return ResidualStats(
        n_good=n_good, n_bad=n_bad,
        mean_error=me, median_error=md, std_error=sd,
        max_error=mx, min_error=mn,
        skewness=sk, kurtosis=ku,
        shapiro_w=sw, shapiro_p=sp,
        bias_slope=slope, bias_intercept=intercept,
        bias_r2=bias_r2, bias_significant=sig,
        mean_ppm=mean_ppm, median_ppm=med_ppm, max_ppm=max_ppm,
        r_squared=max(r2, 0.0), rss=rss, tss=tss,
    )


# ---------------------------------------------------------------------------
# Block usage statistics
# ---------------------------------------------------------------------------

def compute_block_usage(
    block_names: list[str],
    rows: list[dict],
    block_info: dict[str, dict],
) -> BlockUsageStats:
    """Compute block usage statistics for a set of GOOD rows."""
    n_good = len(rows)
    if not rows or not block_names:
        return BlockUsageStats(
            blocks=[], co_occurrence={},
            mean_blocks_per_peak=0, max_blocks_per_peak=0,
            peaks_common_only=0,
        )

    block_usages: list[BlockUsage] = []
    # per-peak: which blocks are used (count > 0)
    peak_block_sets: list[list[str]] = []
    blocks_per_peak: list[int] = []

    peaks_common_only = 0

    # Gather per-block counts
    block_counts_all: dict[str, list[int]] = {b: [] for b in block_names}

    for row in rows:
        used = []
        for b in block_names:
            try:
                count = int(float(row.get(b, 0) or 0))
            except (ValueError, TypeError):
                count = 0
            block_counts_all[b].append(count)
            if count > 0:
                used.append(b)
        peak_block_sets.append(used)
        blocks_per_peak.append(len(used))
        if not used:
            peaks_common_only += 1

    for b in block_names:
        counts = block_counts_all[b]
        used_counts = [c for c in counts if c > 0]
        n_used = len(used_counts)
        info = block_info.get(b, {})
        block_usages.append(BlockUsage(
            name=b,
            mass=info.get("value"),
            block_type=info.get("type", "known"),
            n_used=n_used,
            frac_used=n_used / n_good if n_good else 0,
            mean_copies=_mean([float(c) for c in used_counts]) if used_counts else 0,
            max_copies=max(counts) if counts else 0,
            flagged=n_used <= 2 and n_good > 5,
        ))

    # Co-occurrence: how often block pairs appear together in the same peak
    co_occ: dict[tuple[str, str], int] = {}
    for used in peak_block_sets:
        for i, a in enumerate(used):
            for b in used[i + 1:]:
                key = (a, b) if a < b else (b, a)
                co_occ[key] = co_occ.get(key, 0) + 1

    return BlockUsageStats(
        blocks=block_usages,
        co_occurrence=co_occ,
        mean_blocks_per_peak=_mean([float(x) for x in blocks_per_peak]),
        max_blocks_per_peak=max(blocks_per_peak) if blocks_per_peak else 0,
        peaks_common_only=peaks_common_only,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_diagnostics(
    results_tsv_path: str,
    output_dir: str,
) -> dict:
    """Run all diagnostics and write a detailed report.

    Returns a JSON-serialisable dict with keys:
      ``models`` — list of per-model diagnostics
    """
    block_names, rows = _parse_results(results_tsv_path)
    if not rows:
        return {"models": []}

    blocks_tsv = os.path.join(output_dir, "blocks.tsv")
    block_info = _parse_blocks(blocks_tsv)

    has_multimodel = "Model" in (rows[0] if rows else {})

    # Group rows by model
    model_rows: dict[str, list[dict]] = defaultdict(list)
    model_bics: dict[str, float | None] = {}

    for row in rows:
        model_key = row.get("Model", "Model") if has_multimodel else "Model"
        model_rows[model_key].append(row)
        if model_key not in model_bics:
            try:
                model_bics[model_key] = float(row["Model_BIC"]) if row.get("Model_BIC") else None
            except (ValueError, TypeError):
                model_bics[model_key] = None

    models_out: list[dict] = []

    for model_key, mrows in model_rows.items():
        # Separate GOOD / BAD
        good_rows = [r for r in mrows if r.get("Status", "").strip() == "GOOD"]

        observed = []
        reconstructed = []
        errors = []
        statuses = []

        for r in mrows:
            try:
                observed.append(float(r.get("Observed", 0) or 0))
                reconstructed.append(float(r.get("Reconstructed", 0) or 0))
                errors.append(float(r.get("Error", 0) or 0))
                statuses.append(r.get("Status", "BAD"))
            except (ValueError, TypeError):
                continue

        # ---- residual diagnostics ----
        resid = compute_residual_diagnostics(observed, reconstructed, errors, statuses)

        # ---- block usage (GOOD peaks only) ----
        active_blocks = (
            _infer_model_blocks(model_key, mrows, block_names)
            if has_multimodel else block_names
        )

        usage = compute_block_usage(active_blocks, good_rows, block_info)

        # Per-peak points for residual scatter chart
        peak_points = []
        for i, r in enumerate(mrows):
            if i >= len(observed):
                break
            signed_err = observed[i] - reconstructed[i]
            peak_points.append({
                "peak_id": int(r.get("Peak_ID", 0) or 0),
                "mz": round(observed[i], 5),
                "error": round(signed_err, 6),
                "status": statuses[i].strip(),
            })

        models_out.append({
            "model": model_key,
            "bic": model_bics.get(model_key),
            "residuals": _serialise_residuals(resid),
            "block_usage": _serialise_usage(usage),
            "peak_points": peak_points,
        })

    # ---- write detailed report ----
    report_path = os.path.join(output_dir, "diagnostics_report.txt")
    _write_report(models_out, block_names, report_path)
    print(f"Diagnostics report written to {report_path}")

    return {"models": models_out}


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _serialise_residuals(r: ResidualStats) -> dict:
    return {
        "n_good": r.n_good,
        "n_bad": r.n_bad,
        "mean_error": round(r.mean_error, 6),
        "median_error": round(r.median_error, 6),
        "std_error": round(r.std_error, 6),
        "max_error": round(r.max_error, 6),
        "min_error": round(r.min_error, 6),
        "skewness": round(r.skewness, 4),
        "kurtosis": round(r.kurtosis, 4),
        "shapiro_w": round(r.shapiro_w, 4) if r.shapiro_w is not None else None,
        "shapiro_p": round(r.shapiro_p, 4) if r.shapiro_p is not None else None,
        "normality_ok": r.shapiro_p is not None and r.shapiro_p > 0.05,
        "bias_slope": round(r.bias_slope, 8),
        "bias_intercept": round(r.bias_intercept, 6),
        "bias_r2": round(r.bias_r2, 6),
        "bias_significant": r.bias_significant,
        "mean_ppm": round(r.mean_ppm, 2),
        "median_ppm": round(r.median_ppm, 2),
        "max_ppm": round(r.max_ppm, 2),
        "r_squared": round(r.r_squared, 8),
        "rss": round(r.rss, 6),
        "tss": round(r.tss, 2),
    }


def _serialise_usage(u: BlockUsageStats) -> dict:
    blocks_out = []
    for b in u.blocks:
        blocks_out.append({
            "name": b.name,
            "mass": round(b.mass, 5) if b.mass is not None else None,
            "type": b.block_type,
            "n_used": b.n_used,
            "frac_used": round(b.frac_used, 3),
            "mean_copies": round(b.mean_copies, 2),
            "max_copies": b.max_copies,
            "flagged": b.flagged,
        })

    co_occ_out = [
        {"blocks": list(k), "count": v}
        for k, v in sorted(u.co_occurrence.items(), key=lambda x: -x[1])
    ]

    return {
        "blocks": blocks_out,
        "co_occurrence": co_occ_out,
        "mean_blocks_per_peak": round(u.mean_blocks_per_peak, 2),
        "max_blocks_per_peak": u.max_blocks_per_peak,
        "peaks_common_only": u.peaks_common_only,
        "n_flagged": sum(1 for b in u.blocks if b.flagged),
    }


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------

def _write_report(models: list[dict], block_names: list[str], path: str) -> None:
    """Write a detailed, human-readable diagnostics report to a text file."""

    lines: list[str] = []
    W = 72

    lines.append("=" * W)
    lines.append("GlycanSolver — Model Diagnostics Report")
    lines.append("=" * W)
    lines.append("")

    for m in models:
        model_key = m["model"]
        bic = m["bic"]
        r = m["residuals"]
        u = m["block_usage"]

        lines.append("-" * W)
        lines.append(f"Model: {model_key}")
        if bic is not None:
            lines.append(f"BIC  : {bic:.2f}")
        lines.append("-" * W)

        # ---- residual diagnostics ----
        lines.append("")
        lines.append("  RESIDUAL DIAGNOSTICS")
        lines.append("  " + "~" * 40)
        lines.append(f"  Peaks: {r['n_good']} GOOD, {r['n_bad']} BAD")
        lines.append(f"  R²   : {r['r_squared']:.8f}")
        lines.append("")
        lines.append("  Error distribution (absolute, GOOD peaks):")
        lines.append(f"    Mean     : {r['mean_error']:.6f} Da")
        lines.append(f"    Median   : {r['median_error']:.6f} Da")
        lines.append(f"    Std dev  : {r['std_error']:.6f} Da")
        lines.append(f"    Min      : {r['min_error']:.6f} Da")
        lines.append(f"    Max      : {r['max_error']:.6f} Da")
        lines.append(f"    Skewness : {r['skewness']:.4f}")
        lines.append(f"    Kurtosis : {r['kurtosis']:.4f} (excess, normal=0)")
        lines.append("")

        lines.append("  PPM errors:")
        lines.append(f"    Mean     : {r['mean_ppm']:.2f} ppm")
        lines.append(f"    Median   : {r['median_ppm']:.2f} ppm")
        lines.append(f"    Max      : {r['max_ppm']:.2f} ppm")
        lines.append("")

        lines.append("  Normality test (Shapiro–Wilk on signed residuals):")
        if r["shapiro_w"] is not None:
            verdict = "PASS (residuals consistent with normal)" if r["normality_ok"] \
                else "FAIL (residuals deviate from normal, p < 0.05)"
            lines.append(f"    W = {r['shapiro_w']:.4f},  p = {r['shapiro_p']:.4f}")
            lines.append(f"    Verdict: {verdict}")
        else:
            lines.append("    (not available — requires scipy or n ≥ 3)")
        lines.append("")

        lines.append("  Systematic bias (OLS: signed_error ~ observed_mass):")
        lines.append(f"    Slope     : {r['bias_slope']:.8f} Da/Da")
        lines.append(f"    Intercept : {r['bias_intercept']:.6f} Da")
        lines.append(f"    R²        : {r['bias_r2']:.6f}")
        if r["bias_significant"]:
            lines.append("    ⚠  Slope is significantly ≠ 0 (p < 0.05).")
            lines.append("       Error grows/shrinks with mass — possible calibration issue.")
        else:
            lines.append("    ✓  No significant mass-dependent bias detected.")
        lines.append("")

        # ---- block usage ----
        lines.append("  BLOCK USAGE STATISTICS")
        lines.append("  " + "~" * 40)
        lines.append(f"  Mean blocks/peak : {u['mean_blocks_per_peak']:.2f}")
        lines.append(f"  Max blocks/peak  : {u['max_blocks_per_peak']}")
        lines.append(f"  Peaks (common only): {u['peaks_common_only']}")
        lines.append("")

        lines.append(f"  {'Block':<12} {'Type':<12} {'Used':<6} {'Frac':<8} {'Mean×':<8} {'Max×':<6} {'Flag':}")
        lines.append("  " + "-" * 60)
        for b in u["blocks"]:
            flag = "⚠ LOW" if b["flagged"] else ""
            lines.append(
                f"  {b['name']:<12} {b['type']:<12} "
                f"{b['n_used']:<6} {b['frac_used']:<8.3f} "
                f"{b['mean_copies']:<8.2f} {b['max_copies']:<6} {flag}"
            )
        lines.append("")

        if u["n_flagged"] > 0:
            flagged_names = [b["name"] for b in u["blocks"] if b["flagged"]]
            lines.append(f"  ⚠  Flagged blocks (used by ≤2 peaks): {', '.join(flagged_names)}")
            lines.append("     These blocks may be over-fitted — the model might assign")
            lines.append("     them to explain noise rather than real structure.")
            lines.append("")

        if u["co_occurrence"]:
            lines.append("  Block co-occurrence (top pairs):")
            for entry in u["co_occurrence"][:10]:
                a, b_name = entry["blocks"]
                lines.append(f"    {a} + {b_name} : {entry['count']} peaks")
            lines.append("")

        lines.append("")

    # Write
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
