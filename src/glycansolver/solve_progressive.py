"""
Progressive Glycan Solver - Incremental Model Building

Instead of fitting all blocks simultaneously with a fixed number of unknowns,
this solver builds the model incrementally:

1. Starts with known blocks only (baseline)
2. Evaluates candidate unknown blocks one at a time
3. Uses BIC (Bayesian Information Criterion) to decide when to stop adding blocks
4. Prefers simpler models that explain the data well

This avoids:
- Getting stuck with too many unknown blocks
- Forcing all peaks to fit when a simpler answer exists
- Depending on the user guessing the right number of unknowns
"""

import math
import os
import random as _random_mod
import time
import warnings
from collections import Counter, defaultdict

import cvxpy as cp
import numpy as np

from .block_init import get_smart_block_init, load_blocks_dictionary, load_blocks_dictionary_with_categories
from .utils import (
    common_composition_name,
    ensure_output_directory,
    find_nearest_multiple,
    load_peaks,
    merge_structure_formula,
    write_candidates_tsv,
    write_exhaustive_tsv_output,
    write_multimodel_tsv_output,
    write_tsv_output,
)
from .biosynthetic import analyse_biosynthetic_paths
from .block_dependencies import (
    infer_block_dependencies,
    reorder_exhaustive_results,
    reorder_model_label,
    write_dependency_report,
)
from .diagnostics import run_diagnostics

# Suppress harmless warnings:
# 1. Python 3.13+ free-threaded build: _cvxcore C extension re-enables the GIL
#    (upstream cvxpy issue — single-threaded code, so this is harmless).
# 2. cvxpy "Solution may be inaccurate" — we already check prob.status and
#    handle "optimal_inaccurate" explicitly.
warnings.filterwarnings(
    "ignore",
    message="The global interpreter lock \\(GIL\\) has been enabled",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Solution may be inaccurate",
    category=UserWarning,
)


class SolverCancelledError(Exception):
    """Raised when a user-requested cancellation is detected."""


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_bic(n_obs, rss, n_params):
    """
    Bayesian Information Criterion for model selection.
    Lower BIC = better. Penalizes model complexity.

    BIC = n * ln(RSS/n) + k * ln(n)
    """
    if rss < 1e-10:
        rss = 1e-10
    return n_obs * np.log(rss / n_obs) + n_params * np.log(n_obs)


def quick_evaluate_candidate(residuals, candidate_mass, tolerance):
    """
    Fast heuristic: count how many residuals are approximately integer
    multiples of candidate_mass. Higher score → candidate likely explains
    more currently-unexplained peaks.
    """
    if candidate_mass <= 0:
        return 0
    score = 0
    for r in residuals:
        n = round(abs(r) / candidate_mass)
        if n >= 1 and abs(abs(r) - n * candidate_mass) <= tolerance:
            score += 1
    return score


# ---------------------------------------------------------------------------
# Phase optimizer – runs the alternating x/b optimization for a fixed set
# of blocks.  Unknown blocks are NOT forced to be used.
# ---------------------------------------------------------------------------

def run_phase(
    observations,
    common_block,
    block_masses,
    block_is_known,
    known_mass_limits,
    max_known_limit,
    max_unknown_limit,
    tol_recon,
    tol_final,
    max_iter,
    lb_unknown,
    ub_unknown,
    bad_allowed=0,
    postgoal=10,
    common_block_fixed=True,
    timeout_seconds=None,
    start_time=None,
    verbose=False,
    initial_x=None,
    should_cancel=None,
):
    """
    Alternating optimization for a *fixed* set of blocks.

    Key difference from the classic solver: unknown blocks are optional –
    the optimizer decides whether each unknown block improves the fit.

    Parameters
    ----------
    initial_x : np.ndarray or None
        If provided, warm-start the first iteration with this integer
        coefficient matrix (n × k_total).  Avoids a cold MILP solve.

    Returns
    -------
    dict with keys:
        x, b, common, bic, rss, n_good, n_bad, mean_error, errors,
        converged, used_unknowns
    """
    n = len(observations)
    k_total = len(block_masses)
    k_known = int(np.sum(block_is_known))
    k_unknown = k_total - k_known

    b = block_masses.copy().astype(float)
    known_differential = b[:k_known].copy()
    y = observations - common_block

    # ---- hyper-parameters ----
    lambda_block = 10.0
    lambda_known = 2e4
    lambda_multiple = 1e4
    lambda_err = 1e3
    max_lambda_err = 1e6
    multiple_base = 1.00035
    tol_conv = 1e-3

    # ---- state ----
    x_val = None
    if initial_x is not None and initial_x.shape == (n, k_total):
        x_val = initial_x.copy()
    stagnation_counter = 0
    no_improve_counter = 0   # tracks iterations with no improvement in bad_count
    best_x = None
    best_b = None
    best_common = common_block
    best_bad_count = n
    reached_goal = False
    post_goal_iters = 0
    common_history = [common_block]

    for it in range(max_iter):
        if should_cancel and should_cancel():
            raise SolverCancelledError("Analysis stopped by user request.")

        # timeout
        if timeout_seconds and start_time and time.time() - start_time > timeout_seconds:
            print(f"    Phase timeout at iteration {it}")
            break

        # ------ x-update ------
        x = cp.Variable((n, k_total), integer=True)
        z = cp.Variable(k_total, boolean=True)
        s = cp.Variable(n, nonneg=True)

        error_vec = x @ b - y
        constraints = [x >= 0]

        # Known blocks – must be active and used at least once
        for r in range(k_known):
            limit = known_mass_limits[r] if r < len(known_mass_limits) else max_known_limit
            constraints += [
                x[:, r] <= limit * z[r],
                x[:, r] <= limit,
                z[r] == 1,
                cp.sum(x[:, r]) >= 1,
            ]

        # Unknown blocks – usage is *optional*
        for r in range(k_known, k_total):
            constraints += [
                x[:, r] <= max_unknown_limit * z[r],
                x[:, r] <= max_unknown_limit,
            ]
            # No z[r]==1 and no sum>=1 → block may stay unused

        # Slack
        constraints += [
            error_vec <= tol_recon + s,
            -error_vec <= tol_recon + s,
        ]

        obj = (
            cp.sum_squares(error_vec)
            + lambda_err * cp.sum(s)
            + lambda_block * cp.sum(z[k_known:])   # penalise unknown activation
        )

        prob = cp.Problem(cp.Minimize(obj), constraints)  # type: ignore[arg-type]
        try:
            # Warm-start from previous x_val if available
            if x_val is not None:
                x.value = np.round(x_val).astype(float)
                z.value = np.array([
                    1.0 if np.any(np.round(x_val[:, r]) > 0) else 0.0
                    for r in range(k_total)
                ])
                prob.solve(solver=cp.GUROBI, TimeLimit=120, warm_start=True)
            else:
                prob.solve(solver=cp.GUROBI, TimeLimit=120)
            if prob.status in ("optimal", "optimal_inaccurate") and x.value is not None:
                x_val = x.value
            elif x_val is None:
                x_val = np.zeros((n, k_total))
                for r in range(k_known):
                    x_val[:, r] = 1
        except Exception as e:
            if verbose:
                print(f"    x-update error: {e}")
            if x_val is None:
                x_val = np.zeros((n, k_total))
                for r in range(k_known):
                    x_val[:, r] = 1

        if should_cancel and should_cancel():
            raise SolverCancelledError("Analysis stopped by user request.")

        # ------ b-update ------
        b_var = cp.Variable(k_total, nonneg=True)
        constraints_b = [b_var >= 0]
        if k_unknown > 0:
            constraints_b += [
                b_var[k_known:] >= lb_unknown,
                b_var[k_known:] <= ub_unknown,
            ]

        error_b = cp.matmul(x_val, b_var) - y
        penalty_known = lambda_known * cp.sum_squares(
            b_var[:k_known] - known_differential
        )

        multiple_penalty = 0
        if k_unknown > 0 and lambda_multiple > 0:
            for i in range(k_known, k_total):
                nearest_mult = find_nearest_multiple(b[i], multiple_base)
                multiple_penalty += cp.sum_squares(b_var[i] - nearest_mult)

        obj_b = (
            cp.sum_squares(error_b)
            + penalty_known
            + lambda_multiple * multiple_penalty
        )
        prob_b = cp.Problem(cp.Minimize(obj_b), constraints_b)  # type: ignore[arg-type]

        try:
            prob_b.solve(solver=cp.GUROBI, TimeLimit=60)
            if (
                prob_b.status in ("optimal", "optimal_inaccurate")
                and b_var.value is not None
            ):
                b_new = b_var.value
            else:
                b_new = b.copy()
        except Exception:
            b_new = b.copy()

        # ------ common-block update (gentle) ------
        current_recon = np.sum(x_val * b_new, axis=1)
        optimal_common = np.mean(observations - current_recon)

        if common_block_fixed:
            initial_common = common_history[0]
            w_opt, w_init = 1.0, lambda_known
            common_block = (
                (w_opt * optimal_common + w_init * initial_common)
                / (w_opt + w_init)
            )
        else:
            change = optimal_common - common_block
            if abs(change) > 10.0:
                optimal_common = common_block + np.sign(change) * 10.0
            common_block = max(0, optimal_common)

        common_history.append(common_block)
        y = observations - common_block

        # ------ convergence / stagnation ------
        if k_unknown > 0:
            block_delta = np.linalg.norm(b_new[k_known:] - b[k_known:])
            old_recon = common_block + np.sum(x_val * b, axis=1)
            new_recon = common_block + np.sum(x_val * b_new, axis=1)
            err_delta = (
                np.mean(np.abs(observations - old_recon))
                - np.mean(np.abs(observations - new_recon))
            )
            if block_delta < tol_conv and err_delta < 0.01:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
        else:
            if np.linalg.norm(b_new - b) < tol_conv:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

        b = b_new

        # evaluate
        recon = common_block + np.sum(x_val * b, axis=1)
        errors = np.abs(observations - recon)
        bad_count = int(np.sum(errors >= tol_final))

        # ---- progress output ----
        mean_err = float(np.mean(errors))
        print(
            f"    iter {it:3d}: bad={bad_count}/{n}, "
            f"mean_err={mean_err:.4f}, max_err={np.max(errors):.4f}"
        )

        if bad_count < best_bad_count:
            no_improve_counter = 0   # real improvement
            best_bad_count = bad_count
            best_x = x_val.copy()
            best_b = b.copy()
            best_common = common_block
        elif bad_count == best_bad_count:
            no_improve_counter += 1
            # still save (b may have improved slightly)
            best_x = x_val.copy()
            best_b = b.copy()
            best_common = common_block
        else:
            no_improve_counter += 1

        if bad_count <= bad_allowed and not reached_goal:
            reached_goal = True

        if reached_goal:
            post_goal_iters += 1
            if post_goal_iters >= postgoal:
                break

        # Early exit: if bad_count hasn't improved, we're done.
        # Allow more attempts when most peaks are still bad (solver may
        # need extra iterations to escape a poor initial x).
        max_stale = 3 if best_bad_count < n * 0.8 else 6
        if no_improve_counter >= max_stale:
            break

        if bad_count > bad_allowed and lambda_err < max_lambda_err:
            lambda_err *= 1.5

    # ---- final metrics ----
    if best_x is not None and best_b is not None:
        recon = best_common + np.sum(best_x * best_b, axis=1)
        errors = np.abs(observations - recon)
        rss = float(np.sum(errors ** 2))
        n_good = int(np.sum(errors < tol_final))
        n_bad = n - n_good

        used_unknowns = 0
        for r in range(k_known, k_total):
            if np.any(np.round(best_x[:, r]) > 0):
                used_unknowns += 1

        bic = compute_bic(n, rss, k_known + used_unknowns)
    else:
        errors = np.full(n, float("inf"))
        rss = float("inf")
        n_good = 0
        n_bad = n
        bic = float("inf")
        used_unknowns = 0

    return {
        "x": best_x,
        "b": best_b,
        "common": best_common,
        "bic": bic,
        "rss": rss,
        "n_good": n_good,
        "n_bad": n_bad,
        "mean_error": float(np.mean(errors)) if len(errors) > 0 else float("inf"),
        "errors": errors,
        "converged": reached_goal,
        "used_unknowns": used_unknowns,
    }


# ---------------------------------------------------------------------------
# Biosynthetically parsimonious consensus  (BioConsensus)
# ---------------------------------------------------------------------------

def _build_bio_consensus(
    valid_models: dict,
    observations: np.ndarray,
    common_block: float,
    b: np.ndarray,
    final_names: list[str],
    k_total: int,
    final_tolerance: float,
):
    """Build a consensus that maximises biosynthetic network plausibility.

    Unlike the regular Consensus (which picks the simplest *model* per
    peak), BioConsensus picks the per-peak composition that is best
    connected to other peaks via single enzymatic steps (L₁ = 1
    transitions) and avoids extreme single-block usage.

    Algorithm
    ---------
    1. For each peak, collect every distinct composition from every model
       that explains it (error < tolerance).
    2. For each candidate composition, compute:
       a. **Clean connectivity** — how many *other* peaks have at least one
          candidate composition at L₁ = 1 distance.
       b. **Total copies** — overall parsimony (fewer building blocks
          = simpler, more likely correct).
       c. **Reconstruction error** — fit quality (tiebreaker).
    3. Rank candidates lexicographically:
       ``(connectivity↑, -total_copies↓, -error↓)``
       and pick the best.

    Returns
    -------
    consensus_x : np.ndarray, shape (n, k_total)
    consensus_errors : np.ndarray, shape (n,)
    """
    n = len(observations)

    # ------------------------------------------------------------------
    # Step 1: collect candidate compositions per peak (deduplicated)
    # ------------------------------------------------------------------
    # candidates[i] = [(comp_tuple, model_label, error), ...]
    candidates: list[list[tuple[tuple[int, ...], str, float]]] = [[] for _ in range(n)]
    seen: list[set[tuple[int, ...]]] = [set() for _ in range(n)]

    for label, res in valid_models.items():
        if res is None:
            continue
        for i in range(n):
            if res["errors"][i] < final_tolerance:
                comp = tuple(int(round(c)) for c in res["x_full"][i])
                if comp not in seen[i]:
                    seen[i].add(comp)
                    candidates[i].append((comp, label, float(res["errors"][i])))

    # Flat set of candidate compositions per peak (for fast lookups)
    comp_sets: list[set[tuple[int, ...]]] = [
        {c for c, _, _ in cands} for cands in candidates
    ]

    # ------------------------------------------------------------------
    # Step 2 & 3: score and select
    # ------------------------------------------------------------------
    def _clean_connectivity(comp_i: tuple[int, ...], peak_idx: int) -> int:
        """Count other peaks that have ≥1 candidate at L₁ = 1 distance."""
        count = 0
        for j in range(n):
            if j == peak_idx:
                continue
            for comp_j in comp_sets[j]:
                delta_l1 = sum(abs(a - b_) for a, b_ in zip(comp_i, comp_j))
                if delta_l1 == 1:
                    count += 1
                    break          # one neighbour is enough, move on
        return count

    consensus_x = np.zeros((n, k_total))
    consensus_errors = np.zeros(n)

    for i in range(n):
        if not candidates[i]:
            # No model explains this peak — leave as zeros
            recon = common_block
            consensus_errors[i] = abs(observations[i] - recon)
            continue

        best_comp: tuple[int, ...] | None = None
        best_key: tuple | None = None

        for comp, _label, error in candidates[i]:
            counts = list(comp)
            total_copies = sum(c for c in counts if c > 0)
            conn = _clean_connectivity(comp, i)

            # Lexicographic: higher connectivity, fewer total copies,
            # smaller error.
            key = (conn, -total_copies, -error)
            if best_key is None or key > best_key:
                best_key = key
                best_comp = comp

        if best_comp is not None:
            consensus_x[i] = list(best_comp)

        recon = common_block + sum(
            consensus_x[i, r] * b[r] for r in range(k_total)
        )
        consensus_errors[i] = abs(observations[i] - recon)

    return consensus_x, consensus_errors


# ---------------------------------------------------------------------------
# Enumerate ALL valid integer compositions for a single peak
# ---------------------------------------------------------------------------

def _enumerate_compositions_for_peak(
    target: float,
    b: np.ndarray,
    limits: list[int],
    tolerance: float,
    max_results: int = 500,
) -> list[tuple[tuple[int, ...], float]]:
    """Enumerate all integer compositions x ≥ 0 for a differential mass.

    Finds every non-negative integer vector *x* such that
    ``|target − x · b| < tolerance`` and ``x[j] ≤ limits[j]``.

    Uses recursive DFS with pruning (all block masses are positive).

    Parameters
    ----------
    target : float
        ``obs_i − common_block`` (the differential mass to explain).
    b : np.ndarray
        Block masses (all positive).
    limits : list[int]
        Per-block upper bounds.
    tolerance : float
        Maximum absolute reconstruction error.
    max_results : int
        Safety cap – stop early if too many results.

    Returns
    -------
    list of (composition_tuple, abs_error), sorted by ascending error.
    """
    k = len(b)
    results: list[tuple[tuple[int, ...], float]] = []

    # Pre-compute maximum attainable contribution from blocks idx..k-1
    max_from = np.zeros(k + 1)
    for j in range(k - 1, -1, -1):
        max_from[j] = max_from[j + 1] + limits[j] * b[j]

    def _dfs(idx: int, remaining: float, current: list[int]) -> None:
        if len(results) >= max_results:
            return
        if idx == k:
            if abs(remaining) < tolerance:
                results.append((tuple(current), abs(remaining)))
            return
        # Prune: remaining must be reachable
        if remaining < -tolerance:
            return  # blocks only add mass
        if remaining > max_from[idx] + tolerance:
            return  # can't reach even at maximum

        upper = min(limits[idx], int(remaining / b[idx]) + 1) if b[idx] > 0 else 0
        upper = min(upper, limits[idx])
        for count in range(upper + 1):
            new_remaining = remaining - count * b[idx]
            if new_remaining < -tolerance:
                break  # more of this block only worsens the fit
            current.append(count)
            _dfs(idx + 1, new_remaining, current)
            current.pop()

    _dfs(0, target, [])
    results.sort(key=lambda x: x[1])
    return results


# ---------------------------------------------------------------------------
# BioConsensus2 – SA-based biosynthetically optimal composition selection
# ---------------------------------------------------------------------------

def _build_bio_consensus2(
    observations: np.ndarray,
    common_block: float,
    b: np.ndarray,
    final_names: list[str],
    k_total: int,
    k_known: int,
    known_mass_limits: list[int],
    max_known: int,
    max_unknown: int,
    tolerance: float,
    final_tolerance: float,
):
    """Select per-peak compositions via simulated annealing.

    Unlike BioConsensus (which only sees the single solution Gurobi
    returned per model), BioConsensus2:

    1. **Enumerates** every valid integer composition per peak (across all
       blocks, not tied to a model subset).
    2. Uses **simulated annealing** to pick one composition per peak that
       maximises biosynthetic network connectivity.

    Connectivity scoring
    --------------------
    For each pair of selected compositions from different peaks, compute
    the difference vector *d*.  Only steps with L₁ ≤ 2 are considered
    (single or double enzymatic additions/subtractions).  The score is
    ``Σ_d freq(d)²`` where ``freq(d)`` is the number of peak-pairs that
    share step *d*.  Squaring strongly rewards *recurring* biosynthetic
    operations: 10 peak-pairs sharing "+1 dHex" contribute 100, whereas
    10 different step types contribute only 10.

    A small error-quality tiebreaker prefers lower reconstruction error.

    Returns
    -------
    consensus_x : np.ndarray, shape (n, k_total)
    consensus_errors : np.ndarray, shape (n,)
    all_peak_alternatives : list[list[tuple[tuple[int,...], float]]]
        Per-peak list of *all* valid (composition, error) pairs.
    """
    n = len(observations)
    rng = _random_mod.Random(42)  # reproducible

    # ---- per-block limits ----
    limits: list[int] = []
    for j in range(k_total):
        if j < k_known:
            lim = known_mass_limits[j] if j < len(known_mass_limits) else max_known
        else:
            lim = max_unknown
        limits.append(lim)

    # ---- Step 1: enumerate all valid compositions per peak ----
    all_peak_alternatives: list[list[tuple[tuple[int, ...], float]]] = []
    for i in range(n):
        target = observations[i] - common_block
        comps = _enumerate_compositions_for_peak(target, b, limits, final_tolerance)
        all_peak_alternatives.append(
            comps if comps else [(tuple(0 for _ in range(k_total)), float("inf"))]
        )

    n_with_alts = sum(1 for a in all_peak_alternatives if len(a) > 1)
    n_total = sum(len(a) for a in all_peak_alternatives)
    print(f"    Enumerated {n_total} compositions across {n} peaks "
          f"({n_with_alts} have multiple alternatives)")

    # ---- helper: canonical step vector (L₁ ≤ 2 or None) ----
    def _step_vector(comp_a: tuple, comp_b: tuple):
        d = tuple(cb - ca for ca, cb in zip(comp_a, comp_b))
        l1 = sum(abs(x) for x in d)
        if l1 < 1 or l1 > 2:
            return None
        d_neg = tuple(-x for x in d)
        return max(d, d_neg)  # canonical form

    # ---- early exit if nothing to optimise ----
    movable = [i for i in range(n) if len(all_peak_alternatives[i]) > 1]
    if not movable:
        consensus_x = np.zeros((n, k_total))
        consensus_errors = np.zeros(n)
        for i in range(n):
            consensus_x[i] = list(all_peak_alternatives[i][0][0])
            recon = common_block + float(
                np.dot(consensus_x[i], b)
            )
            consensus_errors[i] = abs(observations[i] - recon)
        print("    No peaks with multiple alternatives — skipping SA")
        return consensus_x, consensus_errors, all_peak_alternatives

    # ---- Step 2: simulated annealing ----
    # Initialise: lowest-error composition per peak
    state = [0] * n

    # Build initial step-frequency table
    step_freq: Counter = Counter()
    for i in range(n):
        ci = all_peak_alternatives[i][0][0]
        for j in range(i + 1, n):
            cj = all_peak_alternatives[j][0][0]
            sv = _step_vector(ci, cj)
            if sv is not None:
                step_freq[sv] += 1

    # Score = Σ_d freq(d)² + ERR_WEIGHT * Σ_i quality(i)
    ERR_WEIGHT = 0.1

    def _quality(peak_idx: int, alt_idx: int) -> float:
        err = all_peak_alternatives[peak_idx][alt_idx][1]
        if err >= final_tolerance:
            return -1.0
        return 1.0 - err / final_tolerance

    current_score = float(
        sum(f * f for f in step_freq.values())
        + ERR_WEIGHT * sum(_quality(i, state[i]) for i in range(n))
    )
    best_state = list(state)
    best_score = current_score

    # SA parameters — scale with problem size
    n_sa_steps = min(200_000, max(50_000, n_total * 200))
    T_init = max(10.0, best_score * 0.05) if best_score > 0 else 10.0
    T_min = 0.01
    alpha = (T_min / T_init) ** (1.0 / n_sa_steps)
    T = T_init

    print(f"    SA: {n_sa_steps} steps, T={T_init:.1f}→{T_min}, "
          f"{len(movable)} movable peaks")

    n_accepted = 0
    n_improved = 0

    for _step_num in range(n_sa_steps):
        peak = rng.choice(movable)
        n_alts = len(all_peak_alternatives[peak])
        new_idx = rng.randint(0, n_alts - 2)
        if new_idx >= state[peak]:
            new_idx += 1

        old_comp = all_peak_alternatives[peak][state[peak]][0]
        new_comp = all_peak_alternatives[peak][new_idx][0]

        # Net step-frequency changes from switching this peak
        net_changes: dict[tuple, int] = defaultdict(int)
        for j in range(n):
            if j == peak:
                continue
            cj = all_peak_alternatives[j][state[j]][0]
            sv_old = _step_vector(old_comp, cj)
            sv_new = _step_vector(new_comp, cj)
            if sv_old == sv_new:
                continue
            if sv_old is not None:
                net_changes[sv_old] -= 1
            if sv_new is not None:
                net_changes[sv_new] += 1

        # Score delta
        delta = ERR_WEIGHT * (
            _quality(peak, new_idx) - _quality(peak, state[peak])
        )
        for sv, dc in net_changes.items():
            old_f = step_freq.get(sv, 0)
            new_f = old_f + dc
            delta += new_f * new_f - old_f * old_f

        # Metropolis acceptance
        if delta > 0 or (T > 0 and rng.random() < math.exp(delta / T)):
            state[peak] = new_idx
            for sv, dc in net_changes.items():
                step_freq[sv] += dc
                if step_freq[sv] == 0:
                    del step_freq[sv]
            current_score += delta
            n_accepted += 1
            if current_score > best_score:
                best_state = list(state)
                best_score = current_score
                n_improved += 1

        T *= alpha

    print(f"    SA done: {n_accepted} accepted, {n_improved} improvements, "
          f"score {best_score:.1f}")

    # ---- Build result from best_state ----
    consensus_x = np.zeros((n, k_total))
    consensus_errors = np.zeros(n)
    for i in range(n):
        comp = all_peak_alternatives[i][best_state[i]][0]
        consensus_x[i] = list(comp)
        recon = common_block + float(np.dot(consensus_x[i], b))
        consensus_errors[i] = abs(observations[i] - recon)

    # ---- Report top biosynthetic steps ----
    final_freq: Counter = Counter()
    for i in range(n):
        ci = all_peak_alternatives[i][best_state[i]][0]
        for j in range(i + 1, n):
            cj = all_peak_alternatives[j][best_state[j]][0]
            sv = _step_vector(ci, cj)
            if sv is not None:
                final_freq[sv] += 1

    if final_freq:
        print("    Top biosynthetic steps:")
        for sv, count in final_freq.most_common(10):
            parts = []
            for r in range(k_total):
                if sv[r] > 0:
                    parts.append(f"+{sv[r]}{final_names[r]}")
                elif sv[r] < 0:
                    parts.append(f"{sv[r]}{final_names[r]}")
            step_str = " ".join(parts) if parts else "(none)"
            print(f"      {step_str}: {count} pair(s)")

    # ---- Per-peak alternative scoring ----
    # For each peak with multiple alternatives, compute the connectivity
    # score that each alternative would achieve in the current best_state
    # context (i.e., how many L1/L2 neighbours each alternative has with
    # the selected compositions of all other peaks).
    print("\n    Per-peak alternative scores (connectivity + quality):")
    n_ambiguous = 0
    n_clear_winner = 0
    margin_ratios = []  # score margin between #1 and #2

    for i in range(n):
        alts = all_peak_alternatives[i]
        if len(alts) <= 1:
            continue

        # Score each alternative for this peak against the chosen
        # compositions of all other peaks.
        alt_scores = []
        for ai, (comp_i, err_i) in enumerate(alts):
            conn = 0
            for j in range(n):
                if j == i:
                    continue
                cj = all_peak_alternatives[j][best_state[j]][0]
                sv = _step_vector(comp_i, cj)
                if sv is not None:
                    conn += 1
            qual = (1.0 - err_i / final_tolerance) if err_i < final_tolerance else -1.0
            total = conn + ERR_WEIGHT * qual
            alt_scores.append((ai, comp_i, err_i, conn, qual, total))

        # Sort by descending total score
        alt_scores.sort(key=lambda x: -x[5])
        chosen_ai = best_state[i]

        # Determine margin between best and runner-up
        if len(alt_scores) >= 2:
            margin = alt_scores[0][5] - alt_scores[1][5]
            margin_ratios.append(margin)
            if margin < 0.5:
                n_ambiguous += 1
            else:
                n_clear_winner += 1
        else:
            n_clear_winner += 1

        # Only print details for peaks with >1 alternative
        chosen_comp = alts[chosen_ai][0]
        chosen_parts = [f"{c}{final_names[r]}"
                        for r, c in enumerate(chosen_comp) if c > 0]
        print(f"    Peak {i+1} ({observations[i]:.3f}) — "
              f"{len(alts)} alternatives, "
              f"chose: {' + '.join(chosen_parts) if chosen_parts else '(empty)'}")
        for ai, comp_i, err_i, conn, qual, total in alt_scores[:5]:
            parts = [f"{c}{final_names[r]}"
                     for r, c in enumerate(comp_i) if c > 0]
            tag = " <-- selected" if ai == chosen_ai else ""
            print(f"      {' + '.join(parts) if parts else '(empty)':40s}  "
                  f"conn={conn:2d}  qual={qual:+.2f}  "
                  f"score={total:.2f}  err={err_i:.4f}{tag}")
        if len(alt_scores) > 5:
            print(f"      ... and {len(alt_scores) - 5} more alternatives")

    # Summary
    n_multi = sum(1 for a in all_peak_alternatives if len(a) > 1)
    print(f"\n    Score summary: {n_multi} peaks with multiple alternatives")
    if margin_ratios:
        print(f"      Clear winner (margin >= 0.5): {n_clear_winner}")
        print(f"      Ambiguous    (margin <  0.5): {n_ambiguous}")
        print(f"      Margin: min={min(margin_ratios):.2f}, "
              f"median={sorted(margin_ratios)[len(margin_ratios)//2]:.2f}, "
              f"max={max(margin_ratios):.2f}")

    return consensus_x, consensus_errors, all_peak_alternatives


# ---------------------------------------------------------------------------
# Exhaustive model comparison – test every non-empty subset of blocks
# ---------------------------------------------------------------------------

def _run_exhaustive_comparison(
    observations,
    common_block,
    b,
    final_names,
    k_known,
    k_total,
    known_mass_limits,
    max_known,
    max_unknown,
    tolerance,
    final_tolerance,
    exhaustive_level=2,
    should_cancel=None,
):
    """Test non-empty subsets of the available blocks.

    Parameters
    ----------
    exhaustive_level : int
        1 = only subsets that include a block named 'Hex' (faster),
        2 = all subsets.

    Returns
    -------
    exhaustive_results : dict
        ``{model_label: result_dict}``  (value is None on failure).
    peak_stats : dict
        ``{peak_idx: {n_models_tested, n_explaining, consistent}}``.
    """
    from itertools import combinations

    n = len(observations)
    all_indices = list(range(k_total))

    # Find index of Hex block (exact match, case-insensitive)
    hex_idx: int | None = None
    for idx, nm in enumerate(final_names):
        if nm.lower() == "hex":
            hex_idx = idx
            break

    subsets: list[list[int]] = []
    for r in range(1, k_total + 1):
        for combo in combinations(all_indices, r):
            subsets.append(list(combo))

    # Level 1: keep only subsets that include Hex
    if exhaustive_level == 1 and hex_idx is not None:
        subsets = [s for s in subsets if hex_idx in s]
        print("  (exhaustive level 1: restricted to subsets containing Hex)")

    # Sort: fewest blocks first, then alphabetically by block names
    subsets.sort(key=lambda s: (len(s), [final_names[i] for i in s]))
    n_models = len(subsets)
    print(f"  Testing {n_models} block combinations")
    if n_models > 128:
        print("  (this may take a while)")

    exhaustive_results: dict[str, dict | None] = {}
    # Track subsets that achieved a perfect fit (n_bad == 0).
    # Any superset of a perfect-fit subset will have equal or worse BIC
    # (same/lower RSS but strictly more parameters), so it can be skipped.
    perfect_subsets: list[set[int]] = []

    for si, subset in enumerate(subsets):
        if should_cancel and should_cancel():
            raise SolverCancelledError("Analysis stopped by user request.")

        # ---- early skip: superset of a perfect-fit model ----
        subset_set = set(subset)
        if any(ps.issubset(subset_set) for ps in perfect_subsets):
            bnames_sub = [final_names[i] for i in subset]
            model_label = "+".join(bnames_sub)
            print(f"  [{si + 1}/{n_models}] {model_label} ... "
                  f"skipped (superset of perfect-fit model)")
            continue

        bnames_sub = [final_names[i] for i in subset]
        model_label = "+".join(bnames_sub)
        b_sub = b[np.array(subset)]
        m = len(subset)

        print(f"  [{si + 1}/{n_models}] {model_label} ...", end=" ", flush=True)

        y_sub = observations - common_block
        x_m = cp.Variable((n, m), integer=True)
        s_m = cp.Variable(n, nonneg=True)
        error_m = x_m @ b_sub - y_sub
        cons_m: list = [x_m >= 0]
        for ri, bi in enumerate(subset):
            lim = (
                known_mass_limits[bi]
                if bi < len(known_mass_limits)
                else (max_known if bi < k_known else max_unknown)
            )
            cons_m += [x_m[:, ri] <= lim]
        cons_m += [
            error_m <= tolerance + s_m,
            -error_m <= tolerance + s_m,
        ]
        obj_m = cp.sum_squares(error_m) + 1e3 * cp.sum(s_m)
        prob_m = cp.Problem(cp.Minimize(obj_m), cons_m)

        try:
            prob_m.solve(solver=cp.GUROBI, TimeLimit=120)
            if (
                prob_m.status in ("optimal", "optimal_inaccurate")
                and x_m.value is not None
            ):
                x_val_m = x_m.value
                recon_m = common_block + np.sum(x_val_m * b_sub, axis=1)
                errors_m = np.abs(observations - recon_m)
                rss_m = float(np.sum(errors_m ** 2))
                n_good_m = int(np.sum(errors_m < final_tolerance))
                n_bad_m = n - n_good_m
                bic_m = compute_bic(n, rss_m, m)

                # Pad x to full block set
                x_full = np.zeros((n, k_total))
                for ri, bi in enumerate(subset):
                    x_full[:, bi] = x_val_m[:, ri]

                exhaustive_results[model_label] = {
                    "x": x_val_m,
                    "x_full": x_full,
                    "errors": errors_m,
                    "bic": bic_m,
                    "n_good": n_good_m,
                    "n_bad": n_bad_m,
                    "mean_error": float(np.mean(errors_m)),
                    "blocks_used": bnames_sub,
                    "block_indices": subset,
                    "n_blocks": m,
                }
                print(f"good={n_good_m}/{n}, BIC={bic_m:.1f}")

                # Record perfect fit for superset pruning
                if n_bad_m == 0:
                    perfect_subsets.append(set(subset))
            else:
                exhaustive_results[model_label] = None
                print("failed")
        except Exception as e:
            exhaustive_results[model_label] = None
            print(f"error: {e}")

    # ---- per-peak statistics ----
    valid = {k: v for k, v in exhaustive_results.items() if v is not None}
    n_models_tested = len(valid)
    peak_stats: dict[int, dict] = {}

    for i in range(n):
        explaining_comps: list[tuple[int, ...]] = []
        explaining_models: list[tuple[str, dict]] = []   # (label, res)
        for _label, res in valid.items():
            if res["errors"][i] < final_tolerance:
                comp = tuple(int(round(c)) for c in res["x_full"][i])
                explaining_comps.append(comp)
                explaining_models.append((_label, res))

        n_exp = len(explaining_comps)
        consistent = len(set(explaining_comps)) <= 1 if n_exp > 0 else True

        # ---- best (simplest) explaining model for this peak ----
        # Prefer: 1) fewest blocks in the model,
        #         2) fewer total block copies in this peak's composition
        #            (fewer copies ⇒ larger blocks were used),
        #         3) higher minimum block mass among blocks used
        #            (prefer larger blocks).
        best_model_label: str | None = None
        best_formula: str | None = None
        if explaining_models:
            def _peak_complexity(item):
                label, res = item
                x_row = res["x_full"][i]
                counts = [int(round(c)) for c in x_row]
                total_copies = sum(c for c in counts if c > 0)
                # Minimum mass among blocks actually used — larger is better
                used_masses = [
                    b[j] for j, c in enumerate(counts) if c > 0
                ]
                min_mass = min(used_masses) if used_masses else 0.0
                return (res["n_blocks"], total_copies, -min_mass)

            explaining_models.sort(key=_peak_complexity)
            best_label, best_res = explaining_models[0]
            best_model_label = best_label

            x_row = best_res["x_full"][i]
            best_comp = [int(round(c)) for c in x_row]
            parts = []
            for r in range(k_total):
                if best_comp[r] > 0:
                    parts.append(f"{best_comp[r]}{final_names[r]}")
            best_formula = " + ".join(parts) if parts else "(empty)"
        else:
            best_comp = None

        peak_stats[i] = {
            "n_models_tested": n_models_tested,
            "n_explaining": n_exp,
            "consistent": consistent,
            "best_model": best_model_label,
            "best_formula": best_formula,
            "best_composition": best_comp,
        }

    return exhaustive_results, peak_stats


# ---------------------------------------------------------------------------
# Main progressive solver
# ---------------------------------------------------------------------------

def solve_progressive(
    peaks,
    output,
    mode,
    matrix,
    common,
    names,
    unknown,          # now means "up to" this many unknowns
    tolerance,
    final_tolerance,
    bad,
    max_known,
    max_unknown,
    lower_bound,
    upper_bound,
    min_diff,
    candidates_only,
    blocks_dict,
    postgoal,
    timeout,
    verbose,
    masses=None,
    exclude=None,
    protect=None,
    exhaustive=1,
    common_composition=None,
    should_cancel=None,
    glycan_type=None,
):
    """
    Progressive glycan solver — builds the model incrementally.

    Parameters
    ----------
    exhaustive : int
        0 = no exhaustive comparison (nested only),
        1 = test all block combinations that include Hex (default),
        2 = test all block combinations.

    Phases
    ------
    1. Generate candidate blocks from pairwise peak differences.
    2. Fit known blocks only (baseline BIC).
    3. For each candidate, evaluate benefit via quick heuristic; then run
       short optimizations for the top candidates; add the best if BIC
       improves.  Repeat up to *unknown* times.
    4. Final full-length refinement with the selected block set.
    """

    # ================================================================
    # SETUP
    # ================================================================

    iterations = 500  # fixed internal budget (early-exit logic handles convergence)

    # Print equivalent CLI command for reproducibility / debugging
    cmd_parts = ["glycansolver"]
    cmd_parts.append(f"-p {peaks}")
    cmd_parts.append(f"-o {output}")
    cmd_parts.append(f"-D {mode}")
    cmd_parts.append(f"-X {matrix}")
    cmd_parts.append(f"-c {common}")
    if names:
        cmd_parts.append(f'-n "{names}"')
    cmd_parts.append(f"-u {unknown}")
    cmd_parts.append(f"-t {tolerance}")
    cmd_parts.append(f"-f {final_tolerance}")
    cmd_parts.append(f"-b {bad}")
    cmd_parts.append(f"-K {max_known}")
    cmd_parts.append(f"-U {max_unknown}")
    cmd_parts.append(f"-L {lower_bound}")
    cmd_parts.append(f"-M {upper_bound}")
    cmd_parts.append(f"-I {min_diff}")
    cmd_parts.append(f"-A {postgoal}")
    cmd_parts.append(f"-T {timeout}")
    if blocks_dict:
        cmd_parts.append(f"-d {blocks_dict}")
    if exclude:
        cmd_parts.append(f'-e "{exclude}"')
    cmd_parts.append(f"--exhaustive {exhaustive}")
    if glycan_type:
        cmd_parts.append(f"--glycan-type {glycan_type}")
    if verbose:
        cmd_parts.append("-v")
    if candidates_only:
        cmd_parts.append("-C")
    print("[CLI equivalent] " + " ".join(cmd_parts))
    print()

    if should_cancel and should_cancel():
        raise SolverCancelledError("Analysis stopped by user request.")

    ensure_output_directory(output)

    # Blocks dictionary
    blocks_dict_path = None
    if blocks_dict and os.path.exists(blocks_dict):
        print(f"Using blocks dictionary: {os.path.basename(blocks_dict)}")
        blocks_dict_path = blocks_dict
    elif blocks_dict:
        print(f"Warning: blocks dictionary not found: {os.path.basename(blocks_dict)}")

    # Load observations
    print(f"Loading peaks from: {peaks}")
    observations = load_peaks(peaks)
    n = len(observations)
    print(f"Loaded {n} peaks")

    if should_cancel and should_cancel():
        raise SolverCancelledError("Analysis stopped by user request.")

    # ---- Load blocks dictionary early so names can resolve to masses ----
    bd = load_blocks_dictionary(blocks_dict_path, glycan_type=glycan_type) if blocks_dict_path else {}
    # Build a name→(mass, limit) lookup from the dictionary
    _name_to_mass: dict[str, float] = {}
    for m, nm in bd.items():
        _name_to_mass[nm.lower()] = m

    # ---- known blocks ----
    known_masses = []
    known_mass_limits = []
    if masses:
        for mass_str in masses.split(","):
            try:
                if ":" in mass_str:
                    mp, lp = mass_str.strip().split(":")
                    known_masses.append(float(mp))
                    known_mass_limits.append(int(lp))
                else:
                    known_masses.append(float(mass_str.strip()))
                    known_mass_limits.append(max_known)
            except ValueError:
                print(f"Warning: invalid mass '{mass_str}', using 162.052824")
                known_masses.append(162.052824)
                known_mass_limits.append(max_known)

    # ---- known block names ----
    # When more names than masses are given, resolve extra names from
    # the blocks dictionary so that e.g. -n "Hex,HexNAc,dHex,NeuAc"
    # automatically adds the corresponding masses.
    if names:
        raw_name_tokens = [n.strip() for n in names.split(",")]
        known_block_names: list[str] | None = []
        # Expand masses to match names using dictionary lookup.
        # Supports Name:limit syntax, e.g. "HexNAc:4".
        for i, token in enumerate(raw_name_tokens):
            # Parse optional :limit suffix
            if ":" in token:
                nm_part, lim_part = token.split(":", 1)
                nm_part = nm_part.strip()
                try:
                    limit = int(lim_part)
                except ValueError:
                    limit = max_known
            else:
                nm_part = token
                limit = max_known

            known_block_names.append(nm_part)

            # If this index already has a mass (from -m), just override
            # the limit when the user specified one via the name token.
            if i < len(known_masses):
                if ":" in token:  # explicit limit given
                    known_mass_limits[i] = limit
                continue

            # Otherwise resolve the name from the dictionary
            nm_lower = nm_part.lower()
            if nm_lower in _name_to_mass:
                known_masses.append(_name_to_mass[nm_lower])
                known_mass_limits.append(limit)
                print(
                    f"Resolved '{nm_part}' from dictionary: "
                    f"{_name_to_mass[nm_lower]:.6f}"
                    + (f" (max {limit})" if limit != max_known else "")
                )
            else:
                print(
                    f"Warning: cannot resolve '{nm_part}' "
                    f"from blocks dictionary — skipping"
                )
        # Trim names to match masses
        known_block_names = known_block_names[:len(known_masses)]
    else:
        known_block_names = None

    if not known_masses:
        known_masses = [162.052824]
        known_mass_limits = [max_known]

    k_known = len(known_masses)
    known_differential = np.array(known_masses)

    if known_block_names is None:
        known_block_names = [f"Known_{i+1}" for i in range(k_known)]
    while len(known_block_names) < k_known:
        known_block_names.append(f"Known_{len(known_block_names)+1}")

    print(
        "Known blocks: "
        + ", ".join(
            f"{nm}={m:.4f}" + (f" (max {lim})" if lim != max_known else "")
            for nm, m, lim in zip(known_block_names, known_masses, known_mass_limits)
        )
    )

    # ---- excluded blocks ----
    excluded_masses = []
    excluded_names = []
    if exclude:
        # Reuse the already-loaded blocks dictionary (bd)
        for exc in exclude.split(","):
            exc = exc.strip()
            found = False
            for mass, name in bd.items():
                if name.lower() == exc.lower():
                    excluded_masses.append(mass)
                    excluded_names.append(name)
                    found = True
                    break
            if not found:
                try:
                    excluded_masses.append(float(exc))
                    excluded_names.append(exc)
                except ValueError:
                    print(f"Warning: cannot resolve excluded block '{exc}'")
        if excluded_masses:
            print(
                "Excluded blocks: "
                + ", ".join(
                    f"{nm}={m:.4f}" for nm, m in zip(excluded_names, excluded_masses)
                )
            )

    max_unknown_blocks = unknown  # "up to" semantics
    print(f"Will try up to {max_unknown_blocks} unknown blocks (progressive)")

    # ---- common block ----
    if common < 0:
        common_block = float(np.min(observations))
        common_block_fixed = False
    else:
        common_block = float(common)
        common_block_fixed = True

    # Derive a human-readable name for the common block
    common_name = common_composition_name(common_composition)
    print(
        f"Common block: {common_block:.4f} ({common_name})"
        f" ({'fixed' if common_block_fixed else 'estimated'})"
    )

    # ---- timeout ----
    _start_time = time.time()  # noqa: F841 (reserved for future global timeout)
    timeout_seconds = timeout * 60 if timeout > 0 else None

    # ================================================================
    # GENERATE CANDIDATES
    # ================================================================
    # We ask for a modest number of initial blocks just to seed the
    # candidate list.  The progressive loop evaluates them one-by-one.
    candidate_k = max_unknown_blocks   # only need this many "selected"

    (
        _initial_unknowns,
        _treat_first,
        all_candidate_values,
        ranked_clusters,
        filtered_candidates,
    ) = get_smart_block_init(
        observations,
        common_block,
        known_differential,
        candidate_k,
        min_diff=min_diff,
        verbose=verbose,
        blocks_dict_path=blocks_dict_path,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        glycan_type=glycan_type,
    )

    write_candidates_tsv(ranked_clusters, filtered_candidates, output)

    if candidates_only:
        print("Stopping after candidate generation (--candidates-only).")
        return

    # ---- resolve protected masses (user-selected candidates) ----
    protected_masses = []
    if protect:
        for token in protect:
            token = token.strip()
            if not token:
                continue
            # Try to resolve name from blocks dictionary
            resolved = False
            for mass, name in bd.items():
                if name.lower() == token.lower():
                    protected_masses.append(mass)
                    resolved = True
                    break
            if not resolved:
                try:
                    protected_masses.append(float(token))
                except ValueError:
                    pass
        if protected_masses:
            print(f"Protected candidates (will not be excluded): "
                  f"{', '.join(f'{m:.4f}' for m in protected_masses)}")

    # ---- apply exclusion filter ----
    if excluded_masses:
        min_gap = 0.5
        kept = []
        for val in all_candidate_values:
            # Never exclude a protected (user-selected) candidate
            if any(abs(val - pm) < min_gap for pm in protected_masses):
                kept.append(val)
                continue
            dominated = False
            for exc in excluded_masses:
                if abs(val - exc) < min_gap:
                    dominated = True
                    break
                for mult in range(2, 6):
                    if abs(val - exc * mult) < min_gap:
                        dominated = True
                        break
                if dominated:
                    break
                # combinations with known blocks
                if not dominated:
                    for km in known_masses:
                        for c1 in range(4):
                            for c2 in range(4):
                                if c1 == 0 and c2 == 0:
                                    continue
                                if abs(val - (c1 * km + c2 * exc)) < min_gap:
                                    dominated = True
                                    break
                            if dominated:
                                break
                        if dominated:
                            break
            if not dominated:
                kept.append(val)
            elif verbose:
                print(f"  Excluded candidate {val:.4f} (matches excluded block)")
        removed = len(all_candidate_values) - len(kept)
        if removed:
            print(f"Exclusion filter removed {removed} candidates")
        all_candidate_values = kept

    # Load dictionary for naming AND categories
    blocks_name_dict = {}
    blocks_cat_dict = {}
    if blocks_dict_path:
        blocks_name_dict_raw, blocks_cat_dict = load_blocks_dictionary_with_categories(
            blocks_dict_path, glycan_type=glycan_type
        )
        # Also keep the simple mass→name mapping
        blocks_name_dict = blocks_name_dict_raw

    def _name_for_mass(mass, fallback="?"):
        for m, nm in blocks_name_dict.items():
            if abs(mass - m) < 0.5:
                return nm
        return fallback

    def _category_for_mass(mass):
        """Return category for a candidate mass: 'common', 'rare', 'mod', or 'unknown'."""
        for m, cat in blocks_cat_dict.items():
            if abs(mass - m) < 0.5:
                return cat
        return "unknown"

    print(f"\nCandidate blocks for progressive fitting: {len(all_candidate_values)}")
    for i, val in enumerate(all_candidate_values[:20]):
        nm = _name_for_mass(val, "")
        cat = _category_for_mass(val)
        label = f" ({nm})" if nm else ""
        print(f"  {i+1}. {val:.4f}{label}  [{cat}]")

    # ================================================================
    # PHASE 1 – BASELINE (known blocks only)
    # ================================================================
    print(f"\n{'='*60}")
    print("PHASE 1: Baseline fit — known blocks only")
    print(f"{'='*60}")

    # Baseline is simple (known blocks only), converges fast
    phase_iters = min(iterations, 40)
    # Give baseline at most 40% of total timeout
    baseline_timeout = timeout_seconds * 0.40 if timeout_seconds else None
    baseline_start = time.time()

    baseline = run_phase(
        observations,
        common_block,
        known_differential,
        np.ones(k_known, dtype=bool),
        known_mass_limits,
        max_known,
        max_unknown,
        tolerance,
        final_tolerance,
        phase_iters,
        lower_bound,
        upper_bound,
        bad_allowed=n,           # don't force all in baseline
        postgoal=min(postgoal, 5),
        common_block_fixed=common_block_fixed,
        timeout_seconds=baseline_timeout,
        start_time=baseline_start,
        verbose=verbose,
        should_cancel=should_cancel,
    )

    print(
        f"\nBaseline result: BIC={baseline['bic']:.1f}, "
        f"good={baseline['n_good']}/{n}, bad={baseline['n_bad']}, "
        f"mean_error={baseline['mean_error']:.4f}"
    )

    best_model = baseline
    selected_unknowns = []          # list of (mass, name)
    used_candidate_indices = set()
    current_common = baseline["common"]

    # ================================================================
    # PHASE 2 – PROGRESSIVE BLOCK ADDITION
    # ================================================================
    for step in range(max_unknown_blocks):
        if should_cancel and should_cancel():
            raise SolverCancelledError("Analysis stopped by user request.")

        print(f"\n{'='*60}")
        print(
            f"PHASE 2.{step+1}: Evaluating candidate unknown block "
            f"{step+1}/{max_unknown_blocks}"
        )
        print(f"{'='*60}")

        # ---- select top candidates regardless of category ----
        available = []
        for idx, cand_mass in enumerate(all_candidate_values):
            if idx in used_candidate_indices:
                continue
            too_close = any(
                abs(cand_mass - sm) < 0.5 for sm, _ in selected_unknowns
            )
            # also skip if too close to a known block
            if not too_close:
                for km in known_masses:
                    if abs(cand_mass - km) < 0.5:
                        too_close = True
                        break
            if too_close:
                continue
            available.append((idx, cand_mass))

        if not available:
            print("No more candidates available.")
            break

        # Take the top-N from the full ranked list (candidates that
        # match a known dictionary block already have a score boost
        # from get_smart_block_init).
        top_n = min(5, len(available))
        top_candidates = [(idx, mass, 0) for idx, mass in available[:top_n]]

        print(f"Top {top_n} candidates:")
        for idx, mass, _ in top_candidates:
            label = _name_for_mass(mass, "")
            cat = _category_for_mass(mass)
            tag = f" ({label})" if label else ""
            print(f"  {mass:.4f}{tag}  [{cat}]")

        # ---- full evaluation of top candidates ----
        trial_results = []
        current_unknown_masses = np.array([m for m, _ in selected_unknowns])

        # Build a warm-start x from the current best model, extended
        # with a zero column for the new candidate block.
        prev_x = best_model["x"]
        if prev_x is not None:
            trial_warm_x = np.hstack([prev_x, np.zeros((n, 1))])
        else:
            trial_warm_x = None

        for idx, cand_mass, _qscore in top_candidates:
            trial_unknown = (
                np.append(current_unknown_masses, cand_mass)
                if len(current_unknown_masses) > 0
                else np.array([cand_mass])
            )
            all_masses = np.concatenate([known_differential, trial_unknown])
            is_known = np.array(
                [True] * k_known + [False] * len(trial_unknown)
            )

            # Short trial: just enough to see if the candidate helps
            trial_iters = min(iterations, 15)

            if verbose:
                print(f"\n  Trial: adding {cand_mass:.4f} …")

            # Each trial gets its own dedicated time budget
            trial_start = time.time()
            trial_timeout = 600  # 10 minutes per candidate trial

            result = run_phase(
                observations,
                current_common,
                all_masses,
                is_known,
                known_mass_limits,
                max_known,
                max_unknown,
                tolerance,
                final_tolerance,
                trial_iters,
                lower_bound,
                upper_bound,
                bad_allowed=bad,
                postgoal=min(postgoal, 5),
                common_block_fixed=common_block_fixed,
                timeout_seconds=trial_timeout,
                start_time=trial_start,
                verbose=verbose,
                initial_x=trial_warm_x,
                should_cancel=should_cancel,
            )

            trial_results.append((idx, cand_mass, result))
            nm = _name_for_mass(cand_mass, "?")
            print(
                f"  {cand_mass:.4f} ({nm}): BIC={result['bic']:.1f}, "
                f"good={result['n_good']}/{n}, bad={result['n_bad']}, "
                f"mean_err={result['mean_error']:.4f}"
            )

        if not trial_results:
            print("No trials completed (timeout?).")
            break

        # select best by BIC, break ties by n_good
        trial_results.sort(key=lambda t: (t[2]["bic"], -t[2]["n_good"]))
        best_idx, best_mass, best_result = trial_results[0]

        bic_improvement = best_model["bic"] - best_result["bic"]
        peaks_improvement = best_result["n_good"] - best_model["n_good"]

        # resolve name using refined mass from optimisation
        refined_mass = best_mass
        if best_result["b"] is not None:
            r_idx = k_known + len(selected_unknowns)
            if r_idx < len(best_result["b"]):
                refined_mass = best_result["b"][r_idx]
        block_name = _name_for_mass(refined_mass, f"Unknown_{step+1}")

        # Check whether the candidate block is actually used in the solution
        candidate_col = k_known + len(selected_unknowns)  # column index of the new block
        candidate_used = (
            best_result["x"] is not None
            and candidate_col < best_result["x"].shape[1]
            and np.any(np.round(best_result["x"][:, candidate_col]) > 0)
        )

        print(f"\nBest candidate this round: {best_mass:.4f} → refined {refined_mass:.4f} ({block_name})")
        print(f"  BIC change : {bic_improvement:+.1f} ({'better' if bic_improvement > 0 else 'worse/equal'})")
        print(f"  Peaks gained: {peaks_improvement:+d}")
        print(f"  Block used : {'yes' if candidate_used else 'no'}")

        # ---- decision: add this block? ----
        # Minimum BIC improvement threshold to guard against floating-point noise
        MIN_BIC_DELTA = 2.0
        # Accept if: (a) candidate block is actually used in the solution, AND
        #            (b) BIC meaningfully improves AND ≥1 new peak  OR  ≥2 more peaks explained
        prev_n_good = best_model["n_good"]
        if candidate_used and ((bic_improvement > MIN_BIC_DELTA and peaks_improvement >= 1) or peaks_improvement >= 2):
            used_candidate_indices.add(best_idx)
            selected_unknowns.append((refined_mass, block_name))
            best_model = best_result
            current_common = best_result["common"]
            print(f"  ✓ ACCEPTED — model now has {k_known + len(selected_unknowns)} blocks")
            print(
                f"    Progress: {prev_n_good} → {best_result['n_good']} "
                f"good peaks"
            )
        else:
            reasons = []
            if not candidate_used:
                reasons.append("block not used in solution")
            if peaks_improvement < 1:
                reasons.append(f"no new peaks explained (peaks Δ={peaks_improvement:+d})")
            elif bic_improvement <= MIN_BIC_DELTA and peaks_improvement < 2:
                reasons.append(f"insufficient improvement (BIC Δ={bic_improvement:+.1f}, peaks Δ={peaks_improvement:+d})")
            print(f"  ✗ REJECTED — {'; '.join(reasons)}. Stopping progressive search.")
            break

    # ================================================================
    # PHASE 3a – PENULTIMATE REFINEMENT (all peaks)
    # ================================================================
    k_unknown_final = len(selected_unknowns)
    print(f"\n{'='*60}")
    print(
        f"PHASE 3a: Refinement (all peaks) with {k_known} known + "
        f"{k_unknown_final} discovered blocks"
    )
    print(f"{'='*60}")

    if k_unknown_final > 0:
        final_unknown_masses = np.array([m for m, _ in selected_unknowns])
    else:
        final_unknown_masses = np.array([])

    final_all_masses = (
        np.concatenate([known_differential, final_unknown_masses])
        if len(final_unknown_masses) > 0
        else known_differential.copy()
    )
    final_is_known = np.array(
        [True] * k_known + [False] * k_unknown_final
    )

    # Initialise from best model's refined masses where available
    if (
        best_model["b"] is not None
        and len(best_model["b"]) == len(final_all_masses)
    ):
        final_all_masses = best_model["b"].copy()

    # Phase 3a gets its own time budget (not limited by global timeout)
    phase3a_start = time.time()
    phase3a_timeout = 600  # 10 minutes dedicated to all-peaks refinement

    # Warm-start Phase 3a from the best x found in Phase 2
    phase3a_initial_x = best_model["x"] if best_model["x"] is not None else None

    final_result = run_phase(
        observations,
        current_common,
        final_all_masses,
        final_is_known,
        known_mass_limits,
        max_known,
        max_unknown,
        tolerance,
        final_tolerance,
        iterations,           # full iterations
        lower_bound,
        upper_bound,
        bad_allowed=bad,
        postgoal=postgoal,
        common_block_fixed=common_block_fixed,
        timeout_seconds=phase3a_timeout,
        start_time=phase3a_start,
        verbose=verbose,
        initial_x=phase3a_initial_x,
        should_cancel=should_cancel,
    )

    # Use whichever is better
    if final_result["bic"] <= best_model["bic"]:
        best_model = final_result
    elif best_model["b"] is not None and len(best_model["b"]) == len(final_all_masses):
        pass   # keep previous best_model
    else:
        best_model = final_result

    # ================================================================
    # PHASE 3b – FINAL REFINEMENT (good peaks only)
    # ================================================================
    # Re-fit block masses using only the peaks that were "good" in 3a.
    # This prevents bad/unexplained peaks from skewing the refined masses.
    phase3a_errors = best_model["errors"]
    good_mask = phase3a_errors < final_tolerance
    n_good_3a = int(np.sum(good_mask))

    if n_good_3a >= max(k_known + k_unknown_final, 3):
        print(f"\n{'='*60}")
        print(
            f"PHASE 3b: Final refinement (good peaks only, {n_good_3a}/{n})"
        )
        print(f"{'='*60}")

        good_obs = observations[good_mask]

        # Start from the masses refined in 3a
        refined_masses = best_model["b"].copy() if best_model["b"] is not None else final_all_masses.copy()
        refined_common = best_model["common"]

        # Phase 3b gets its own time budget (not limited by global timeout)
        phase3b_start = time.time()
        phase3b_timeout = 600  # 10 minutes dedicated to good-peaks refinement

        # Warm-start Phase 3b from best model x (filtered to good peaks)
        phase3b_initial_x = None
        if best_model["x"] is not None:
            phase3b_initial_x = best_model["x"][good_mask]

        refined_result = run_phase(
            good_obs,
            refined_common,
            refined_masses,
            final_is_known,
            known_mass_limits,
            max_known,
            max_unknown,
            tolerance,
            final_tolerance,
            iterations,
            lower_bound,
            upper_bound,
            bad_allowed=0,
            postgoal=postgoal,
            common_block_fixed=common_block_fixed,
            timeout_seconds=phase3b_timeout,
            start_time=phase3b_start,
            verbose=verbose,
            initial_x=phase3b_initial_x,
            should_cancel=should_cancel,
        )

        if refined_result["b"] is not None:
            b_refined = refined_result["b"]
            common_refined = refined_result["common"]

            # ---- Re-solve x for ALL peaks using the refined block masses ----
            k_total = len(b_refined)
            y_all = observations - common_refined
            x_all = cp.Variable((n, k_total), integer=True)
            s_all = cp.Variable(n, nonneg=True)
            error_all = x_all @ b_refined - y_all
            cons = [x_all >= 0]
            for r in range(k_known):
                lim = known_mass_limits[r] if r < len(known_mass_limits) else max_known
                cons += [x_all[:, r] <= lim]
            for r in range(k_known, k_total):
                cons += [x_all[:, r] <= max_unknown]
            cons += [
                error_all <= tolerance + s_all,
                -error_all <= tolerance + s_all,
            ]
            obj_all = cp.sum_squares(error_all) + 1e3 * cp.sum(s_all)
            prob_all = cp.Problem(cp.Minimize(obj_all), cons)  # type: ignore[arg-type]
            try:
                prob_all.solve(solver=cp.GUROBI, TimeLimit=60)
                if should_cancel and should_cancel():
                    raise SolverCancelledError("Analysis stopped by user request.")
                if prob_all.status in ("optimal", "optimal_inaccurate") and x_all.value is not None:
                    x_val_all = x_all.value
                    recon_all = common_refined + np.sum(x_val_all * b_refined, axis=1)
                    errors_all = np.abs(observations - recon_all)
                    rss_all = float(np.sum(errors_all ** 2))
                    n_good_all = int(np.sum(errors_all < final_tolerance))

                    used_unk = 0
                    for r in range(k_known, k_total):
                        if np.any(np.round(x_val_all[:, r]) > 0):
                            used_unk += 1
                    bic_all = compute_bic(n, rss_all, k_known + used_unk)

                    print(
                        f"\n  3b result: good={n_good_all}/{n}, "
                        f"mean_err={np.mean(errors_all):.4f}, BIC={bic_all:.1f}"
                    )

                    best_model = {
                        "x": x_val_all,
                        "b": b_refined,
                        "common": common_refined,
                        "bic": bic_all,
                        "rss": rss_all,
                        "n_good": n_good_all,
                        "n_bad": n - n_good_all,
                        "mean_error": float(np.mean(errors_all)),
                        "errors": errors_all,
                        "converged": refined_result["converged"],
                        "used_unknowns": used_unk,
                    }
                else:
                    print("  3b x-solve failed; keeping 3a result")
            except Exception as e:
                print(f"  3b x-solve error: {e}; keeping 3a result")
        else:
            print("  3b refinement failed; keeping 3a result")
    else:
        print(
            f"\nSkipping Phase 3b — too few good peaks ({n_good_3a}) "
            f"for reliable refinement"
        )

    # ================================================================
    # PHASE 4 – MODEL COMPARISON
    # ================================================================
    # When exhaustive=True, test every non-empty subset of blocks.
    # Otherwise fall back to the nested (progressive) comparison.
    x_val = best_model["x"]
    b = best_model["b"]
    common_block = best_model["common"]

    if x_val is None or b is None:
        print("No feasible solution found.")
        return

    k_total = len(b)

    # ---- build final names ----
    final_names = list(known_block_names)
    unknown_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for i in range(k_unknown_final):
        block_val = b[k_known + i]
        nm = _name_for_mass(block_val, "")
        if nm:
            final_names.append(nm)
        else:
            final_names.append(
                unknown_letters[i] if i < len(unknown_letters) else f"Unk{i+1}"
            )

    # Initialise model-comparison variables (overwritten in the active branch)
    _exhaustive_results = None
    _peak_stats = None
    _model_results = None
    _peak_explanations = None
    _peak_simplest = None

    if exhaustive and k_total >= 1:
        if should_cancel and should_cancel():
            raise SolverCancelledError("Analysis stopped by user request.")

        print(f"\n{'='*60}")
        print(f"PHASE 4: Exhaustive model comparison (level {exhaustive})")
        print(f"{'='*60}")

        _exhaustive_results, _peak_stats = _run_exhaustive_comparison(
            observations, common_block, b, final_names, k_known, k_total,
            known_mass_limits, max_known, max_unknown, tolerance, final_tolerance,
            exhaustive_level=exhaustive,
            should_cancel=should_cancel,
        )

        # ---- Infer block biosynthetic dependencies ----
        _dep_info = infer_block_dependencies(
            _exhaustive_results, final_names, final_tolerance,
        )
        write_dependency_report(_dep_info, output)

        # Reorder model labels to follow inferred biosynthetic order
        _bio_order = _dep_info["biosynthetic_order"]
        _exhaustive_results = reorder_exhaustive_results(
            _exhaustive_results, _bio_order,
        )
        # Update peak_stats best_model labels to match
        for _pi in range(n):
            _bm = _peak_stats[_pi].get("best_model")
            if _bm:
                _peak_stats[_pi]["best_model"] = reorder_model_label(
                    _bm, _bio_order,
                )

        # ---- summary table (sorted by BIC) ----
        _valid_models = sorted(
            [(k, v) for k, v in _exhaustive_results.items() if v is not None],
            key=lambda x: x[1]["bic"],
        )
        n_models_tested = _peak_stats[0]["n_models_tested"] if _peak_stats else 0

        print(f"\n  {'Model':<40} {'Blocks':>6} {'Good':>5} {'Bad':>5} "
              f"{'MeanErr':>9} {'BIC':>10}")
        print(f"  {'-'*40} {'-'*6} {'-'*5} {'-'*5} {'-'*9} {'-'*10}")
        for _lbl, _res in _valid_models:
            _ns = _lbl[:37] + "..." if len(_lbl) > 40 else _lbl
            print(f"  {_ns:<40} {_res['n_blocks']:>6} {_res['n_good']:>5} "
                  f"{_res['n_bad']:>5} {_res['mean_error']:>9.4f} "
                  f"{_res['bic']:>10.1f}")

        # ---- per-peak analysis ----
        print(f"\n  Per-peak exhaustive analysis "
              f"({n_models_tested} models tested):")
        _any_issue = False
        for i in range(n):
            ps = _peak_stats[i]
            if not ps["consistent"]:
                _any_issue = True
                print(f"    Peak {i+1} ({observations[i]:.3f}): "
                      f"{ps['n_explaining']}/{n_models_tested} models, "
                      f"INCONSISTENT compositions")
            elif ps["n_explaining"] == 0:
                _any_issue = True
                print(f"    Peak {i+1} ({observations[i]:.3f}): "
                      f"NOT explained by any model")
        if not _any_issue:
            print("    All explained peaks have consistent compositions "
                  "across models.")

        # ---- best (simplest) explaining model per peak ----
        print("\n  Best explaining model per peak "
              "(simplest model, preferring larger blocks on ties):")
        for i in range(n):
            ps = _peak_stats[i]
            bm = ps.get("best_model")
            bf = ps.get("best_formula")
            if bm:
                print(f"    Peak {i+1} ({observations[i]:.3f}): "
                      f"best_explained_by [{bm}]  {bf}")
            else:
                print(f"    Peak {i+1} ({observations[i]:.3f}): "
                      f"not explained by any model")

        # ---- model-level stats: % of good peaks best-explained ----
        # For each model, count how many good peaks choose it as simplest.
        _model_best_counts: dict[str, int] = {}
        _n_good_peaks = 0
        for i in range(n):
            ps = _peak_stats[i]
            bm = ps.get("best_model")
            if bm:
                _n_good_peaks += 1
                _model_best_counts[bm] = _model_best_counts.get(bm, 0) + 1

        if _n_good_peaks > 0:
            print(f"\n  Model recommendation summary "
                  f"({_n_good_peaks} good peaks total):")
            print(f"  {'Model':<40} {'Best for':>8} {'%':>6}")
            print(f"  {'-'*40} {'-'*8} {'-'*6}")
            for _lbl, _cnt in sorted(
                _model_best_counts.items(),
                key=lambda x: -x[1],
            ):
                pct = 100.0 * _cnt / _n_good_peaks
                print(f"  {_lbl:<40} {_cnt:>8} {pct:>5.1f}%")

        # ---- build consensus model ----
        # For each peak, use the composition from its simplest explaining
        # model.  This "Consensus" model is not a real subset model but a
        # per-peak best-of-all-models aggregation.
        consensus_x_full = np.zeros((n, k_total))
        consensus_errors = np.zeros(n)
        _n_consensus_good = 0
        for i in range(n):
            ps = _peak_stats[i]
            comp = ps.get("best_composition")
            if comp is not None:
                consensus_x_full[i] = comp
            # Compute error from whatever composition we have (zeros if unexplained)
            recon = common_block + sum(
                consensus_x_full[i, r] * b[r] for r in range(k_total)
            )
            consensus_errors[i] = abs(observations[i] - recon)
            if consensus_errors[i] < final_tolerance:
                _n_consensus_good += 1

        # Determine which block types are actually used in the consensus
        _consensus_blocks_used = set()
        for i in range(n):
            for r in range(k_total):
                if int(round(consensus_x_full[i, r])) > 0:
                    _consensus_blocks_used.add(r)
        _consensus_n_blocks = len(_consensus_blocks_used)
        _consensus_rss = float(np.sum(consensus_errors ** 2))
        _consensus_bic = compute_bic(n, _consensus_rss, _consensus_n_blocks) if _n_consensus_good > 0 else float("inf")

        consensus_block_names = [final_names[r] for r in sorted(_consensus_blocks_used)]
        consensus_label = "Consensus"

        _exhaustive_results[consensus_label] = {
            "x": consensus_x_full,       # n × k_total (not subset-shaped)
            "x_full": consensus_x_full,
            "errors": consensus_errors,
            "bic": _consensus_bic,
            "n_good": _n_consensus_good,
            "n_bad": n - _n_consensus_good,
            "mean_error": float(np.mean(consensus_errors)),
            "blocks_used": consensus_block_names,
            "block_indices": sorted(_consensus_blocks_used),
            "n_blocks": _consensus_n_blocks,
            "is_consensus": True,
        }

        print("\n  Consensus model (per-peak simplest explanation):")
        print(f"    Blocks used: {'+'.join(consensus_block_names)}")
        print(f"    Good: {_n_consensus_good}/{n}, "
              f"BIC: {_consensus_bic:.1f}, "
              f"Mean error: {float(np.mean(consensus_errors)):.4f}")

        # ---- build BioConsensus model ----
        # Uses biosynthetic network connectivity to pick the most
        # biologically plausible composition per peak.
        bio_x_full, bio_errors = _build_bio_consensus(
            _exhaustive_results,        # includes all subset models
            observations,
            common_block,
            b,
            final_names,
            k_total,
            final_tolerance,
        )
        _n_bio_good = int(np.sum(bio_errors < final_tolerance))

        _bio_blocks_used = set()
        for i in range(n):
            for r in range(k_total):
                if int(round(bio_x_full[i, r])) > 0:
                    _bio_blocks_used.add(r)
        _bio_n_blocks = len(_bio_blocks_used)
        _bio_rss = float(np.sum(bio_errors ** 2))
        _bio_bic = compute_bic(n, _bio_rss, _bio_n_blocks) if _n_bio_good > 0 else float("inf")

        bio_block_names = [final_names[r] for r in sorted(_bio_blocks_used)]

        _exhaustive_results["BioConsensus"] = {
            "x": bio_x_full,
            "x_full": bio_x_full,
            "errors": bio_errors,
            "bic": _bio_bic,
            "n_good": _n_bio_good,
            "n_bad": n - _n_bio_good,
            "mean_error": float(np.mean(bio_errors)),
            "blocks_used": bio_block_names,
            "block_indices": sorted(_bio_blocks_used),
            "n_blocks": _bio_n_blocks,
            "is_consensus": True,
        }

        print("\n  BioConsensus model (biosynthetically parsimonious):")
        print(f"    Blocks used: {'+'.join(bio_block_names)}")
        print(f"    Good: {_n_bio_good}/{n}, "
              f"BIC: {_bio_bic:.1f}, "
              f"Mean error: {float(np.mean(bio_errors)):.4f}")

        # Show per-peak differences between Consensus and BioConsensus
        _n_diff = 0
        for i in range(n):
            c_old = [int(round(consensus_x_full[i, r])) for r in range(k_total)]
            c_new = [int(round(bio_x_full[i, r])) for r in range(k_total)]
            if c_old != c_new:
                _n_diff += 1
                old_parts = [f"{c}{final_names[r]}" for r, c in enumerate(c_old) if c > 0]
                new_parts = [f"{c}{final_names[r]}" for r, c in enumerate(c_new) if c > 0]
                old_f = "Common + " + " + ".join(old_parts) if old_parts else "Common"
                new_f = "Common + " + " + ".join(new_parts) if new_parts else "Common"
                print(f"    Peak {i+1} ({observations[i]:.3f}): "
                      f"Consensus={old_f}  =>  BioConsensus={new_f}")
        if _n_diff == 0:
            print("    (identical to Consensus — all peaks agree)")
        else:
            print(f"    {_n_diff} peak(s) differ between Consensus and BioConsensus")

        # ---- build BioConsensus2 model ----
        # Enumerates ALL valid integer compositions per peak (not just
        # the single solution Gurobi returned per model) and uses an ILP
        # to select the set that maximises L₁=1 connectivity.
        print("\n  BioConsensus2 (full enumeration + ILP connectivity):")
        bio2_x_full, bio2_errors, bio2_alternatives = _build_bio_consensus2(
            observations,
            common_block,
            b,
            final_names,
            k_total,
            k_known,
            known_mass_limits,
            max_known,
            max_unknown,
            tolerance,
            final_tolerance,
        )
        _n_bio2_good = int(np.sum(bio2_errors < final_tolerance))

        _bio2_blocks_used = set()
        for i in range(n):
            for r in range(k_total):
                if int(round(bio2_x_full[i, r])) > 0:
                    _bio2_blocks_used.add(r)
        _bio2_n_blocks = len(_bio2_blocks_used)
        _bio2_rss = float(np.sum(bio2_errors ** 2))
        _bio2_bic = (
            compute_bic(n, _bio2_rss, _bio2_n_blocks)
            if _n_bio2_good > 0 else float("inf")
        )

        bio2_block_names = [final_names[r] for r in sorted(_bio2_blocks_used)]

        _exhaustive_results["BioConsensus2"] = {
            "x": bio2_x_full,
            "x_full": bio2_x_full,
            "errors": bio2_errors,
            "bic": _bio2_bic,
            "n_good": _n_bio2_good,
            "n_bad": n - _n_bio2_good,
            "mean_error": float(np.mean(bio2_errors)),
            "blocks_used": bio2_block_names,
            "block_indices": sorted(_bio2_blocks_used),
            "n_blocks": _bio2_n_blocks,
            "is_consensus": True,
        }

        print(f"    Blocks used: {'+'.join(bio2_block_names)}")
        print(f"    Good: {_n_bio2_good}/{n}, "
              f"BIC: {_bio2_bic:.1f}, "
              f"Mean error: {float(np.mean(bio2_errors)):.4f}")

        # Show per-peak differences vs BioConsensus
        _n_diff2 = 0
        for i in range(n):
            c_bio = [int(round(bio_x_full[i, r])) for r in range(k_total)]
            c_bio2 = [int(round(bio2_x_full[i, r])) for r in range(k_total)]
            if c_bio != c_bio2:
                _n_diff2 += 1
                old_parts = [f"{c}{final_names[r]}" for r, c in enumerate(c_bio) if c > 0]
                new_parts = [f"{c}{final_names[r]}" for r, c in enumerate(c_bio2) if c > 0]
                old_f = "Common + " + " + ".join(old_parts) if old_parts else "Common"
                new_f = "Common + " + " + ".join(new_parts) if new_parts else "Common"
                n_alts = len(bio2_alternatives[i])
                print(f"    Peak {i+1} ({observations[i]:.3f}): "
                      f"BioConsensus={old_f}  =>  BioConsensus2={new_f}"
                      f"  ({n_alts} alternatives)")
        if _n_diff2 == 0:
            print("    (identical to BioConsensus)")
        else:
            print(f"    {_n_diff2} peak(s) differ between "
                  f"BioConsensus and BioConsensus2")

        # Show peaks with multiple alternatives
        for i in range(n):
            alts = bio2_alternatives[i]
            if len(alts) > 1:
                chosen = tuple(int(round(bio2_x_full[i, r]))
                               for r in range(k_total))
                chosen_parts = [f"{c}{final_names[r]}"
                                for r, c in enumerate(chosen) if c > 0]
                print(f"    Peak {i+1} ({observations[i]:.3f}) — "
                      f"{len(alts)} valid compositions, "
                      f"chose: {' + '.join(chosen_parts)}")
                for comp, err in alts[:8]:  # show up to 8
                    parts = [f"{c}{final_names[r]}"
                             for r, c in enumerate(comp) if c > 0]
                    tag = " <-- selected" if comp == chosen else ""
                    print(f"      {' + '.join(parts)}  "
                          f"(err={err:.4f}){tag}")
                if len(alts) > 8:
                    print(f"      ... and {len(alts) - 8} more")

        # ---- update peak_stats to reflect BioConsensus2 choices ----
        for i in range(n):
            bio2_comp = [int(round(bio2_x_full[i, r])) for r in range(k_total)]
            if bio2_errors[i] < final_tolerance:
                parts = []
                for r in range(k_total):
                    if bio2_comp[r] > 0:
                        parts.append(f"{bio2_comp[r]}{final_names[r]}")
                bio2_formula = " + ".join(parts) if parts else "(empty)"
                _peak_stats[i]["best_model"] = "BioConsensus2"
                _peak_stats[i]["best_formula"] = bio2_formula
                _peak_stats[i]["best_composition"] = bio2_comp
                _peak_stats[i]["n_alternatives"] = len(bio2_alternatives[i])
                _peak_stats[i]["alternatives"] = bio2_alternatives[i]

        # ---- update best_model to BIC-best exhaustive model ----
        if _valid_models:
            _best_lbl, _best_exh = _valid_models[0]
            print(f"\n  Best model by BIC: {_best_lbl} "
                  f"(BIC={_best_exh['bic']:.1f}, "
                  f"good={_best_exh['n_good']}/{n})")
            best_model = {
                "x": _best_exh["x_full"],
                "b": b,
                "common": common_block,
                "bic": _best_exh["bic"],
                "rss": float(np.sum(_best_exh["errors"] ** 2)),
                "n_good": _best_exh["n_good"],
                "n_bad": _best_exh["n_bad"],
                "mean_error": _best_exh["mean_error"],
                "errors": _best_exh["errors"],
                "converged": True,
                "used_unknowns": sum(
                    1 for bi in _best_exh["block_indices"] if bi >= k_known
                ),
            }

    elif k_total >= 2:
        print(f"\n{'='*60}")
        print("PHASE 4: Nested model comparison")
        print(f"{'='*60}")

        # For each model level m (using first m blocks), solve x with
        # the refined masses, recording compositions and errors.
        # model_results[m] = { "x": ..., "errors": ..., "bic": ..., ... }
        model_results = {}

        for m in range(k_known, k_total + 1):
            if should_cancel and should_cancel():
                raise SolverCancelledError("Analysis stopped by user request.")

            b_sub = b[:m]
            y_sub = observations - common_block

            x_m = cp.Variable((n, m), integer=True)
            s_m = cp.Variable(n, nonneg=True)
            error_m = x_m @ b_sub - y_sub
            cons_m = [x_m >= 0]
            for r in range(min(m, k_known)):
                lim = known_mass_limits[r] if r < len(known_mass_limits) else max_known
                cons_m += [x_m[:, r] <= lim]
            for r in range(k_known, m):
                cons_m += [x_m[:, r] <= max_unknown]
            cons_m += [
                error_m <= tolerance + s_m,
                -error_m <= tolerance + s_m,
            ]
            obj_m = cp.sum_squares(error_m) + 1e3 * cp.sum(s_m)
            prob_m = cp.Problem(cp.Minimize(obj_m), cons_m)  # type: ignore[arg-type]

            try:
                prob_m.solve(solver=cp.GUROBI, TimeLimit=60)
                if prob_m.status in ("optimal", "optimal_inaccurate") and x_m.value is not None:
                    x_val_m = x_m.value
                    recon_m = common_block + np.sum(x_val_m * b_sub, axis=1)
                    errors_m = np.abs(observations - recon_m)
                    rss_m = float(np.sum(errors_m ** 2))
                    n_good_m = int(np.sum(errors_m < final_tolerance))
                    n_bad_m = n - n_good_m
                    bic_m = compute_bic(n, rss_m, m)

                    model_results[m] = {
                        "x": x_val_m,
                        "errors": errors_m,
                        "bic": bic_m,
                        "n_good": n_good_m,
                        "n_bad": n_bad_m,
                        "mean_error": float(np.mean(errors_m)),
                        "blocks_used": final_names[:m],
                    }
                else:
                    model_results[m] = None
            except Exception as e:
                print(f"  Model with {m} blocks failed: {e}")
                model_results[m] = None

        # ---- summary table ----
        print(f"\n  {'Blocks':<6} {'Names':<40} {'Good':>5} {'Bad':>5} {'MeanErr':>9} {'BIC':>10}")
        print(f"  {'-'*6} {'-'*40} {'-'*5} {'-'*5} {'-'*9} {'-'*10}")
        best_bic_model = None
        for m in range(k_known, k_total + 1):
            res = model_results.get(m)
            if res is None:
                print(f"  {m:<6} {'(failed)':<40}")
                continue
            names_str = "+".join(res["blocks_used"])
            if len(names_str) > 40:
                names_str = names_str[:37] + "..."
            if best_bic_model is None or res["bic"] < model_results[best_bic_model]["bic"]:
                best_bic_model = m
            print(
                f"  {m:<6} {names_str:<40} {res['n_good']:>5} {res['n_bad']:>5} "
                f"{res['mean_error']:>9.4f} {res['bic']:>10.1f}"
            )

        if best_bic_model is not None:
            print(f"\n  Best model by BIC: {best_bic_model} blocks "
                  f"({'+'.join(model_results[best_bic_model]['blocks_used'])})")

        # ---- per-peak analysis: simplest adequate explanation ----
        # For each peak, find the lowest m where error < final_tolerance
        peak_simplest = {}       # peak_idx -> m
        peak_explanations = {}   # peak_idx -> { m: (composition_str, error) }

        for i in range(n):
            peak_explanations[i] = {}
            for m in range(k_known, k_total + 1):
                res = model_results.get(m)
                if res is None:
                    continue
                x_row = res["x"][i, :]
                err = res["errors"][i]
                parts = []
                for r in range(m):
                    count = int(round(x_row[r]))
                    if count > 0:
                        parts.append(f"{count}{final_names[r]}")
                comp = " + ".join(parts) if parts else "(empty)"
                peak_explanations[i][m] = (comp, err)

                if i not in peak_simplest and err < final_tolerance:
                    peak_simplest[i] = m

        # ---- detect peaks where adding a block changes composition ----
        # A peak is "ambiguous" if the full model gives a different
        # composition than the simplest adequate model.
        print("\n  Per-peak analysis (simplest adequate model):")
        ambiguous_count = 0
        for i in range(n):
            simp_m = peak_simplest.get(i)
            full_comp, full_err = peak_explanations[i].get(k_total, ("?", float("inf")))

            if simp_m is not None:
                simp_comp, simp_err = peak_explanations[i][simp_m]
                # Check if full model gives a different composition
                if full_comp != simp_comp and full_err < final_tolerance:
                    ambiguous_count += 1
                    print(
                        f"    Peak {i+1} ({observations[i]:.3f}): "
                        f"simplest ({simp_m} blocks): {simp_comp} "
                        f"[err={simp_err:.3f}]"
                    )
                    print(
                        f"      alternative ({k_total} blocks): {full_comp} "
                        f"[err={full_err:.3f}]"
                    )
            else:
                # Peak not explained by any model
                print(
                    f"    Peak {i+1} ({observations[i]:.3f}): "
                    f"NOT explained by any model subset "
                    f"(best: {full_comp} [err={full_err:.3f}])"
                )

        if ambiguous_count == 0:
            print("    No ambiguous peaks — all good peaks have consistent compositions.")

        # ---- update best_model to the BIC-best nested model ----
        if best_bic_model is not None and best_bic_model != k_total:
            bic_best_res = model_results[best_bic_model]
            # Pad x with zeros for missing block columns so shape stays (n, k_total)
            x_padded = np.zeros((n, k_total))
            x_padded[:, :best_bic_model] = bic_best_res["x"]
            best_model = {
                "x": x_padded,
                "b": b,
                "common": common_block,
                "bic": bic_best_res["bic"],
                "rss": float(np.sum(bic_best_res["errors"] ** 2)),
                "n_good": bic_best_res["n_good"],
                "n_bad": bic_best_res["n_bad"],
                "mean_error": bic_best_res["mean_error"],
                "errors": bic_best_res["errors"],
                "converged": True,
                "used_unknowns": best_bic_model - k_known,
            }
            print(f"\n  Using BIC-best model ({best_bic_model} blocks) for final output.")
        # Store peak explanations for output
        _peak_explanations = peak_explanations
        _peak_simplest = peak_simplest
        _model_results = model_results
    # else: all variables already initialised to None

    # ================================================================
    # OUTPUT
    # ================================================================
    x_val = best_model["x"]
    b = best_model["b"]
    common_block = best_model["common"]

    if x_val is None or b is None:
        print("No feasible solution found.")
        return

    k_total = len(b)

    # ---- build original (non-adjusted / theoretical) block masses ----
    b_original = np.zeros(k_total)
    for i in range(k_known):
        b_original[i] = known_differential[i]
    for i in range(k_known, k_total):
        nm_lower = final_names[i].lower()
        if nm_lower in _name_to_mass:
            b_original[i] = _name_to_mass[nm_lower]
        else:
            b_original[i] = b[i]  # fallback to optimized

    # ---- theoretical common block mass from composition ----
    # The original `common` parameter already includes all components:
    # sugar masses + label + adduct (polarity) + reduction correction,
    # as computed by compute_common_mass() in the web UI or supplied
    # directly via CLI.  The optimizer may drift `common_block` during
    # refinement, so use the original input value as the theoretical mass.
    common_block_theoretical = float(common) if common >= 0 else common_block

    # ---- summary ----
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nCommon Block ({common_name}): {common_block:.5f}  (theoretical: {common_block_theoretical:.5f})")
    print("Differential blocks:")
    for i, nm in enumerate(final_names):
        used_count = int(np.sum(np.round(x_val[:, i]) > 0))
        btype = "known" if i < k_known else "discovered"
        print(f"  {nm}: {b[i]:.5f}  (theoretical: {b_original[i]:.5f}, {btype}, used by {used_count} peaks)")

    print("\nModel summary:")
    print(f"  Blocks : {k_known} known + {k_unknown_final} discovered = {k_total}")
    print(f"  BIC    : {best_model['bic']:.1f}")
    print(f"  Good   : {best_model['n_good']}/{n}")
    print(f"  Bad    : {best_model['n_bad']}")
    print(f"  Mean Δ : {best_model['mean_error']:.5f}")

    print("\nObservation Decompositions:")
    for i in range(n):
        parts = []
        for r in range(k_total):
            count = int(round(x_val[i, r]))
            if count > 0:
                parts.append(f"{count}{final_names[r]}")
        recipe = f"{common_name} ({common_block:.3f})"
        if parts:
            recipe += " + " + " + ".join(parts)
        # Full glycan structure
        struct = merge_structure_formula(common_composition, final_names, x_val[i, :])
        struct_str = f"  [{struct}]" if struct else ""

        recon_obs = common_block + np.sum(x_val[i, :] * b)
        recon_theor = common_block_theoretical + np.sum(x_val[i, :] * b_original)
        error = abs(observations[i] - recon_obs)
        flag = " [BAD]" if error >= final_tolerance else ""
        print(
            f"Peak {i+1} ({observations[i]:.3f}) => {recipe}{struct_str}, "
            f"error={error:.3f}, recon={recon_obs:.3f}, "
            f"recon_theor={recon_theor:.3f}{flag}"
        )

        # Print alternative explanation if this peak has one
        if _peak_explanations is not None and i in _peak_explanations:
            simp_m = _peak_simplest.get(i) if _peak_simplest else None
            if simp_m is not None and simp_m < k_total:
                simp_comp, simp_err = _peak_explanations[i][simp_m]
                # Current output may be from full or BIC-best model;
                # show simpler alternative if composition differs
                current_comp = " + ".join(parts) if parts else "(empty)"
                if simp_comp != current_comp:
                    print(
                        f"       alt ({simp_m} blocks): {common_block:.3f} + {simp_comp}, "
                        f"error={simp_err:.3f}"
                    )

    # ---- BioConsensus2 decompositions ----
    if _exhaustive_results is not None and "BioConsensus2" in _exhaustive_results:
        bio2_res = _exhaustive_results["BioConsensus2"]
        if bio2_res is not None:
            bio2_x = bio2_res["x_full"]
            print(f"\n{'='*60}")
            print("BioConsensus2 Decompositions:")
            print(f"  Good: {bio2_res['n_good']}/{n}, "
                  f"BIC: {bio2_res['bic']:.1f}, "
                  f"Mean error: {bio2_res['mean_error']:.4f}")
            print(f"{'='*60}")
            for i in range(n):
                parts2 = []
                for r in range(k_total):
                    count2 = int(round(bio2_x[i, r]))
                    if count2 > 0:
                        parts2.append(f"{count2}{final_names[r]}")
                recipe2 = f"{common_name} ({common_block:.3f})"
                if parts2:
                    recipe2 += " + " + " + ".join(parts2)
                struct2 = merge_structure_formula(common_composition, final_names, bio2_x[i, :])
                struct_str2 = f"  [{struct2}]" if struct2 else ""
                recon2 = common_block + np.sum(bio2_x[i, :] * b)
                recon2_theor = common_block_theoretical + np.sum(bio2_x[i, :] * b_original)
                err2 = abs(observations[i] - recon2)
                flag2 = " [BAD]" if err2 >= final_tolerance else ""
                print(
                    f"Peak {i+1} ({observations[i]:.3f}) => {recipe2}{struct_str2}, "
                    f"error={err2:.3f}, recon={recon2:.3f}, "
                    f"recon_theor={recon2_theor:.3f}{flag2}"
                )

    # ---- TSV output ----
    if _exhaustive_results is not None:
        # Exhaustive output: one row per (peak × subset-model) with
        # per-peak summary columns (N_Models_Tested, N_Models_Explaining,
        # Composition_Consistent).
        write_exhaustive_tsv_output(
            observations,
            common_block,
            b,
            final_names,
            _exhaustive_results,
            output,
            final_tolerance,
            _peak_stats,
            common_name=common_name,
            common_composition=common_composition,
            b_original=b_original,
            c_theoretical=common_block_theoretical,
        )
    elif _model_results is not None:
        # Multi-model output: one row per (peak × model) so the user can
        # filter by Model in a spreadsheet and compare BIC across models.
        write_multimodel_tsv_output(
            observations,
            common_block,
            b,
            final_names,
            _model_results,
            output,
            final_tolerance,
            k_known=k_known,
            common_name=common_name,
            common_composition=common_composition,
            b_original=b_original,
            c_theoretical=common_block_theoretical,
        )
    else:
        # Fallback: single-model output (no nested comparison available)
        write_tsv_output(
            observations,
            common_block,
            b,
            x_val,
            final_names,
            output,
            final_tolerance,
            k_total=k_total,
            common_name=common_name,
            common_composition=common_composition,
            b_original=b_original,
            c_theoretical=common_block_theoretical,
        )

    blocks_path = os.path.join(output, "blocks.tsv")
    with open(blocks_path, "w") as f:
        f.write("Block\tValue\tTheoretical\tType\n")
        f.write(f"{common_name}\t{common_block:.5f}\t{common_block_theoretical:.5f}\tfixed\n")
        for i, nm in enumerate(final_names):
            btype = "known" if i < k_known else "discovered"
            f.write(f"{nm}\t{b[i]:.5f}\t{b_original[i]:.5f}\t{btype}\n")
    print(f"Blocks written to {blocks_path}")

    # Biosynthetic plausibility analysis
    results_path = os.path.join(output, "results.tsv")
    if os.path.exists(results_path):
        try:
            analyse_biosynthetic_paths(results_path, output)
        except Exception as exc:
            print(f"Warning: biosynthetic analysis failed: {exc}")

        # Model diagnostics (residuals + block usage)
        try:
            run_diagnostics(results_path, output)
        except Exception as exc:
            print(f"Warning: diagnostics failed: {exc}")
