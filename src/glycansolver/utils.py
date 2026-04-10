import os
import re

import numpy as np


# ---------------------------------------------------------------------------
# Common-block composition helpers
# ---------------------------------------------------------------------------

H_MASS = 1.00784          # monoisotopic hydrogen atom
NA_MASS = 22.989769        # monoisotopic sodium
O_MASS = 15.994915        # monoisotopic oxygen (for reduction correction)

ADDUCTS: dict[str, float] = {
    "neg_h": -H_MASS,       # [M - H]-
    "pos_na":  NA_MASS,      # [M + Na]+
    "pos_h": H_MASS,       # [M + H]⁺
}

REDUCTION_CORRECTION: dict[str, float] = {
    "nr": 0.0,                # non-reductive
    "r":  -2 * H_MASS,        # reductive (1 double bond → lose 2H)
    "red_end": 2 * H_MASS + O_MASS,  # free reducing end (+H₂O)
}


def load_labels(path: str | None = None) -> list[dict]:
    """Load derivatization labels from a TSV file.

    Returns ``[{"name": ..., "mass": ..., "description": ...}, ...]``.
    """
    labels: list[dict] = []
    if path is None:
        return labels
    if not os.path.exists(path):
        return labels
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                labels.append({
                    "name": parts[0].strip(),
                    "mass": float(parts[1].strip()),
                    "description": parts[2].strip() if len(parts) >= 3 else "",
                })
    return labels


def compute_common_mass(
    composition: dict[str, int],
    block_masses: dict[str, float],
    label_mass: float,
    polarity: str,
    reduction: str,
) -> float:
    """Compute common-block mass from composition + label + adduct + reduction.

    Parameters
    ----------
    composition : dict
        Sugar composition, e.g. ``{"Hex": 3, "HexNAc": 2}``.
    block_masses : dict
        Name → monoisotopic mass (from blocks.txt).
    label_mass : float
        Mass of the derivatization label (e.g. 137.047679 for 2-AA).
    polarity : str
        ``"neg_h"``, ``"pos_na"`` ([M+Na]+), or ``"pos_h"`` ([M+H]+).
    reduction : str
        ``"nr"`` (non-reductive) or ``"r"`` (reductive).
    """
    base = sum(count * block_masses[name] for name, count in composition.items()
               if name in block_masses)
    base += label_mass
    base += ADDUCTS.get(polarity, 0.0)
    base += REDUCTION_CORRECTION.get(reduction, 0.0)
    return base


def common_composition_name(composition: dict[str, int] | None) -> str:
    """Format a composition dict as a name string: ``'3Hex+2HexNAc'``."""
    if not composition:
        return "Common"
    parts = [f"{count}{name}" for name, count in composition.items() if count > 0]
    return "+".join(parts) if parts else "Common"


def merge_structure_formula(
    common_composition: dict[str, int] | None,
    block_names: list[str],
    x_row,
) -> str | None:
    """Merge common composition with differential counts into a full structure.

    Returns ``None`` when *common_composition* is not provided.
    """
    if common_composition is None:
        return None
    merged: dict[str, int] = {}
    for name, count in common_composition.items():
        merged[name] = merged.get(name, 0) + count
    for r, name in enumerate(block_names):
        c = int(round(x_row[r]))
        if c > 0:
            merged[name] = merged.get(name, 0) + c
    # Order: block_names first, then any remaining from common
    parts: list[str] = []
    seen: set[str] = set()
    all_names = list(block_names)
    for name in common_composition:
        if name not in seen and name not in block_names:
            all_names.append(name)
    for name in all_names:
        if name in merged and merged[name] > 0 and name not in seen:
            parts.append(f"{merged[name]}{name}")
            seen.add(name)
    return " + ".join(parts) if parts else "(core only)"


# Load peaks from MSD file
def parse_msd_file(file_path):
    try:
        # Read the file content
        with open(file_path, "r") as f:
            content = f.read()

        # Extract the peaklist section using regex since it might not be well-formed XML
        peaklist_match = re.search(r"<peaklist>(.*?)</peaklist>", content, re.DOTALL)
        if not peaklist_match:
            raise ValueError(f"Could not find peaklist section in {file_path}")

        peaklist_content = peaklist_match.group(1)

        # Parse each peak element to extract mz values
        mz_values = []
        for peak_match in re.finditer(r"<peak\s+([^>]+)", peaklist_content):
            peak_attrs = peak_match.group(1)
            mz_match = re.search(r'mz="([^"]+)"', peak_attrs)
            if mz_match:
                mz_values.append(float(mz_match.group(1)))

        if not mz_values:
            print(f"Warning: No mz values found in {file_path}")

        return np.array(mz_values)

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error parsing MSD file: {e}") from e


# Load peaks from file if provided
def load_peaks(peaks_file):
    try:
        # Check file extension
        _, ext = os.path.splitext(peaks_file)

        # If it's an MSD file, use the MSD parser
        if ext.lower() == ".msd":
            print("Detected MSD file format, using MSD parser")
            return parse_msd_file(peaks_file)

        # Otherwise use the standard text file parser
        with open(peaks_file, "r") as f:
            lines = f.readlines()
            peaks = [
                float(line.strip())
                for line in lines
                if line.strip() and not line.strip().startswith("#")
            ]
            return np.array(peaks)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error loading peaks file: {e}") from e


# Ensure output directory exists
def ensure_output_directory(output_dir):
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            raise RuntimeError(f"Error creating output directory: {e}") from e
    elif not os.path.isdir(output_dir):
        raise RuntimeError(f"{output_dir} exists but is not a directory")


# Add these helper functions before the optimization loop
def distance_to_nearest_multiple(value, base=1.0004):
    """Calculate the distance to the nearest multiple of base."""
    nearest_multiple = round(value / base) * base
    return abs(value - nearest_multiple)


def find_nearest_multiple(value, base=1.0004):
    """Find the nearest multiple of base."""
    return round(value / base) * base


def write_tsv_output(
    observations,
    c_est,
    b,
    x_val,
    final_names,
    output_dir,
    tol_final,
    k_total,
    common_name="Common",
    common_composition=None,
    b_original=None,
    c_theoretical=None,
):
    try:
        output_path = os.path.join(output_dir, "results.tsv")
        with open(output_path, "w") as f:
            # Write header
            header = [
                "Peak_ID",
                "Observed",
                "Reconstructed",
                "Reconstructed_Theoretical",
                "Error",
                "Status",
                "Formula",
                "Structure",
            ]
            for name in final_names:
                header.append(name)
            f.write("\t".join(header) + "\n")

            # Write data rows
            for i in range(len(observations)):
                # Calculate reconstruction and error
                recon_obs = c_est + np.sum(x_val[i, :] * b)
                error = np.abs(observations[i] - recon_obs)
                status = "BAD" if error >= tol_final else "GOOD"

                # Generate formula
                parts = []
                for r in range(k_total):
                    count = int(round(x_val[i, r]))
                    if count > 0:
                        parts.append("{}{}".format(count, final_names[r]))
                if parts:
                    formula = common_name + " + " + " + ".join(parts)
                else:
                    formula = common_name

                # Full structure (merged common + differential)
                structure = merge_structure_formula(
                    common_composition, final_names, x_val[i, :])
                if structure is None:
                    structure = formula

                # Theoretical reconstruction from non-adjusted masses
                c_theor = c_theoretical if c_theoretical is not None else c_est
                recon_theor = c_theor + np.sum(x_val[i, :] * b_original) if b_original is not None else recon_obs

                # Start the row with basic info
                row = [
                    f"{i + 1}",
                    f"{observations[i]:.5f}",
                    f"{recon_obs:.5f}",
                    f"{recon_theor:.5f}",
                    f"{error:.5f}",
                    status,
                    formula,
                    structure,
                ]

                # Add counts for each building block
                for r in range(k_total):
                    count = int(round(x_val[i, r]))
                    row.append(str(count))

                f.write("\t".join(row) + "\n")

            print(f"TSV results written to {output_path}")
    except Exception as e:
        print(f"Error writing TSV output: {e}")


def write_candidates_tsv(ranked_clusters, filtered_candidates, output_dir):
    try:
        with open(os.path.join(output_dir, "candidates.tsv"), "w") as f:
            # Write header
            f.write(
                "Median\tScore\tSize\tMin\tMax\tStatus\tFilter_Reason\tSample_Values\n"
            )

            # Write active candidates data
            for median, score, size, cluster in ranked_clusters:
                min_val, max_val = min(cluster), max(cluster)
                values_str = ",".join([f"{v:.5f}" for v in cluster[:5]])
                if len(cluster) > 5:
                    values_str += f",...({len(cluster) - 5} more)"
                f.write(
                    f"{median:.5f}\t{score:.5f}\t{size}\t{min_val:.5f}\t{max_val:.5f}\tACTIVE\t\t{values_str}\n"
                )

            # Write filtered candidates - check if it's the expected format
            if filtered_candidates and len(filtered_candidates) > 0:
                # Check the format of the first element
                first_elem = filtered_candidates[0]
                if isinstance(first_elem, tuple) and len(first_elem) == 2:
                    # Format: (diff, filter_reason)
                    for diff, filter_reason in filtered_candidates:
                        f.write(
                            f"{diff:.5f}\t0.00000\t1\t{diff:.5f}\t{diff:.5f}\tFILTERED\t{filter_reason}\t{diff:.5f}\n"
                        )
                else:
                    # Legacy format or unexpected format - handle gracefully
                    for item in filtered_candidates:
                        if isinstance(item, (int, float)):
                            f.write(
                                f"{item:.5f}\t0.00000\t1\t{item:.5f}\t{item:.5f}\tFILTERED\tUnknown\t{item:.5f}\n"
                            )

            print(f"Candidates written to {output_dir}/candidates.tsv")
    except Exception as e:
        print(f"Error writing candidates TSV: {e}")


def write_exhaustive_tsv_output(
    observations,
    c_est,
    b_full,
    final_names,
    exhaustive_results,
    output_dir,
    tol_final,
    peak_stats,
    common_name="Common",
    common_composition=None,
    b_original=None,
    c_theoretical=None,
):
    """Write results TSV from exhaustive model comparison.

    The file has one row per (peak x model) with three extra summary
    columns appended: *N_Models_Tested*, *N_Models_Explaining* and
    *Composition_Consistent*.

    Parameters
    ----------
    observations : np.ndarray
        Observed peak masses.
    c_est : float
        Estimated common-block mass.
    b_full : np.ndarray
        Block masses for the full set (length k_total).
    final_names : list[str]
        Human-readable block names (same order as *b_full*).
    exhaustive_results : dict[str, dict | None]
        ``{model_label: result_dict}`` from ``_run_exhaustive_comparison``.
    output_dir : str
        Directory to write ``results.tsv`` into.
    tol_final : float
        Error tolerance separating GOOD from BAD.
    peak_stats : dict[int, dict]
        ``{peak_idx: {n_models_tested, n_explaining, consistent}}``.
    common_name : str
        Name for the common block (e.g. ``'3Hex+2HexNAc'``).
    common_composition : dict[str, int] | None
        Sugar composition of the common block (for Structure column).
    """
    try:
        k_total = len(b_full)
        n = len(observations)
        output_path = os.path.join(output_dir, "results.tsv")

        with open(output_path, "w") as f:
            # ---- header ----
            header = [
                "Peak_ID", "Observed",
                "Model", "Model_Blocks", "Model_BIC",
                "Reconstructed", "Reconstructed_Theoretical",
                "Error", "Status", "Formula", "Structure",
            ]
            for name in final_names:
                header.append(name)
            header += [
                "N_Models_Tested",
                "N_Models_Explaining",
                "Composition_Consistent",
                "Best_Model",
                "Best_Formula",
                "Best_Structure",
            ]
            f.write("\t".join(header) + "\n")

            # ---- sort models: fewest blocks first, then BIC ascending ----
            sorted_models = sorted(
                [(lbl, res) for lbl, res in exhaustive_results.items()
                 if res is not None],
                key=lambda x: (x[1]["n_blocks"], x[1]["bic"]),
            )

            # ---- rows ----
            for i in range(n):
                ps = peak_stats.get(i, {})
                n_tested = ps.get("n_models_tested", 0)
                n_explaining = ps.get("n_explaining", 0)
                consistent_str = "yes" if ps.get("consistent", True) else "no"
                best_model_str = ps.get("best_model") or ""
                best_formula_raw = ps.get("best_formula") or ""
                # Prefix common name to best_formula
                if best_formula_raw and best_formula_raw != "(empty)":
                    best_formula_str = common_name + " + " + best_formula_raw
                elif best_formula_raw:
                    best_formula_str = common_name
                else:
                    best_formula_str = ""
                # Best structure: merge common composition with best composition
                best_comp = ps.get("best_composition")
                if best_comp is not None and common_composition is not None:
                    best_structure_str = merge_structure_formula(
                        common_composition, final_names, best_comp)
                    if best_structure_str is None:
                        best_structure_str = best_formula_str
                else:
                    best_structure_str = best_formula_str

                for model_label, res in sorted_models:
                    x_row = res["x_full"][i]
                    err = res["errors"][i]
                    status = "BAD" if err >= tol_final else "GOOD"

                    recon_obs = c_est + float(np.sum(x_row * b_full))
                    c_theor = c_theoretical if c_theoretical is not None else c_est
                    recon_theor = c_theor + float(np.sum(x_row * b_original)) if b_original is not None else recon_obs

                    # Formula string
                    parts = []
                    for r in range(k_total):
                        count = int(round(x_row[r]))
                        if count > 0:
                            parts.append(f"{count}{final_names[r]}")
                    formula = (common_name + " + " + " + ".join(parts)) if parts else common_name

                    # Structure string
                    structure = merge_structure_formula(
                        common_composition, final_names, x_row)
                    if structure is None:
                        structure = formula

                    row = [
                        f"{i + 1}",
                        f"{observations[i]:.5f}",
                        model_label,
                        str(res["n_blocks"]),
                        f"{res['bic']:.2f}",
                        f"{recon_obs:.5f}",
                        f"{recon_theor:.5f}",
                        f"{err:.5f}",
                        status,
                        formula,
                        structure,
                    ]

                    # Block counts (full vector, already padded to k_total)
                    for r in range(k_total):
                        row.append(str(int(round(x_row[r]))))

                    row += [
                        str(n_tested), str(n_explaining), consistent_str,
                        best_model_str, best_formula_str, best_structure_str,
                    ]
                    f.write("\t".join(row) + "\n")

        print(f"Exhaustive results written to {output_path}")
    except Exception as e:
        print(f"Error writing exhaustive TSV output: {e}")


def write_multimodel_tsv_output(
    observations,
    c_est,
    b_full,
    final_names,
    model_results,
    output_dir,
    tol_final,
    k_known,
    common_name="Common",
    common_composition=None,
    b_original=None,
    c_theoretical=None,
):
    """Write a results TSV with one row per (peak × model).

    Each row is labelled with the model it came from so the user can
    filter by *Model* in a spreadsheet.  The model-level BIC is also
    included so that different model complexities can be compared at a
    glance.

    Parameters
    ----------
    observations : np.ndarray
        Observed peak masses.
    c_est : float
        Estimated common-block mass.
    b_full : np.ndarray
        Block masses for the full model (length k_total).
    final_names : list[str]
        Human-readable names for every block (same order as *b_full*).
    model_results : dict[int, dict | None]
        ``{m: result_dict}`` produced by Phase 4 nested model comparison.
        *m* is the number of blocks in that model.  ``result_dict`` has
        keys ``x``, ``errors``, ``bic``, ``n_good``, ``n_bad``,
        ``mean_error``, ``blocks_used``.  Value may be ``None`` if the
        model failed.
    output_dir : str
        Directory to write ``results.tsv`` into.
    tol_final : float
        Error tolerance that separates GOOD from BAD peaks.
    k_known : int
        Number of known (always-present) blocks.
    common_name : str
        Name for the common block.
    common_composition : dict[str, int] | None
        Sugar composition of the common block (for Structure column).
    """
    try:
        k_total = len(b_full)
        n = len(observations)

        output_path = os.path.join(output_dir, "results.tsv")
        with open(output_path, "w") as f:
            # ---- header ----
            header = [
                "Peak_ID",
                "Observed",
                "Model",
                "Model_Blocks",
                "Model_BIC",
                "Reconstructed",
                "Reconstructed_Theoretical",
                "Error",
                "Status",
                "Formula",
                "Structure",
            ]
            for name in final_names:
                header.append(name)
            f.write("\t".join(header) + "\n")

            # ---- sort model levels ----
            sorted_levels = sorted(
                m for m in model_results if model_results[m] is not None
            )

            # ---- rows ----
            for i in range(n):
                for m in sorted_levels:
                    res = model_results[m]
                    if res is None:
                        continue

                    x_row = res["x"][i, :]
                    err = res["errors"][i]
                    bic = res["bic"]
                    status = "BAD" if err >= tol_final else "GOOD"

                    # Reconstruction
                    b_sub = b_full[:m]
                    recon_obs = c_est + float(np.sum(x_row * b_sub))
                    c_theor = c_theoretical if c_theoretical is not None else c_est
                    b_orig_sub = b_original[:m] if b_original is not None else b_sub
                    recon_theor = c_theor + float(np.sum(x_row * b_orig_sub))

                    # Model label — e.g. "Hex+dHex" or "Known_only"
                    blocks_used = final_names[:m]
                    model_label = "+".join(blocks_used)

                    # Formula string
                    parts = []
                    for r in range(m):
                        count = int(round(x_row[r]))
                        if count > 0:
                            parts.append(f"{count}{final_names[r]}")
                    formula = (common_name + " + " + " + ".join(parts)) if parts else common_name

                    # Structure: pad x_row to k_total for merge
                    x_padded = [0] * k_total
                    for r in range(m):
                        x_padded[r] = int(round(x_row[r]))
                    structure = merge_structure_formula(
                        common_composition, final_names, x_padded)
                    if structure is None:
                        structure = formula

                    row = [
                        f"{i + 1}",
                        f"{observations[i]:.5f}",
                        model_label,
                        str(m),
                        f"{bic:.2f}",
                        f"{recon_obs:.5f}",
                        f"{recon_theor:.5f}",
                        f"{err:.5f}",
                        status,
                        formula,
                        structure,
                    ]

                    # Block counts — pad with 0 for blocks not in this model
                    for r in range(k_total):
                        if r < m:
                            row.append(str(int(round(x_row[r]))))
                        else:
                            row.append("")
                    f.write("\t".join(row) + "\n")

            print(f"Multi-model TSV results written to {output_path}")
    except Exception as e:
        print(f"Error writing multi-model TSV output: {e}")
