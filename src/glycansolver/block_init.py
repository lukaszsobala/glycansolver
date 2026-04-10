import os

import numpy as np

from .utils import find_nearest_multiple


def get_smart_block_init(
    observations,
    common,
    known_differential,
    k_unknown,
    min_diff,
    verbose,
    min_gap=0.25,
    blocks_dict_path=None,
    lower_bound=None,
    upper_bound=None,
    glycan_type=None,
):
    """
    Use observed differences to inform initialization of unknown blocks.
    Ensures unknown blocks don't overlap with known blocks.

    Args:
        observations: observed mass values
        common: common block value
        known_differential: array of known differential block values
        k_unknown: number of unknown blocks to initialize
        min_diff: minimum difference threshold to consider
        verbose: whether to print detailed information
        min_gap: minimum difference between blocks
        blocks_dict_path: path to the blocks dictionary file (optional)
        lower_bound: lower bound for unknown block mass (optional)
        upper_bound: upper bound for unknown block mass (optional)
        glycan_type: glycan type filter for blocks dictionary (optional)

    Returns:
        Array of initial values for unknown blocks, treatment flag, all candidate values, ranked clusters, and filtered candidates
    """
    # Load blocks dictionary if provided
    blocks_dict = {}
    if blocks_dict_path and os.path.exists(blocks_dict_path):
        blocks_dict = load_blocks_dictionary(blocks_dict_path, glycan_type=glycan_type)
        if blocks_dict:
            print(f"Loaded {len(blocks_dict)} entries from blocks dictionary")

    # Calculate pairwise differences between observations
    y = observations - common
    diffs = []
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            diff = abs(y[i] - y[j])
            if diff > min_diff:  # Only consider differences above this value
                diffs.append(diff)

    # Keep track of filtered candidates and reasons
    filtered_candidates = []

    # Filter out differences too close to known blocks AND their multiples
    filtered_diffs = []
    for diff in diffs:
        too_close = False
        filter_reason = ""

        # Check against original known blocks
        for known in known_differential:
            if abs(diff - known) < min_gap:
                too_close = True
                filter_reason = f"Too close to known block {known:.3f}"
                break

            # Check against multiples of known blocks (up to 8x)
            for multiplier in range(2, 10):  # Check 2x - 10x
                if abs(diff - (known * multiplier)) < min_gap:
                    too_close = True
                    filter_reason = f"Multiple of {known:.3f} ({multiplier}x)"
                    if verbose:
                        print(
                            f"Filtered out {diff:.3f} as multiple of {known:.3f} ({multiplier}x)"
                        )
                    break

            if too_close:
                break

        # Check against combinations of known blocks (up to 3+3 of each)
        if not too_close and len(known_differential) >= 2:
            # Check all combinations where each block can appear 0 to 3 times
            for i, known1 in enumerate(known_differential):
                for j, known2 in enumerate(
                    known_differential[i:], i
                ):  # Start from i to avoid duplicates
                    for count1 in range(4):  # 0 to 3 copies of first block
                        for count2 in range(4):  # 0 to 3 copies of second block
                            # Skip the (0,0) case as it would filter out everything
                            if count1 == 0 and count2 == 0:
                                continue

                            combined = count1 * known1 + count2 * known2
                            # Check if the difference is too close to the combined value

                            if (
                                abs(diff - combined) < min_gap
                            ):  # the data precision will not depend on the number of blocks
                                too_close = True
                                filter_reason = f"Combination of {count1}x{known1:.3f} + {count2}x{known2:.3f}"
                                if verbose:
                                    print(
                                        f"Filtered out {diff:.3f} as combination of {count1}x{known1:.3f} + {count2}x{known2:.3f}"
                                    )
                                break
                        if too_close:
                            break
                    if too_close:
                        break
                if too_close:
                    break

        # Check against differences between known blocks (only positive differences)
        if not too_close and len(known_differential) >= 2:
            for i, known1 in enumerate(known_differential):
                for j, known2 in enumerate(known_differential):
                    if i != j:  # Don't compare a block with itself
                        # Only consider positive differences where first value > second value
                        if known1 > known2:
                            block_diff = (
                                known1 - known2
                            )  # No abs() - only positive differences
                            if abs(diff - block_diff) < min_gap:
                                too_close = True
                                filter_reason = (
                                    f"Difference between {known1:.3f} and {known2:.3f}"
                                )
                                if verbose:
                                    print(
                                        f"Filtered out {diff:.3f} as difference between {known1:.3f} and {known2:.3f}"
                                    )
                                break
                if too_close:
                    break

        # Check against lower/upper bounds for unknown blocks
        if not too_close and lower_bound is not None and diff < lower_bound:
            too_close = True
            filter_reason = f"Below lower bound {lower_bound:.3f}"
        if not too_close and upper_bound is not None and diff > upper_bound:
            too_close = True
            filter_reason = f"Above upper bound {upper_bound:.3f}"

        if too_close:
            filtered_candidates.append((diff, filter_reason))
        else:
            filtered_diffs.append(diff)

    # If we don't have enough differences, fall back to random initialization
    if len(filtered_diffs) == 0 or len(filtered_diffs) < k_unknown:
        if verbose:
            print("Not enough distinct differences found. Using random initialization.")
        # Create a random initialization that avoids known values
        unknown_blocks = []
        for _ in range(k_unknown):
            while True:
                lb = lower_bound if lower_bound is not None else 40
                ub = upper_bound if upper_bound is not None else 350
                value = np.random.uniform(lb, ub)
                too_close = False
                for known in known_differential:
                    if abs(value - known) < min_gap:
                        too_close = True
                        break
                for existing in unknown_blocks:
                    if abs(value - existing) < min_gap:
                        too_close = True
                        break
                if not too_close:
                    unknown_blocks.append(value)
                    break
        return np.array(unknown_blocks), False, [], [], filtered_candidates

    # Enhanced clustering approach for similar differences
    # Step 1: Sort differences for easier clustering
    sorted_diffs = sorted(filtered_diffs)

    # Step 2: Cluster differences that are within min_gap of each other
    clusters = []
    current_cluster = [sorted_diffs[0]]

    for i in range(1, len(sorted_diffs)):
        if sorted_diffs[i] - sorted_diffs[i - 1] <= min_gap:
            # Add to current cluster
            current_cluster.append(sorted_diffs[i])
        else:
            # Start a new cluster
            if len(current_cluster) > 0:
                clusters.append(current_cluster)
            current_cluster = [sorted_diffs[i]]

    # Add the last cluster
    if current_cluster:
        clusters.append(current_cluster)

    # Step 3: Score clusters based on size and tightness
    cluster_scores = []
    for cluster in clusters:
        size = len(cluster)
        if size == 1:
            tightness = 1.0  # Single element clusters get baseline tightness
        else:
            # Calculate tightness as inverse of variance (higher is better)
            variance = np.var(cluster)
            if variance < 1e-10:  # Prevent division by very small numbers
                tightness = 1000.0  # Very tight cluster
            else:
                tightness = 1 / (variance + 0.001)  # Add small constant to avoid division by zero

        # Calculate median and add penalty for deviation from 1.00035 multiple
        median = np.median(cluster)

        # Find nearest multiple of 1.00035
        nearest_mult = find_nearest_multiple(median, base=1.00035)

        # Calculate deviation from nearest multiple
        deviation = abs(median - nearest_mult)

        # Create a multiplier penalty that decreases as deviation increases
        # Use exponential decay: exp(-k*x) where k controls how quickly the penalty drops
        # 0 deviation = 1.0 multiplier, large deviation = small multiplier
        multiple_factor = np.exp(
            -4 * deviation
        )  # Adjust constant to control sensitivity

        # Apply the penalty to the score calculation
        # Score is now a combination of cluster size, tightness, and proximity to 1.00035 multiple
        score = size * tightness * multiple_factor

        # Check if this median is in the blocks dictionary (or very close to a value)
        blocks_dict_match = None
        blocks_dict_boost = 1.0  # Default multiplier (no boost)

        if blocks_dict:
            # Check if the median matches (or is close to) any block in the dictionary
            for mass, name in blocks_dict.items():
                if abs(median - mass) < min_gap:
                    blocks_dict_match = name
                    # Apply a significant boost to score
                    blocks_dict_boost = 10.0  # Major boost for dictionary matches
                    if verbose:
                        print(
                            f"Boosting score of {median:.5f} - matches dictionary entry '{name}' ({mass:.5f})"
                        )
                    break

        # Apply dictionary boost if found
        score *= blocks_dict_boost

        if verbose and size > 2:
            if blocks_dict_match:
                print(
                    f"Cluster median {median:.5f}, matches '{blocks_dict_match}', boosted score by {blocks_dict_boost}x"
                )
            else:
                print(
                    f"Cluster median {median:.5f}, nearest multiple {nearest_mult:.5f}, deviation {deviation:.5f}, penalty factor {multiple_factor:.3f}"
                )

        cluster_scores.append((median, score, size, cluster, blocks_dict_match))

    # Sort clusters by score (descending)
    ranked_clusters = sorted(cluster_scores, key=lambda x: x[1], reverse=True)

    # Print the top candidates (up to 10)
    print("\nTop difference cluster candidates:")
    print(
        f"{'Median':<10} {'Score':<10} {'Size':<8} {'Range':<15} {'Dictionary Match':<15} {'Sample Values'}"
    )
    for i, (median, score, size, cluster, dict_match) in enumerate(
        ranked_clusters[:10]
    ):
        min_val, max_val = min(cluster), max(cluster)
        range_str = f"{min_val:.2f}-{max_val:.2f}"
        values_str = ", ".join([f"{v:.2f}" for v in cluster[:3]])
        if len(cluster) > 3:
            values_str += f", ... ({len(cluster) - 3} more)"
        dict_match_str = dict_match if dict_match else ""
        print(
            f"{median:<10.3f} {score:<10.2f} {size:<8} {range_str:<15} {dict_match_str:<15} {values_str}"
        )

    # Extract all candidate values (medians of clusters) for potential use during stagnation
    all_candidates = [median for median, _, _, _, _ in ranked_clusters]

    # For compatibility with existing code, strip the dict_match field from ranked_clusters
    ranked_clusters_compat = [
        (median, score, size, cluster)
        for median, score, size, cluster, _ in ranked_clusters
    ]

    # Special handling for when we have multiple unknown blocks
    if k_unknown >= 2 and len(ranked_clusters) >= k_unknown:
        # Get the top candidate - it will be treated separately
        top_candidate = ranked_clusters[0][0]
        # Get the rest of the needed candidates
        remaining_candidates = [ranked_clusters[i][0] for i in range(1, k_unknown)]
        selected_medians = [top_candidate] + remaining_candidates

        # Flag to indicate that the first unknown block should be treated as known
        # We'll return this information for the optimization stage
        treat_first_as_known = True
    else:
        # Standard selection of the top k_unknown candidates
        selected_medians = [
            median for median, _, _, _, _ in ranked_clusters[:k_unknown]
        ]
        treat_first_as_known = False

    print(f"\nSelected block estimates: {[f'{x:.3f}' for x in selected_medians]}")

    # Print which blocks were matched from the dictionary
    if blocks_dict:
        for i, median in enumerate(selected_medians):
            for mass, name in blocks_dict.items():
                if abs(median - mass) < min_gap:
                    print(
                        f"Selected block {i + 1} ({median:.3f}) matches dictionary entry '{name}' ({mass:.5f})"
                    )

    if treat_first_as_known and k_unknown >= 2:
        print(
            f"First unknown block ({selected_medians[0]:.3f}) will be treated as known during optimization"
        )

    return (
        np.array(selected_medians),
        treat_first_as_known,
        all_candidates,
        ranked_clusters_compat,
        filtered_candidates,
    )


def load_blocks_dictionary(file_path, glycan_type=None):
    """
    Load a dictionary of known blocks from a TSV file.
    File format: Each line contains a block name, mass value, and optional
    category and glycan_type (tab/space-separated).  Lines starting with '#' are comments.

    Args:
        file_path: Path to the blocks dictionary file
        glycan_type: If given, only include blocks matching this type
                     (e.g. 'native' or 'permethylated'). None returns all.

    Returns:
        Dictionary mapping mass values to block names
    """
    blocks_dict = {}
    try:
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0]
                    try:
                        mass = float(parts[1])
                        gtype = parts[4] if len(parts) >= 5 else "native"
                        if glycan_type and gtype != glycan_type:
                            continue
                        cat = parts[2] if len(parts) >= 3 else "common"
                        if cat == "meta":
                            continue
                        blocks_dict[mass] = name
                    except ValueError:
                        print(
                            f"Warning: Invalid mass value in blocks dictionary line {line_num}: {line}"
                        )
                else:
                    print(
                        f"Warning: Invalid format in blocks dictionary line {line_num}: {line}"
                    )

        return blocks_dict
    except Exception as e:
        print(f"Warning: Could not load blocks dictionary file {file_path}: {e}")
        return {}


def load_blocks_dictionary_with_categories(file_path, glycan_type=None):
    """
    Load a dictionary of known blocks with category information.
    File format: name  mass  [category]  [max_limit]  [glycan_type]
    Category: common (universal monosaccharides), rare (unusual residues), mod (modifications).
    Defaults to 'common' if not specified.

    Args:
        file_path: Path to the blocks dictionary file
        glycan_type: If given, only include blocks matching this type. None returns all.

    Returns:
        Tuple of (mass_to_name dict, mass_to_category dict)
    """
    mass_to_name = {}
    mass_to_category = {}
    try:
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0]
                    try:
                        mass = float(parts[1])
                        gtype = parts[4] if len(parts) >= 5 else "native"
                        if glycan_type and gtype != glycan_type:
                            continue
                        category = parts[2] if len(parts) >= 3 else "common"
                        if category.lower() == "meta":
                            continue
                        mass_to_name[mass] = name
                        mass_to_category[mass] = category.lower()
                    except ValueError:
                        print(
                            f"Warning: Invalid mass value in blocks dictionary line {line_num}: {line}"
                        )
                else:
                    print(
                        f"Warning: Invalid format in blocks dictionary line {line_num}: {line}"
                    )

        return mass_to_name, mass_to_category
    except Exception as e:
        print(f"Warning: Could not load blocks dictionary file {file_path}: {e}")
        return {}, {}
