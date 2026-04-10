"""
Block dependency inference from exhaustive model comparison.

By comparing which blocks are actually *used* across all tested model
subsets, this module infers biosynthetic dependencies between blocks.

When a block B is only used (explains additional peaks) in models that
also contain block C, we infer that B depends on C — i.e., C must be
present in a composition before B can be added.

Example
-------
Given blocks {Hex, HexNAc, dHex, NeuAc}:

- Model {Hex}: Hex used → Hex is a root block
- Model {Hex, dHex}: dHex NOT used → dHex needs more context
- Model {Hex, HexNAc}: HexNAc used → HexNAc depends on Hex
- Model {Hex, HexNAc, dHex}: dHex used → dHex needs {Hex, HexNAc}
  → via transitive reduction, direct dependency is HexNAc

Inferred order: Hex → HexNAc → {dHex, NeuAc}
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Dependency inference
# ---------------------------------------------------------------------------

def infer_block_dependencies(
    exhaustive_results: dict[str, dict | None],
    final_names: list[str],
    final_tolerance: float,
) -> dict:
    """Infer biosynthetic dependencies between blocks.

    For each block B, examines every model subset that includes B and
    checks whether B is actually *used* (non-zero coefficient on at least
    one GOOD peak).  The intersection of all "other blocks present" sets
    across models where B is used gives B's prerequisite blocks.  A
    transitive reduction then yields direct dependencies.

    Parameters
    ----------
    exhaustive_results : dict
        ``{model_label: result_dict}`` from exhaustive comparison.
        Consensus models are excluded automatically.
    final_names : list[str]
        Block names in column order (matching ``x_full`` columns).
    final_tolerance : float
        Error threshold for GOOD peaks.

    Returns
    -------
    dict with keys:
        ``prerequisites``     : ``{block: frozenset of required blocks}``
        ``direct_deps``       : ``{block: frozenset of direct dep blocks}``
        ``biosynthetic_order``: ``list[str]`` topological order
        ``order_map``         : ``{block: int}`` position in order
        ``dependency_edges``  : ``list[(parent, child)]``
        ``usage_info``        : per-block usage statistics
    """
    consensus_labels = {"Consensus", "BioConsensus", "BioConsensus2"}

    # Filter to valid, non-consensus models
    valid_models = {
        label: res
        for label, res in exhaustive_results.items()
        if res is not None and label not in consensus_labels
    }

    prerequisites: dict[str, frozenset[str]] = {}
    usage_info: dict[str, dict] = {}

    for b_idx, block_name in enumerate(final_names):
        contexts_where_used: list[frozenset[str]] = []
        contexts_where_not_used: list[frozenset[str]] = []

        for label, res in valid_models.items():
            model_blocks = set(label.split("+"))
            if block_name not in model_blocks:
                continue

            other_blocks = frozenset(model_blocks - {block_name})

            # Check if block B has non-zero usage in at least one GOOD peak
            good_mask = res["errors"] < final_tolerance
            if good_mask.any():
                x_full = res["x_full"]
                block_used = bool(
                    np.any(np.round(x_full[good_mask, b_idx]) > 0)
                )
            else:
                block_used = False

            if block_used:
                contexts_where_used.append(other_blocks)
            else:
                contexts_where_not_used.append(other_blocks)

        if contexts_where_used:
            prereqs = frozenset.intersection(*contexts_where_used)
        else:
            prereqs = frozenset()

        prerequisites[block_name] = prereqs
        usage_info[block_name] = {
            "n_models_with_block": (
                len(contexts_where_used) + len(contexts_where_not_used)
            ),
            "n_models_used": len(contexts_where_used),
            "n_models_not_used": len(contexts_where_not_used),
            "contexts_used": contexts_where_used,
            "contexts_not_used": contexts_where_not_used,
        }

    # Transitive reduction → direct dependencies
    direct_deps = _transitive_reduction(prerequisites)

    # Topological sort → biosynthetic order
    biosynthetic_order = _topological_sort(direct_deps, final_names)

    order_map = {name: idx for idx, name in enumerate(biosynthetic_order)}

    dependency_edges = []
    for child, parents in direct_deps.items():
        for parent in sorted(parents, key=lambda p: order_map.get(p, 999)):
            dependency_edges.append((parent, child))

    return {
        "prerequisites": prerequisites,
        "direct_deps": direct_deps,
        "biosynthetic_order": biosynthetic_order,
        "order_map": order_map,
        "dependency_edges": dependency_edges,
        "usage_info": usage_info,
    }


# ---------------------------------------------------------------------------
# Graph algorithms
# ---------------------------------------------------------------------------

def _transitive_reduction(
    prerequisites: dict[str, frozenset[str]],
) -> dict[str, frozenset[str]]:
    """Remove transitive prerequisites to obtain direct dependencies.

    If A requires {B, C} and B requires {C}, then A's direct dependency
    is only {B} — C is implied transitively through B.
    """
    direct: dict[str, frozenset[str]] = {}

    for block, prereqs in prerequisites.items():
        if not prereqs:
            direct[block] = frozenset()
            continue

        # Collect everything reachable transitively through the prereqs
        indirect: set[str] = set()
        for p in prereqs:
            indirect.update(prerequisites.get(p, frozenset()))

        direct[block] = prereqs - indirect

    return direct


def _topological_sort(
    direct_deps: dict[str, frozenset[str]],
    final_names: list[str],
) -> list[str]:
    """Kahn's algorithm topological sort.

    Ties broken by original position in *final_names* (preserves user
    ordering when biosynthetic order is ambiguous).
    """
    name_to_pos = {name: idx for idx, name in enumerate(final_names)}

    in_degree: dict[str, int] = {name: 0 for name in final_names}
    children: dict[str, list[str]] = {name: [] for name in final_names}

    for child, parents in direct_deps.items():
        if child not in in_degree:
            continue
        in_degree[child] = len(parents)
        for parent in parents:
            if parent in children:
                children[parent].append(child)

    queue = sorted(
        [n for n in final_names if in_degree[n] == 0],
        key=lambda n: name_to_pos[n],
    )

    result: list[str] = []
    while queue:
        node = queue.pop(0)
        result.append(node)
        for child in children.get(node, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
        queue.sort(key=lambda n: name_to_pos[n])

    # Append any remaining blocks (shouldn't happen unless there's a cycle)
    for name in final_names:
        if name not in result:
            result.append(name)

    return result


# ---------------------------------------------------------------------------
# Model label reordering
# ---------------------------------------------------------------------------

def reorder_model_label(
    label: str,
    biosynthetic_order: list[str],
) -> str:
    """Reorder blocks in a ``'+'``-joined model label by biosynthetic order.

    >>> reorder_model_label("Hex+dHex+HexNAc", ["Hex", "HexNAc", "dHex"])
    'Hex+HexNAc+dHex'
    """
    if label in ("Consensus", "BioConsensus", "BioConsensus2"):
        return label

    blocks = label.split("+")
    order_map = {name: idx for idx, name in enumerate(biosynthetic_order)}
    max_pos = len(biosynthetic_order)
    blocks.sort(key=lambda b: order_map.get(b, max_pos))
    return "+".join(blocks)


def reorder_exhaustive_results(
    exhaustive_results: dict[str, dict | None],
    biosynthetic_order: list[str],
) -> dict[str, dict | None]:
    """Return a *new* dict whose model-label keys follow biosynthetic order.

    The ``blocks_used`` list inside each result is also reordered.
    """
    order_map = {name: idx for idx, name in enumerate(biosynthetic_order)}
    max_pos = len(biosynthetic_order)

    new_results: dict[str, dict | None] = {}
    for old_label, res in exhaustive_results.items():
        new_label = reorder_model_label(old_label, biosynthetic_order)
        if res is not None:
            res = dict(res)  # shallow copy
            if "blocks_used" in res:
                res["blocks_used"] = sorted(
                    res["blocks_used"],
                    key=lambda b: order_map.get(b, max_pos),
                )
        new_results[new_label] = res
    return new_results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_dependency_report(
    dep_info: dict,
    output_dir: str,
) -> None:
    """Write ``block_dependencies.tsv`` and print a console summary."""
    prereqs_map = dep_info["prerequisites"]
    direct_map = dep_info["direct_deps"]
    order_map = dep_info["order_map"]
    usage_map = dep_info["usage_info"]
    bio_order = dep_info["biosynthetic_order"]
    edges = dep_info["dependency_edges"]

    # ---- TSV ----
    tsv_path = os.path.join(output_dir, "block_dependencies.tsv")
    with open(tsv_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow([
            "Block", "Prerequisites", "Direct_Dependencies",
            "Biosynthetic_Order", "N_Models_With_Block",
            "N_Models_Used", "N_Models_Not_Used",
        ])
        for name in bio_order:
            prereqs = prereqs_map.get(name, frozenset())
            direct = direct_map.get(name, frozenset())
            usage = usage_map.get(name, {})
            writer.writerow([
                name,
                "+".join(sorted(prereqs)) if prereqs else "(none)",
                "+".join(sorted(direct)) if direct else "(root)",
                order_map.get(name, -1),
                usage.get("n_models_with_block", 0),
                usage.get("n_models_used", 0),
                usage.get("n_models_not_used", 0),
            ])
    print(f"Block dependencies written to {tsv_path}")

    # ---- Console summary ----
    print("\n  Block Biosynthetic Dependencies:")
    print(f"  {'Block':<15} {'Direct Deps':<25} "
          f"{'All Prerequisites':<30} {'Used/Total'}")
    print(f"  {'-' * 15} {'-' * 25} {'-' * 30} {'-' * 10}")

    for name in bio_order:
        prereqs = prereqs_map.get(name, frozenset())
        direct = direct_map.get(name, frozenset())
        usage = usage_map.get(name, {})
        prereqs_str = "+".join(sorted(prereqs)) if prereqs else "(none)"
        direct_str = "+".join(sorted(direct)) if direct else "(root)"
        n_used = usage.get("n_models_used", 0)
        n_total = usage.get("n_models_with_block", 0)
        print(f"  {name:<15} {direct_str:<25} "
              f"{prereqs_str:<30} {n_used}/{n_total}")

    print(f"\n  Inferred Biosynthetic Order: "
          f"{' -> '.join(bio_order)}")

    # ---- Dependency tree (text) ----
    if edges:
        print("\n  Block Dependency Tree:")
        _print_dependency_tree(dep_info)


def _print_dependency_tree(dep_info: dict) -> None:
    """Print a text-based block dependency tree."""
    direct_deps = dep_info["direct_deps"]
    order_map = dep_info["order_map"]
    bio_order = dep_info["biosynthetic_order"]

    # Roots: blocks with no direct dependencies
    roots = [
        name for name in bio_order
        if not direct_deps.get(name, frozenset())
    ]

    # Children map
    children: dict[str, list[str]] = defaultdict(list)
    for parent, child in dep_info["dependency_edges"]:
        children[parent].append(child)

    for parent in children:
        children[parent].sort(key=lambda c: order_map.get(c, 999))

    def _subtree(name: str, prefix: str, is_last: bool) -> None:
        connector = "└── " if is_last else "├── "
        usage = dep_info["usage_info"].get(name, {})
        n_used = usage.get("n_models_used", 0)
        n_total = usage.get("n_models_with_block", 0)
        print(f"  {prefix}{connector}{name}  "
              f"(used in {n_used}/{n_total} models)")
        child_prefix = prefix + ("    " if is_last else "│   ")
        kids = children.get(name, [])
        for i, kid in enumerate(kids):
            _subtree(kid, child_prefix, i == len(kids) - 1)

    for ri, root in enumerate(roots):
        usage = dep_info["usage_info"].get(root, {})
        n_used = usage.get("n_models_used", 0)
        n_total = usage.get("n_models_with_block", 0)
        print(f"  {root}  (used in {n_used}/{n_total} models)")
        kids = children.get(root, [])
        for i, kid in enumerate(kids):
            _subtree(kid, "  ", i == len(kids) - 1)
