"""
Biosynthetic **network** analysis for glycan composition models.

For each composition model produced by the progressive solver, this module
assesses biological plausibility by building the **full pairwise network** of
possible enzymatic connections between observed structures and analysing
network-wide connectivity.

Algorithm
---------
1. Parse the multi-model (or single-model) results.tsv.
2. For each model, take only GOOD peaks and their composition vectors.
3. Build the **complete pairwise graph**: for every pair of peaks compute the
   composition delta and classify the edge.
4. Analyse the full network:
   - Count all clean (L₁=1) edges — direct single enzymatic steps.
   - Compute clean subgraph: connected components, density, per-node degree.
   - Compute *clean reachability*: fraction of peaks in the largest clean
     component (1.0 = all peaks connected by single-step paths).
   - Compute the MST (Kruskal) as the **backbone** for tree display.
5. Root the MST at the simplest composition and DFS-traverse for display,
   annotating each node with its full clean neighbour count.

Edge classification
-------------------

   ======================= ===== ============================================
   Class                   Code  Criterion
   ======================= ===== ============================================
   clean                   C     L₁ = 1  (exactly one block ±1)
   skip                    S     L₁ ≥ 2, same-sign, single block type
   multi                   M     L₁ ≥ 2, same-sign, multiple block types
   mixed                   X     additions AND subtractions in same step
   identity                I     L₁ = 0  (same composition, different mass)
   ======================= ===== ============================================

Scoring
-------
   penalty(C)=0, penalty(S)=L₁−1, penalty(M)=L₁−1, penalty(X)=2×L₁, penalty(I)=2

   *Network score* combines MST backbone cost with full-network connectivity::

       mst_mean_penalty   = mean(MST edge penalties)
       clean_reachability = largest_clean_component / n_good   (0..1)
       mixed_frac         = n_mixed_mst_edges / n_mst_edges
       score = mst_mean_penalty × (2 − clean_reachability) × (1 + mixed_frac)

   Lower score = more biologically plausible.

Output
------
* Returns a structured dict for embedding in JSON responses.
* Writes ``biosynthetic_summary.tsv`` to *output_dir*.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class Edge(NamedTuple):
    from_idx: int              # index in sorted_peaks
    to_idx: int
    delta: list[int]           # composition delta (to − from)
    l1: int                    # sum(abs(delta))
    kind: str                  # "clean" | "skip" | "multi" | "mixed" | "identity"
    penalty: float


class NetworkStats(NamedTuple):
    """Full-network statistics (all pairwise edges considered)."""
    total_edges: int           # n*(n−1)/2
    n_clean: int               # edges with L₁=1
    n_skip: int
    n_multi: int
    n_mixed: int
    n_identity: int            # edges with L₁=0 (same composition)
    clean_density: float       # n_clean / total_edges
    avg_clean_degree: float    # mean clean degree across nodes
    n_clean_components: int    # components in the clean-only subgraph
    largest_clean_component: int  # size of the largest clean component
    clean_reachability: float  # largest_clean_component / n_good


class ModelAnalysis(NamedTuple):
    model: str
    model_bic: float | None
    n_good: int
    # --- full network ---
    network: NetworkStats
    # --- MST backbone ---
    n_mst_edges: int
    mst_clean: int
    mst_skip: int
    mst_multi: int
    mst_mixed: int
    mst_identity: int
    mst_mean_penalty: float
    mst_max_l1: int
    # --- combined score ---
    score: float
    # --- structural data ---
    mst_edges: list[Edge]
    all_clean_edges: list[Edge]        # every L₁=1 edge in the full graph
    sorted_peaks: list[dict]
    mst_adjacency: dict[int, list[tuple[int, Edge]]]
    clean_adjacency: dict[int, list[tuple[int, Edge]]]
    node_clean_degree: list[int]       # per-node: number of L₁=1 neighbours


# ---------------------------------------------------------------------------
# Union-Find (Disjoint Set)
# ---------------------------------------------------------------------------

class _UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def component_sizes(self, n: int) -> dict[int, int]:
        sizes: dict[int, int] = defaultdict(int)
        for i in range(n):
            sizes[self.find(i)] += 1
        return sizes


# ---------------------------------------------------------------------------
# Transition classification
# ---------------------------------------------------------------------------

def _classify_transition(delta: list[int]) -> tuple[str, float]:
    """Return (kind, penalty) for a composition delta vector."""
    l1 = sum(abs(d) for d in delta)
    if l1 == 0:
        return "identity", 2.0   # same composition, different mass — not a real step

    additions = sum(d for d in delta if d > 0)
    subtractions = sum(-d for d in delta if d < 0)
    n_changing = sum(1 for d in delta if d != 0)
    mixed = additions > 0 and subtractions > 0

    if mixed:
        return "mixed", 2.0 * l1
    elif l1 == 1:
        return "clean", 0.0
    elif n_changing == 1:
        return "skip", float(l1 - 1)
    else:
        return "multi", float(l1 - 1)


# ---------------------------------------------------------------------------
# Full pairwise network construction
# ---------------------------------------------------------------------------

def _build_full_network(
    peaks: list[dict],
) -> tuple[
    list[tuple[float, float, int, int, list[int], int, str]],  # all raw edges
    list[Edge],                                                  # clean edges
    dict[int, list[tuple[int, Edge]]],                           # clean adjacency
    list[int],                                                   # node clean degrees
    NetworkStats,
]:
    """Compute the complete pairwise network and clean-subgraph statistics."""
    n = len(peaks)
    total_possible = n * (n - 1) // 2

    all_raw: list[tuple[float, float, int, int, list[int], int, str]] = []
    clean_edges: list[Edge] = []
    clean_adj: dict[int, list[tuple[int, Edge]]] = {i: [] for i in range(n)}
    node_clean_deg: list[int] = [0] * n

    counts = {"clean": 0, "skip": 0, "multi": 0, "mixed": 0, "identity": 0}

    # Union-Find for clean subgraph components
    uf_clean = _UnionFind(n)

    for i in range(n):
        ci = peaks[i]["composition"]
        mi = peaks[i]["observed"]
        for j in range(i + 1, n):
            cj = peaks[j]["composition"]
            mj = peaks[j]["observed"]
            delta = [cj[k] - ci[k] for k in range(len(ci))]
            l1 = sum(abs(d) for d in delta)
            kind, penalty = _classify_transition(delta)
            all_raw.append((penalty, abs(mj - mi), i, j, delta, l1, kind))
            counts[kind] += 1

            if kind == "clean":
                fwd = Edge(from_idx=i, to_idx=j, delta=delta,
                           l1=l1, kind=kind, penalty=penalty)
                clean_edges.append(fwd)
                clean_adj[i].append((j, fwd))
                rev = Edge(from_idx=j, to_idx=i, delta=[-d for d in delta],
                           l1=l1, kind=kind, penalty=penalty)
                clean_adj[j].append((i, rev))
                node_clean_deg[i] += 1
                node_clean_deg[j] += 1
                uf_clean.union(i, j)

    comp_sizes = uf_clean.component_sizes(n)
    n_clean_comps = len(comp_sizes)
    largest_clean = max(comp_sizes.values()) if comp_sizes else 0

    stats = NetworkStats(
        total_edges=total_possible,
        n_clean=counts["clean"],
        n_skip=counts["skip"],
        n_multi=counts["multi"],
        n_mixed=counts["mixed"],
        n_identity=counts["identity"],
        clean_density=counts["clean"] / total_possible if total_possible else 0.0,
        avg_clean_degree=sum(node_clean_deg) / n if n else 0.0,
        n_clean_components=n_clean_comps,
        largest_clean_component=largest_clean,
        clean_reachability=largest_clean / n if n else 0.0,
    )

    return all_raw, clean_edges, clean_adj, node_clean_deg, stats


def _build_mst(
    all_raw: list[tuple[float, float, int, int, list[int], int, str]],
    n: int,
) -> tuple[list[Edge], dict[int, list[tuple[int, Edge]]]]:
    """Build the MST from pre-computed raw edges.

    Returns (mst_edges, mst_adjacency).
    """
    if n <= 1:
        return [], {i: [] for i in range(n)}

    sorted_edges = sorted(all_raw)
    uf = _UnionFind(n)
    mst_edges: list[Edge] = []
    mst_adj: dict[int, list[tuple[int, Edge]]] = {i: [] for i in range(n)}

    for penalty, _md, i, j, delta, l1, kind in sorted_edges:
        if uf.union(i, j):
            fwd = Edge(from_idx=i, to_idx=j, delta=delta,
                       l1=l1, kind=kind, penalty=penalty)
            mst_edges.append(fwd)
            mst_adj[i].append((j, fwd))
            rev = Edge(from_idx=j, to_idx=i, delta=[-d for d in delta],
                       l1=l1, kind=kind, penalty=penalty)
            mst_adj[j].append((i, rev))
            if len(mst_edges) == n - 1:
                break

    return mst_edges, mst_adj


# ---------------------------------------------------------------------------
# Model-level analysis
# ---------------------------------------------------------------------------

def _empty_network() -> NetworkStats:
    return NetworkStats(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0, 0, 0.0)


def analyse_model(
    peaks: list[dict],
    block_names: list[str],
    model_label: str,
    model_bic: float | None,
) -> ModelAnalysis:
    """Analyse biosynthetic plausibility via the full pairwise network.

    Parameters
    ----------
    peaks : list of dicts
        GOOD peaks, each with ``peak_id``, ``observed``, ``composition``.
    block_names : list[str]
        Block names in composition-vector order.
    model_label, model_bic : identification fields.
    """
    empty = ModelAnalysis(
        model=model_label, model_bic=model_bic, n_good=0,
        network=_empty_network(),
        n_mst_edges=0, mst_clean=0, mst_skip=0, mst_multi=0, mst_mixed=0,
        mst_identity=0, mst_mean_penalty=0.0, mst_max_l1=0, score=0.0,
        mst_edges=[], all_clean_edges=[], sorted_peaks=[], mst_adjacency={},
        clean_adjacency={}, node_clean_degree=[],
    )
    if not peaks:
        return empty

    sorted_peaks = sorted(peaks, key=lambda p: p["observed"])
    n = len(sorted_peaks)

    if n == 1:
        ns = NetworkStats(0, 0, 0, 0, 0, 0, 0.0, 0.0, 1, 1, 1.0)
        return empty._replace(
            n_good=1, network=ns, sorted_peaks=sorted_peaks,
            mst_adjacency={0: []}, clean_adjacency={0: []},
            node_clean_degree=[0],
        )

    # ---- Full network ----
    all_raw, clean_edges, clean_adj, node_clean_deg, net_stats = \
        _build_full_network(sorted_peaks)

    # ---- MST backbone ----
    mst_edges, mst_adj = _build_mst(all_raw, n)

    mst_counts: dict[str, int] = {"clean": 0, "skip": 0, "multi": 0, "mixed": 0, "identity": 0}
    mst_penalties: list[float] = []
    for e in mst_edges:
        mst_counts[e.kind] += 1
        mst_penalties.append(e.penalty)

    n_mst = len(mst_edges)
    mst_mean = sum(mst_penalties) / n_mst if n_mst else 0.0
    mst_max_l1 = max((e.l1 for e in mst_edges), default=0)
    mst_mixed = mst_counts["mixed"]
    mixed_frac = mst_mixed / n_mst if n_mst else 0.0

    # ---- Combined score ----
    # Incorporates both MST backbone cost and full-network connectivity:
    # score = mst_mean_penalty × (2 − clean_reachability) × (1 + mixed_frac)
    # When clean_reachability=1 (all connected via single steps), factor = 1.
    # When fragmented, factor approaches 2, penalising poor connectivity.
    score = mst_mean * (2.0 - net_stats.clean_reachability) * (1.0 + mixed_frac)

    return ModelAnalysis(
        model=model_label,
        model_bic=model_bic,
        n_good=n,
        network=net_stats,
        n_mst_edges=n_mst,
        mst_clean=mst_counts["clean"],
        mst_skip=mst_counts["skip"],
        mst_multi=mst_counts["multi"],
        mst_mixed=mst_mixed,
        mst_identity=mst_counts["identity"],
        mst_mean_penalty=mst_mean,
        mst_max_l1=mst_max_l1,
        score=score,
        mst_edges=mst_edges,
        all_clean_edges=clean_edges,
        sorted_peaks=sorted_peaks,
        mst_adjacency=mst_adj,
        clean_adjacency=clean_adj,
        node_clean_degree=node_clean_deg,
    )


# ---------------------------------------------------------------------------
# TSV parser
# ---------------------------------------------------------------------------

def _parse_results(path: str) -> tuple[list[str], list[dict]]:
    """Return (block_names, rows) from a results.tsv."""
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyse_biosynthetic_paths(
    results_tsv_path: str,
    output_dir: str,
) -> dict:
    """Analyse biosynthetic plausibility for all models in a results TSV.

    Returns
    -------
    dict  ``{"block_names": [...], "models": [...]}``
    """
    block_names, rows = _parse_results(results_tsv_path)
    if not rows:
        return {"block_names": [], "models": []}

    # Extract common block name from the first GOOD row's Formula column
    common_name = "Common"
    for row in rows:
        if row.get("Status", "").strip() == "GOOD":
            formula = row.get("Formula", "")
            if " + " in formula:
                common_name = formula.split(" + ")[0]
            elif formula:
                common_name = formula
            break

    has_multimodel = "Model" in (rows[0] if rows else {})

    model_peaks: dict[str, list[dict]] = defaultdict(list)
    model_bics: dict[str, float | None] = {}

    for row in rows:
        if row.get("Status", "").strip() != "GOOD":
            continue

        model_key = row.get("Model", "Model") if has_multimodel else "Model"

        try:
            composition = [int(float(row.get(b, 0) or 0)) for b in block_names]
        except (ValueError, KeyError):
            continue

        if has_multimodel:
            # The TSV already has correct per-block counts (0 for blocks
            # not in the model).  We only need to zero out blocks that
            # are NOT part of this model's block set.
            # Skip this for consensus models — their compositions already
            # use the correct per-peak block counts and are not tied to a
            # single block subset.
            model_label = row.get("Model", "")
            if model_label not in ("Consensus", "BioConsensus", "BioConsensus2"):
                model_block_set = set(model_label.split("+")) if model_label else set()
                if model_block_set:
                    composition = [
                        c if bname in model_block_set else 0
                        for c, bname in zip(composition, block_names)
                    ]

        try:
            bic_val: float | None = float(row["Model_BIC"]) if has_multimodel and row.get("Model_BIC") else None
        except (ValueError, TypeError):
            bic_val = None

        model_peaks[model_key].append({
            "peak_id": int(row.get("Peak_ID", 0) or 0),
            "observed": float(row.get("Observed", 0) or 0),
            "composition": composition,
        })
        if model_key not in model_bics:
            model_bics[model_key] = bic_val

    analyses: list[ModelAnalysis] = []
    for model_key, peaks in model_peaks.items():
        ma = analyse_model(peaks, block_names, model_key, model_bics.get(model_key))
        analyses.append(ma)

    analyses.sort(key=lambda a: (a.score, -(a.n_good or 0)))

    summary_path = os.path.join(output_dir, "biosynthetic_summary.tsv")
    _write_summary(analyses, summary_path)
    print(f"Biosynthetic summary written to {summary_path}")

    tree_path = os.path.join(output_dir, "biosynthetic_tree.tsv")
    _write_tree_tsv(analyses, block_names, tree_path, common_name)
    print(f"Biosynthetic tree written to {tree_path}")

    report_path = os.path.join(output_dir, "biosynthetic_report.txt")
    _write_detailed_report(analyses, block_names, report_path, common_name)
    print(f"Biosynthetic report written to {report_path}")

    return _serialise(analyses, block_names, common_name)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _delta_label(delta: list[int], block_names: list[str]) -> str:
    """Human-readable label for a composition delta, e.g. '+Hex -dHex'."""
    parts = []
    for d, name in zip(delta, block_names):
        if d > 0:
            parts.append(f"+{d}{name}" if d > 1 else f"+{name}")
        elif d < 0:
            parts.append(f"{d}{name}" if d < -1 else f"-{name}")
    return " ".join(parts) if parts else "±0"


def _delta_label_additions(delta: list[int], block_names: list[str]) -> str:
    """Label for a composition delta, oriented towards additions.

    * **Pure addition** (all components ≥ 0): show only additions.
    * **Pure subtraction** (all components ≤ 0): flip sign → show as
      additions from the reverse direction.
    * **Mixed** (both positive and negative components): show the full
      delta (both + and −) so the user sees the true relationship.
    """
    has_pos = any(d > 0 for d in delta)
    has_neg = any(d < 0 for d in delta)

    if has_pos and has_neg:
        # Mixed transition — show full delta, don't hide anything
        return _delta_label(delta, block_names)

    if has_pos:
        # Pure addition
        parts = []
        for d, name in zip(delta, block_names):
            if d > 0:
                parts.append(f"+{d}{name}" if d > 1 else f"+{name}")
        return " ".join(parts) if parts else "±0"

    # Pure subtraction — flip to additions
    parts = []
    for d, name in zip(delta, block_names):
        if d < 0:
            ad = -d
            parts.append(f"+{ad}{name}" if ad > 1 else f"+{name}")
    return " ".join(parts) if parts else "±0"


def _formula_label(composition: list[int], block_names: list[str]) -> str:
    """Compact composition label, e.g. 'Hex×3 HexNAc×2'."""
    parts = []
    for count, name in zip(composition, block_names):
        if count > 0:
            parts.append(f"{name}×{count}")
    return " ".join(parts) if parts else "(empty)"


# ---------------------------------------------------------------------------
# TSV summary
# ---------------------------------------------------------------------------

def _write_summary(analyses: list[ModelAnalysis], path: str) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow([
            "Rank", "Model", "BIC", "N_Good",
            # Full network
            "Total_Pairs", "Clean_Edges", "Skip_Edges", "Multi_Edges",
            "Mixed_Edges", "Identity_Edges", "Clean_Density", "Avg_Clean_Degree",
            "Clean_Components", "Largest_Clean_Comp", "Clean_Reachability",
            # MST backbone
            "MST_Edges", "MST_Clean", "MST_Skip", "MST_Multi", "MST_Mixed", "MST_Identity",
            "MST_Mean_Penalty", "MST_Max_L1",
            # Score
            "Score",
        ])
        for rank, ma in enumerate(analyses, 1):
            ns = ma.network
            is_consensus = ma.model in ("Consensus", "BioConsensus", "BioConsensus2")
            bic_str = "" if is_consensus else (f"{ma.model_bic:.2f}" if ma.model_bic is not None else "")
            writer.writerow([
                rank, ma.model,
                bic_str,
                ma.n_good,
                ns.total_edges, ns.n_clean, ns.n_skip, ns.n_multi, ns.n_mixed,
                ns.n_identity,
                f"{ns.clean_density:.4f}", f"{ns.avg_clean_degree:.2f}",
                ns.n_clean_components, ns.largest_clean_component,
                f"{ns.clean_reachability:.3f}",
                ma.n_mst_edges, ma.mst_clean, ma.mst_skip, ma.mst_multi,
                ma.mst_mixed, ma.mst_identity,
                f"{ma.mst_mean_penalty:.3f}", ma.mst_max_l1,
                f"{ma.score:.3f}",
            ])


# ---------------------------------------------------------------------------
# Biosynthetic tree TSV — one row per peak with parent and addition step
# ---------------------------------------------------------------------------

def _write_tree_tsv(
    analyses: list[ModelAnalysis],
    block_names: list[str],
    path: str,
    common_name: str = "Common",
) -> None:
    """Write biosynthetic_tree.tsv with per-peak tree relationships.

    For each model the tree is rooted at the simplest composition and
    edges are presented as additions (only positive delta components).
    """
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow([
            "Model", "Peak_ID", "Mass", "Formula", "Depth",
            "Parent_Peak_ID", "Addition", "Full_Transition",
            "Edge_Kind", "Edge_L1", "Clean_Degree",
        ])

        for ma in analyses:
            chain = _build_tree_chain(ma, block_names, common_name)
            for node in chain:
                parent_id = ""
                addition = ""
                full_trans = ""
                edge_kind = ""
                edge_l1 = ""

                if node.get("transition"):
                    t = node["transition"]
                    parent_id_val = _find_parent_peak_id(chain, node)
                    parent_id = str(parent_id_val) if parent_id_val is not None else ""
                    addition = t.get("addition_label", "")
                    full_trans = t.get("delta_label", "")
                    edge_kind = t.get("kind", "")
                    edge_l1 = str(t.get("l1", ""))

                writer.writerow([
                    ma.model,
                    node["peak_id"],
                    f"{node['mass']:.5f}",
                    node.get("formula", ""),
                    node["depth"],
                    parent_id,
                    addition,
                    full_trans,
                    edge_kind,
                    edge_l1,
                    node.get("clean_degree", 0),
                ])


def _find_parent_peak_id(chain: list[dict], node: dict) -> int | None:
    """Find the parent peak_id of a node in the chain list."""
    if node["depth"] == 0:
        return None
    # Walk backward to find the most recent node at depth - 1
    idx = chain.index(node)
    for i in range(idx - 1, -1, -1):
        if chain[i]["depth"] == node["depth"] - 1:
            return chain[i]["peak_id"]
    return None


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def _build_tree_chain(
    ma: ModelAnalysis,
    block_names: list[str],
    common_name: str = "Common",
) -> list[dict]:
    """Build a biosynthetic tree rooted at the common block.

    A virtual root node represents the common block (differential
    composition ``[0, 0, …, 0]``).  Every peak's differential
    composition is ≥ 0 in each component, so every peak can be reached
    from the root by purely additive steps.

    Only connections where the child's composition is component-wise ≥
    the parent's are used as tree edges.

    Algorithm
    ---------
    1. Create a virtual root at ``[0] * k`` representing the common
       block.
    2. Process peaks in order of increasing total block count (ties
       broken by mass).
    3. For each peak, consider the virtual root and all already-placed
       real peaks as candidate parents.  Only accept parents where
       **every** delta component ≥ 0 (pure addition).
    4. Among valid parents, prefer: lowest L₁, lowest penalty, closest
       mass.
    5. DFS from the virtual root to produce the flat chain list.
    """
    if not ma.sorted_peaks:
        return []

    k = len(block_names)
    ROOT = -1  # sentinel index for the virtual root
    root_comp = [0] * k

    n = len(ma.sorted_peaks)

    # ---- Order peaks: simplest first ----
    order = sorted(range(n), key=lambda i: (
        sum(c for c in ma.sorted_peaks[i]["composition"]),
        ma.sorted_peaks[i]["observed"],
    ))

    parent_of: dict[int, tuple[int, list[int]]] = {}   # child -> (parent, delta)
    children_of: dict[int, list[int]] = defaultdict(list)
    placed: set[int] = {ROOT}  # virtual root is always placed

    for idx in order:
        comp_child = ma.sorted_peaks[idx]["composition"]

        best_parent: tuple[int, list[int]] | None = None
        best_key: tuple | None = None

        for p in placed:
            comp_p = root_comp if p == ROOT else ma.sorted_peaks[p]["composition"]
            delta = [comp_child[j] - comp_p[j] for j in range(k)]

            # ONLY allow purely additive transitions (all components >= 0)
            if any(d < 0 for d in delta):
                continue

            l1 = sum(abs(d) for d in delta)
            if l1 == 0:
                continue  # identical composition

            _, penalty = _classify_transition(delta)

            # Prefer: lower L₁, lower penalty, parent with lower mass
            mass_p = 0.0 if p == ROOT else ma.sorted_peaks[p]["observed"]
            key = (l1, penalty, mass_p)
            if best_key is None or key < best_key:
                best_key = key
                best_parent = (p, delta)

        if best_parent is not None:
            parent_of[idx] = best_parent
            children_of[best_parent[0]].append(idx)
        else:
            # Fallback: attach directly to root (should not happen since
            # all differential compositions are >= 0).
            delta = list(comp_child)
            parent_of[idx] = (ROOT, delta)
            children_of[ROOT].append(idx)

        placed.add(idx)

    # ---- DFS traversal from the virtual root ----
    chain: list[dict] = []

    def _dfs(idx: int, depth: int, delta_from_parent: list[int] | None) -> None:
        if idx == ROOT:
            # Virtual root node — the common block
            kids = sorted(children_of.get(ROOT, []),
                          key=lambda c: ma.sorted_peaks[c]["observed"])
            entry: dict = {
                "peak_id": 0,
                "mass": 0.0,
                "composition": root_comp,
                "formula": common_name,
                "depth": 0,
                "n_children": len(kids),
                "clean_degree": 0,
                "extra_clean": [],
                "is_virtual_root": True,
            }
            chain.append(entry)
            for kid in kids:
                _, kid_delta = parent_of[kid]
                _dfs(kid, 1, kid_delta)
            return

        peak = ma.sorted_peaks[idx]

        kids = sorted(children_of.get(idx, []),
                      key=lambda c: ma.sorted_peaks[c]["observed"])

        # Extra clean connections — L₁=1 neighbours NOT in this tree
        tree_nbs: set[int] = set()
        if idx in parent_of:
            parent_idx = parent_of[idx][0]
            tree_nbs.add(parent_idx)
            # If this node's transition from its parent is identity
            # (L₁=0, same composition), also exclude siblings — they
            # are reachable via the same step as from the parent.
            parent_delta_here = parent_of[idx][1]
            if sum(abs(d) for d in parent_delta_here) == 0:
                for sib in children_of.get(parent_idx, []):
                    if sib != idx:
                        tree_nbs.add(sib)
        for c in children_of.get(idx, []):
            tree_nbs.add(c)

        # Delta from the tree parent to the current node (used to
        # suppress redundant "also" edges where the neighbour has the
        # same composition as the tree parent).
        parent_delta = parent_of[idx][1] if idx in parent_of else None

        extra_clean = []
        for nb, edge in ma.clean_adjacency.get(idx, []):
            if nb not in tree_nbs:
                # Skip if the neighbour→current delta equals
                # parent→current delta (same composition as parent).
                if parent_delta is not None:
                    nb_to_current = [-d for d in edge.delta]
                    if nb_to_current == parent_delta:
                        continue
                nb_pid = ma.sorted_peaks[nb]["peak_id"]
                # Determine direction: if delta is all <= 0, the
                # neighbour is a potential parent (adding blocks to nb
                # gives the current peak).  Otherwise the current peak
                # is the parent (adding blocks to current gives nb).
                nb_is_parent = (all(d <= 0 for d in edge.delta)
                                and any(d < 0 for d in edge.delta))
                if nb_is_parent:
                    # current = nb + block  →  "#nb +block"
                    also_label = (f"#{nb_pid} "
                                  + _delta_label_additions(
                                        edge.delta, block_names))
                else:
                    # nb = current + block  →  "#nb −block"
                    neg_delta = [-d for d in edge.delta]
                    also_label = (f"#{nb_pid} "
                                  + _delta_label(neg_delta, block_names))
                extra_clean.append({
                    "peak_id": nb_pid,
                    "also_label": also_label,
                })

        entry = {
            "peak_id": peak["peak_id"],
            "mass": round(peak["observed"], 5),
            "composition": peak["composition"],
            "formula": _formula_label(peak["composition"], block_names),
            "depth": depth,
            "n_children": len(kids),
            "clean_degree": (ma.node_clean_degree[idx]
                             if idx < len(ma.node_clean_degree) else 0),
            "extra_clean": extra_clean,
        }
        if delta_from_parent is not None:
            l1 = sum(abs(d) for d in delta_from_parent)
            kind, penalty = _classify_transition(delta_from_parent)
            entry["transition"] = {
                "delta": delta_from_parent,
                "delta_label": _delta_label(delta_from_parent, block_names),
                "addition_label": _delta_label_additions(
                    delta_from_parent, block_names),
                "l1": l1,
                "kind": kind,
                "penalty": penalty,
            }
        chain.append(entry)

        for kid in kids:
            _, kid_delta = parent_of[kid]
            _dfs(kid, depth + 1, kid_delta)

    _dfs(ROOT, 0, None)

    return chain


def _serialise(
    analyses: list[ModelAnalysis],
    block_names: list[str],
    common_name: str = "Common",
) -> dict:
    """Return a JSON-serialisable dict for the web response."""
    models_out = []
    for rank, ma in enumerate(analyses, 1):
        tree_chain = _build_tree_chain(ma, block_names, common_name)
        ns = ma.network

        # Flat list of all clean edges (full network)
        clean_edges_out = []
        for e in ma.all_clean_edges:
            p_from = ma.sorted_peaks[e.from_idx]
            p_to = ma.sorted_peaks[e.to_idx]
            clean_edges_out.append({
                "from_peak_id": p_from["peak_id"],
                "to_peak_id": p_to["peak_id"],
                "from_mass": round(p_from["observed"], 5),
                "to_mass": round(p_to["observed"], 5),
                "delta_label": _delta_label(e.delta, block_names),
                "addition_label": _delta_label_additions(e.delta, block_names),
            })

        # MST edge list
        mst_edges_out = []
        for e in ma.mst_edges:
            p_from = ma.sorted_peaks[e.from_idx]
            p_to = ma.sorted_peaks[e.to_idx]
            mst_edges_out.append({
                "from_peak_id": p_from["peak_id"],
                "to_peak_id": p_to["peak_id"],
                "from_mass": round(p_from["observed"], 5),
                "to_mass": round(p_to["observed"], 5),
                "delta_label": _delta_label(e.delta, block_names),
                "addition_label": _delta_label_additions(e.delta, block_names),
                "l1": e.l1,
                "kind": e.kind,
                "penalty": e.penalty,
            })

        # BIC is not meaningful for consensus models — they cherry-pick
        # per-peak compositions from across all subset models.
        is_consensus = ma.model in ("Consensus", "BioConsensus", "BioConsensus2")
        bic_value = None if is_consensus else ma.model_bic

        models_out.append({
            "rank": rank,
            "model": ma.model,
            "bic": bic_value,
            "n_good": ma.n_good,
            # Full network stats
            "network": {
                "total_pairs": ns.total_edges,
                "n_clean": ns.n_clean,
                "n_skip": ns.n_skip,
                "n_multi": ns.n_multi,
                "n_mixed": ns.n_mixed,
                "n_identity": ns.n_identity,
                "clean_density": round(ns.clean_density, 4),
                "avg_clean_degree": round(ns.avg_clean_degree, 2),
                "n_clean_components": ns.n_clean_components,
                "largest_clean_component": ns.largest_clean_component,
                "clean_reachability": round(ns.clean_reachability, 3),
            },
            # MST backbone stats
            "n_mst_edges": ma.n_mst_edges,
            "mst_clean": ma.mst_clean,
            "mst_skip": ma.mst_skip,
            "mst_multi": ma.mst_multi,
            "mst_mixed": ma.mst_mixed,
            "mst_identity": ma.mst_identity,
            "mst_mean_penalty": round(ma.mst_mean_penalty, 3),
            "mst_max_l1": ma.mst_max_l1,
            # Score
            "score": round(ma.score, 3),
            # Display data
            "chain": tree_chain,
            "clean_edges": clean_edges_out,
            "mst_edges": mst_edges_out,
        })

    return {"block_names": block_names, "models": models_out}


# ---------------------------------------------------------------------------
# Detailed human-readable report
# ---------------------------------------------------------------------------

def _write_detailed_report(
    analyses: list[ModelAnalysis],
    block_names: list[str],
    path: str,
    common_name: str = "Common",
) -> None:
    """Write a comprehensive biosynthetic plausibility report to a text file."""
    W = 76
    lines: list[str] = []

    lines.append("=" * W)
    lines.append("GlycanSolver — Biosynthetic Plausibility Report")
    lines.append("=" * W)
    lines.append("")
    lines.append(f"Blocks analysed: {', '.join(block_names) if block_names else '(none)'}")
    lines.append(f"Models evaluated: {len(analyses)}")
    lines.append("")

    for rank, ma in enumerate(analyses, 1):
        ns = ma.network
        is_consensus = ma.model in ("Consensus", "BioConsensus", "BioConsensus2")
        lines.append("-" * W)
        lines.append(f"Rank {rank}: {ma.model}")
        if not is_consensus and ma.model_bic is not None:
            lines.append(f"BIC  : {ma.model_bic:.2f}")
        lines.append(f"Score: {ma.score:.4f}  (lower = more plausible)")
        lines.append(f"GOOD peaks: {ma.n_good}")
        lines.append("-" * W)

        # ---- full network stats ----
        lines.append("")
        lines.append("  FULL PAIRWISE NETWORK")
        lines.append("  " + "~" * 50)
        lines.append(f"  Total peak pairs     : {ns.total_edges}")
        lines.append(f"  Clean edges (L₁=1)   : {ns.n_clean}")
        lines.append(f"  Skip edges           : {ns.n_skip}")
        lines.append(f"  Multi edges          : {ns.n_multi}")
        lines.append(f"  Mixed edges          : {ns.n_mixed}")
        lines.append(f"  Identity edges (L₁=0): {ns.n_identity}")
        lines.append(f"  Clean density        : {ns.clean_density:.4f}  "
                      f"({ns.clean_density * 100:.1f}%)")
        lines.append(f"  Avg clean degree     : {ns.avg_clean_degree:.2f}")
        lines.append(f"  Clean components     : {ns.n_clean_components}")
        lines.append(f"  Largest clean comp   : {ns.largest_clean_component}/{ma.n_good}")
        lines.append(f"  Clean reachability   : {ns.clean_reachability:.3f}  "
                      f"({ns.clean_reachability * 100:.1f}%)")
        lines.append("")

        # ---- MST backbone ----
        lines.append("  MST BACKBONE")
        lines.append("  " + "~" * 50)
        lines.append(f"  MST edges            : {ma.n_mst_edges}")
        lines.append(f"    Clean              : {ma.mst_clean}")
        lines.append(f"    Skip               : {ma.mst_skip}")
        lines.append(f"    Multi              : {ma.mst_multi}")
        lines.append(f"    Mixed              : {ma.mst_mixed}")
        lines.append(f"    Identity           : {ma.mst_identity}")
        lines.append(f"  Mean penalty         : {ma.mst_mean_penalty:.4f}")
        lines.append(f"  Max L₁ in MST        : {ma.mst_max_l1}")
        lines.append("")

        # ---- scoring breakdown ----
        mixed_frac = ma.mst_mixed / ma.n_mst_edges if ma.n_mst_edges else 0.0
        lines.append("  SCORE BREAKDOWN")
        lines.append("  " + "~" * 50)
        lines.append(f"  mst_mean_penalty     : {ma.mst_mean_penalty:.4f}")
        lines.append(f"  (2 − reachability)   : {2.0 - ns.clean_reachability:.4f}")
        lines.append(f"  (1 + mixed_frac)     : {1.0 + mixed_frac:.4f}")
        lines.append(f"  Product = score      : {ma.score:.4f}")
        lines.append("")

        # ---- per-peak details ----
        lines.append("  PER-PEAK DETAILS")
        lines.append("  " + "~" * 50)
        lines.append(f"  {'#':<4} {'Peak':>6} {'Mass':>12} {'Composition':<30} "
                      f"{'Clean°':>7} {'Extra':>6}")
        lines.append("  " + "-" * 70)

        for i, peak in enumerate(ma.sorted_peaks):
            formula = _formula_label(peak["composition"], block_names)
            clean_deg = ma.node_clean_degree[i]
            # Count extra clean edges (not in MST)
            mst_nbs = {nb for nb, _ in ma.mst_adjacency.get(i, [])}
            extra = sum(1 for nb, _ in ma.clean_adjacency.get(i, [])
                        if nb not in mst_nbs)
            lines.append(
                f"  {i + 1:<4} {peak['peak_id']:>6} {peak['observed']:>12.5f} "
                f"{formula:<30} {clean_deg:>7} {extra:>6}"
            )
        lines.append("")

        # ---- MST tree (text-based) ----
        chain = _build_tree_chain(ma, block_names, common_name)
        if chain:
            lines.append("  BIOSYNTHETIC TREE (rooted at common block)")
            lines.append("  " + "~" * 50)
            active_levels: set[int] = set()
            for idx_c, node in enumerate(chain):
                # Check if last sibling
                is_last = True
                for j in range(idx_c + 1, len(chain)):
                    if chain[j]["depth"] < node["depth"]:
                        break
                    if chain[j]["depth"] == node["depth"]:
                        is_last = False
                        break

                indent = "  "
                if node["depth"] > 0:
                    for d in range(1, node["depth"]):
                        indent += "│   " if d in active_levels else "    "
                    indent += "└── " if is_last else "├── "
                    if is_last:
                        active_levels.discard(node["depth"])
                    else:
                        active_levels.add(node["depth"])

                trans = ""
                if node.get("transition"):
                    t = node["transition"]
                    addition = t.get('addition_label', t['delta_label'])
                    trans = f"[{addition}, {t['kind']}, L₁={t['l1']}] → "

                deg_str = f"(°{node['clean_degree']})"
                extra_str = ""
                if node.get("extra_clean"):
                    n_extra = len(node["extra_clean"])
                    extra_str = f" +{n_extra} extra clean"

                lines.append(
                    f"{indent}{trans}{node['formula']}  "
                    f"{node['mass']:.2f}  {deg_str}{extra_str}"
                )
            lines.append("")

        # ---- clean edge list (abbreviated) ----
        if ma.all_clean_edges:
            lines.append("  CLEAN EDGES (L₁=1 connections, showing additions)")
            lines.append("  " + "~" * 50)
            max_show = 50
            for ei, e in enumerate(ma.all_clean_edges[:max_show]):
                p_from = ma.sorted_peaks[e.from_idx]
                p_to = ma.sorted_peaks[e.to_idx]
                addition = _delta_label_additions(e.delta, block_names)
                lines.append(
                    f"  Peak {p_from['peak_id']:>3} ({p_from['observed']:.3f}) "
                    f"→ Peak {p_to['peak_id']:>3} ({p_to['observed']:.3f}) : "
                    f"{addition}"
                )
            if len(ma.all_clean_edges) > max_show:
                lines.append(f"  ... and {len(ma.all_clean_edges) - max_show} more")
            lines.append("")

        lines.append("")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
