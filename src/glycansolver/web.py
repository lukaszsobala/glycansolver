"""
Glycan Solver – Web Interface

A Flask application that wraps the progressive solver with a two-step
workflow:

  1. **Find Candidates** — runs the solver in candidates-only mode so the
     user can inspect (and select/deselect) the discovered candidate
     blocks before committing to a full optimisation.
  2. **Solve** — runs the full progressive algorithm with the user's
     chosen parameters and candidate selections.
"""

from __future__ import annotations

import csv
import io
import os
import queue
import re
import sys
import secrets
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
)

from . import __version__
from .utils import compute_common_mass, load_labels
from . import usage_counter

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET") or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB upload limit

# Where uploads and solver output go
WORK_DIR = Path(tempfile.gettempdir()) / "glycansolver_web"
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Initialise the usage counter (SQLite DB next to the project root)
usage_counter.init()

# Blocks dictionary shipped with the package
_PKG_DIR = Path(__file__).resolve().parent
_DATA_DIR = _PKG_DIR / "data"
_BLOCKS_DICT = _DATA_DIR / "blocks.txt"
_LABELS_FILE = _DATA_DIR / "labels.txt"
_GUROBI_LOGO = _DATA_DIR / "Gurobi Optimization_idVv0Kw2L1_1.svg"

# Default common-block sugar composition for preset mode
DEFAULT_COMMON_COMPOSITION: dict[str, int] = {"Hex": 3, "HexNAc": 2}

# In-memory job store  {job_id: {status, progress, result, ...}}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Single-worker job queue – ensures only one solve runs at a time
# ---------------------------------------------------------------------------
_job_queue: queue.Queue[str | None] = queue.Queue()


def _queue_position(job_id: str) -> int:
    """Return 1-based position of *job_id* among queued jobs, or 0 if not queued."""
    with _jobs_lock:
        pos = 1
        for jid, info in _jobs.items():
            if info.get("status") == "queued":
                if jid == job_id:
                    return pos
                pos += 1
    return 0


def _queue_length() -> int:
    """Return the number of jobs currently queued (not yet running)."""
    with _jobs_lock:
        return sum(1 for j in _jobs.values() if j.get("status") == "queued")


def _worker_loop() -> None:
    """Background thread that processes jobs one at a time."""
    while True:
        job_id = _job_queue.get()
        if job_id is None:          # poison pill
            break
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job or job.get("status") != "queued":
                _job_queue.task_done()
                continue
            job["status"] = "running"
            runner = job.get("_runner")
        if runner:
            runner()
        _job_queue.task_done()


_worker_thread = threading.Thread(target=_worker_loop, daemon=True)
_worker_thread.start()


class _JobLogWriter:
    """Thread-safe stdout writer that appends text to a job's live log."""

    def __init__(self, job_id: str):
        self.job_id = job_id

    def write(self, s: str):
        if not s:
            return 0
        with _jobs_lock:
            job = _jobs.get(self.job_id)
            if job is not None:
                job["log"] = (job.get("log", "") + s)
        return len(s)

    def flush(self):
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_blocks_dict(glycan_type: str | None = None) -> list[dict]:
    """Return the blocks dictionary as a list of dicts.

    Parameters
    ----------
    glycan_type : str or None
        If given, only return blocks matching this glycan type
        (e.g. ``"native"`` or ``"permethylated"``).  ``None`` returns all.
    """
    blocks = []
    path = _BLOCKS_DICT
    if not path.exists():
        return blocks
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                name = parts[0].strip()
                mass = float(parts[1].strip())
                cat = parts[2].strip() if len(parts) >= 3 else "unknown"
                default_limit = 10 if cat == "common" else 4
                max_limit = int(parts[3].strip()) if len(parts) >= 4 and parts[3].strip().isdigit() else default_limit
                gtype = parts[4].strip() if len(parts) >= 5 else "native"
                if glycan_type and gtype != glycan_type:
                    continue
                if cat == "meta":
                    continue  # skip metadata rows (e.g. _free_red_end)
                full_name = parts[5].strip() if len(parts) >= 6 else name
                blocks.append({"name": name, "mass": mass, "category": cat,
                               "max_limit": max_limit, "glycan_type": gtype,
                               "full_name": full_name})
    return blocks


def _load_free_red_end_masses() -> dict[str, float]:
    """Return ``{glycan_type: mass}`` for ``_free_red_end`` meta rows in blocks.txt."""
    result: dict[str, float] = {}
    path = _BLOCKS_DICT
    if not path.exists():
        return result
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3 and parts[0].strip() == "_free_red_end":
                gtype = parts[4].strip() if len(parts) >= 5 else "native"
                result[gtype] = float(parts[1].strip())
    return result


def _parse_candidates_tsv(path: str, glycan_type: str | None = None) -> list[dict]:
    """Parse the candidates.tsv produced by the solver."""
    candidates = []
    if not os.path.exists(path):
        return candidates
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if row.get("Status", "").strip() != "ACTIVE":
                continue
            candidates.append({
                "median": float(row["Median"]),
                "score": float(row["Score"]),
                "size": int(row["Size"]),
                "name": "",  # will be resolved below
            })
    # Try to name candidates from the blocks dict (filtered by glycan type)
    bd = _load_blocks_dict(glycan_type)
    for c in candidates:
        for b in bd:
            if abs(c["median"] - b["mass"]) < 0.5:
                c["name"] = b["name"]
                break
    return candidates


def _parse_results_tsv(path: str) -> tuple[list[str], list[list[str]]]:
    """Return (headers, rows) from a results.tsv file."""
    headers: list[str] = []
    rows: list[list[str]] = []
    if not os.path.exists(path):
        return headers, rows
    with open(path) as fh:
        for i, line in enumerate(fh):
            cols = line.rstrip("\n").split("\t")
            if i == 0:
                headers = cols
            else:
                rows.append(cols)
    return headers, rows


def _parse_blocks_tsv(path: str) -> list[dict]:
    """Return list of block dicts from blocks.tsv."""
    blocks = []
    if not os.path.exists(path):
        return blocks
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            blocks.append({
                "name": row.get("Block", ""),
                "value": row.get("Value", ""),
                "type": row.get("Type", ""),
            })
    return blocks


def _parse_block_dependencies(path: str) -> dict | None:
    """Parse block_dependencies.tsv into a JSON-friendly structure."""
    blocks = []
    edges = []
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            name = row.get("Block", "")
            direct = row.get("Direct_Dependencies", "(root)")
            n_with = int(row.get("N_Models_With_Block", 0))
            n_used = int(row.get("N_Models_Used", 0))
            blocks.append({
                "name": name,
                "n_models_with": n_with,
                "n_models_used": n_used,
            })
            if direct and direct != "(root)":
                for parent in direct.split("+"):
                    parent = parent.strip()
                    if parent:
                        edges.append({"parent": parent, "child": name})
    if not blocks:
        return None
    return {"blocks": blocks, "edges": edges}


def _save_peaks(text: str | None, file_storage=None) -> str | None:
    """Save peaks to a temp file.  Return the path or None.

    Pasted text always takes priority over an uploaded file.
    """
    job_dir = WORK_DIR / str(uuid.uuid4())
    job_dir.mkdir(parents=True, exist_ok=True)

    if text and text.strip():
        dest = job_dir / "peaks.txt"
        # Clean up pasted text: one number per line
        lines = []
        for line in text.strip().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
        dest.write_text("\n".join(lines) + "\n")
        return str(dest)

    if file_storage and file_storage.filename:
        ext = Path(file_storage.filename).suffix.lower()
        dest = job_dir / f"peaks{ext}"
        file_storage.save(str(dest))
        return str(dest)

    return None


_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _parse_text_peaks(content: str) -> list[float]:
    """Extract peak masses from plain-text content.

    Supports one value per line as well as CSV/TSV-like rows by taking the
    first parseable numeric token from each non-comment line.
    """
    peaks: list[float] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Fast path: full-line float
        try:
            peaks.append(float(line))
            continue
        except ValueError:
            pass

        # Fallback: scan separators and keep first parseable token
        token = None
        for part in re.split(r"[\t,; ]+", line):
            part = part.strip()
            if not part:
                continue
            try:
                peaks.append(float(part))
                token = part
                break
            except ValueError:
                continue

        # Last resort: regex float extraction
        if token is None:
            m = _FLOAT_RE.search(line)
            if m:
                peaks.append(float(m.group(0)))

    return peaks


def _parse_msd_peaks(content: str) -> list[float]:
    """Extract mz values from MSD text."""
    peaklist_match = re.search(r"<peaklist>(.*?)</peaklist>", content, re.DOTALL | re.IGNORECASE)
    if not peaklist_match:
        return []

    peaklist_content = peaklist_match.group(1)
    peaks: list[float] = []
    for peak_match in re.finditer(r"<peak\s+([^>]+)", peaklist_content, re.IGNORECASE):
        peak_attrs = peak_match.group(1)
        mz_match = re.search(r'mz="([^"]+)"', peak_attrs, re.IGNORECASE)
        if not mz_match:
            continue
        try:
            peaks.append(float(mz_match.group(1)))
        except ValueError:
            continue
    return peaks


def _extract_uploaded_peaks(file_storage) -> list[float]:
    """Read and parse peaks from an uploaded file object."""
    if not file_storage or not file_storage.filename:
        return []

    filename = file_storage.filename
    ext = Path(filename).suffix.lower()

    raw = file_storage.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("latin-1", errors="ignore")

    # Reset stream in case anything else needs to read from it later.
    file_storage.stream.seek(0)

    if ext == ".msd":
        return _parse_msd_peaks(content)
    return _parse_text_peaks(content)


def _build_solver_kwargs(form: dict, peaks_path: str, output_dir: str,
                         candidates_only: bool = False) -> dict:
    """Translate form fields into kwargs for solve_progressive()."""
    # ---- Build block-mass lookup from blocks.txt ----
    glycan_type = form.get("glycan_type", "native")
    all_blocks = _load_blocks_dict(glycan_type)
    block_masses = {b["name"]: b["mass"] for b in all_blocks}

    # ---- Common block from composition ----
    common_mode = form.get("common_mode", "composition")
    if common_mode == "manual":
        common = float(form.get("common_manual", 1028.357059))
        common_composition = None
    else:
        # Parse composition from form fields (common_Hex, common_HexNAc, …)
        composition: dict[str, int] = {}
        has_composition_field = False
        for bname in block_masses:
            field_key = f"common_{bname}"
            if field_key in form:
                has_composition_field = True
            cnt = int(form.get(field_key, 0))
            if cnt > 0:
                composition[bname] = cnt
        # Only fall back to default if NO composition fields were present
        # in the form at all (first visit).  If the user explicitly set all
        # counts to 0 (e.g. "2-AA only"), honour that empty composition.
        if not composition and not has_composition_field:
            composition = dict(DEFAULT_COMMON_COMPOSITION)

        # Label
        label_name = form.get("label", "2-AA")
        if label_name == "free_red_end":
            label_mass = _load_free_red_end_masses().get(glycan_type, 0.0)
        else:
            labels = load_labels(str(_LABELS_FILE))
            label_mass = 0.0
            for lbl in labels:
                if lbl["name"] == label_name:
                    label_mass = lbl["mass"]
                    break

        # Polarity & reduction
        polarity = form.get("mode", "neg_h")
        reduction = form.get("reduction", "nr")

        common = compute_common_mass(
            composition, block_masses, label_mass, polarity, reduction,
        )
        common_composition = composition

    # ---- Known blocks: build -n string from tri-state select ----
    block_entries = []
    masses_entries = []  # parallel list for custom blocks that need explicit mass
    exclude_entries = []
    for b in all_blocks:
        bname = b["name"]
        state = form.get(f"block_{bname}_state", "exclude")
        if state == "exclude":
            exclude_entries.append(bname)
        elif state == "use":
            limit = form.get(f"block_{bname}_limit", "")
            if limit:
                block_entries.append(f"{bname}:{limit}")
            else:
                block_entries.append(bname)
            masses_entries.append(None)  # resolved from dict
        # state == "discover" → neither known nor excluded

    # ---- Custom (user-added) blocks ----
    custom_names = []
    for key in sorted(form.keys()):
        if key.endswith("_custom") and form[key] == "1":
            cname = key.replace("block_", "").replace("_custom", "")
            state = form.get(f"block_{cname}_state", "exclude")
            # Skip if already handled above (shouldn't happen) or excluded
            if any(e.split(":")[0] == cname for e in block_entries):
                continue
            cmass = form.get(f"block_{cname}_mass", "")
            if not cmass:
                continue
            if state == "exclude":
                continue
            elif state == "use":
                limit = form.get(f"block_{cname}_limit", "4")
                block_entries.append(f"{cname}:{limit}")
                masses_entries.append(float(cmass))
                custom_names.append(cname)
            # state == "discover" → candidate, not known

    names_str = ",".join(block_entries) if block_entries else "Hex"

    # Build masses string: only needed when custom blocks are present.
    # Dictionary blocks are resolved by name; custom blocks need explicit mass.
    masses_str = None
    if custom_names:
        parts = []
        for entry, mass_val in zip(block_entries, masses_entries):
            if mass_val is not None:
                # Custom block — include mass (with optional :limit)
                parts.append(str(mass_val))
            else:
                # Dictionary block — omit (will be resolved from name)
                # We must provide a mass so indices align, so resolve it here
                bname = entry.split(":")[0]
                bmass = block_masses.get(bname, "")
                parts.append(str(bmass) if bmass else "0")
        masses_str = ",".join(parts)

    return {
        "peaks": peaks_path,
        "output": output_dir,
        "mode": form.get("mode", "neg_h"),
        "matrix": form.get("matrix", "2aa_nr"),
        "common": common,
        "common_composition": common_composition,
        "masses": masses_str,    # None unless custom blocks present
        "names": names_str,
        "unknown": int(form.get("max_unknown_blocks", 3)),
        "tolerance": float(form.get("tolerance", 0.3)),
        "final_tolerance": float(form.get("final_tolerance", 0.5)),
        "bad": int(form.get("bad", 0)),
        "max_known": int(form.get("max_known", 10)),
        "max_unknown": int(form.get("max_unknown_copies", 4)),
        "lower_bound": float(form.get("lower_bound", 35.0)),
        "upper_bound": float(form.get("upper_bound", 370.0)),
        "min_diff": 40.0,
        "candidates_only": candidates_only,
        "blocks_dict": str(_BLOCKS_DICT) if _BLOCKS_DICT.exists() else None,
        "postgoal": 20,
        "timeout": int(form.get("timeout", 15)),
        "verbose": False,
        "exclude": ",".join(exclude_entries) if exclude_entries else None,
        "exhaustive": int(form.get("exhaustive", "1")),
        "glycan_type": glycan_type,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    usage_counter.record("visit")
    labels = load_labels(str(_LABELS_FILE))

    # Discover all glycan types present in blocks.txt
    all_blocks = _load_blocks_dict()  # unfiltered
    glycan_types_set: set[str] = set()
    for b in all_blocks:
        glycan_types_set.add(b.get("glycan_type", "native"))
    # Ensure 'native' is always first
    glycan_types = ["native"] + sorted(glycan_types_set - {"native"})

    # Build per-type block lists and mass dicts
    blocks_by_type: dict[str, list[dict]] = {}
    block_masses_by_type: dict[str, dict[str, float]] = {}
    for gt in glycan_types:
        bl = _load_blocks_dict(gt)
        blocks_by_type[gt] = bl
        block_masses_by_type[gt] = {b["name"]: b["mass"] for b in bl}

    free_red_end_masses = _load_free_red_end_masses()

    return render_template(
        "index.html",
        blocks=blocks_by_type["native"],
        labels=labels,
        block_masses=block_masses_by_type["native"],
        default_composition=DEFAULT_COMMON_COMPOSITION,
        glycan_types=glycan_types,
        blocks_by_type=blocks_by_type,
        block_masses_by_type=block_masses_by_type,
        free_red_end_masses=free_red_end_masses,
        app_version=__version__,
    )


@app.route("/example_peaks")
def example_peaks():
    """Return the contents of the bundled example.txt peak list."""
    path = _DATA_DIR / "example.txt"
    if not path.exists():
        return jsonify({"error": "Example file not found"}), 404
    return send_file(path, mimetype="text/plain")


@app.route("/gurobi_logo.svg")
def gurobi_logo():
    """Serve the bundled Gurobi SVG used in the header."""
    if not _GUROBI_LOGO.exists():
        return jsonify({"error": "Logo not found"}), 404
    return send_file(_GUROBI_LOGO, mimetype="image/svg+xml")


@app.route("/find_candidates", methods=["POST"])
def find_candidates():
    """Step 1: run candidates-only and return the candidate list."""
    usage_counter.record("find_candidates")
    form = request.form.to_dict()
    peaks_path = _save_peaks(
        form.get("peaks_text"),
        request.files.get("peaks_file"),
    )
    if not peaks_path:
        return jsonify({"error": "No peaks provided."}), 400

    output_dir = str(WORK_DIR / str(uuid.uuid4()))
    os.makedirs(output_dir, exist_ok=True)

    kwargs = _build_solver_kwargs(form, peaks_path, output_dir,
                                  candidates_only=True)

    # Import here to avoid circular imports at module level
    from .solve_progressive import solve_progressive

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        solve_progressive(**kwargs)
    except Exception:
        pass
    finally:
        log = sys.stdout.getvalue()
        sys.stdout = old_stdout

    # Parse candidates
    cand_path = os.path.join(output_dir, "candidates.tsv")
    glycan_type = form.get("glycan_type", "native")
    candidates = _parse_candidates_tsv(cand_path, glycan_type)

    return jsonify({
        "candidates": candidates[:20],
        "output_dir": output_dir,
        "peaks_path": peaks_path,
        "log": log,
    })


@app.route("/extract_peaks", methods=["POST"])
def extract_peaks():
    """Parse uploaded peaks and return them for immediate textarea display."""
    file_storage = request.files.get("peaks_file")
    if not file_storage or not file_storage.filename:
        return jsonify({"error": "No file uploaded."}), 400

    peaks = _extract_uploaded_peaks(file_storage)
    if not peaks:
        return jsonify({"error": "No peaks could be extracted from this file."}), 400

    peaks_text = "\n".join(str(p) for p in peaks)
    return jsonify({
        "peaks_text": peaks_text,
        "count": len(peaks),
        "filename": file_storage.filename,
    })


@app.route("/solve", methods=["POST"])
def solve():
    """Step 2: run the full progressive solver (async via thread)."""
    usage_counter.record("solve")
    form = request.form.to_dict()

    peaks_path = form.get("peaks_path")
    if not peaks_path:
        # Re-save from form if coming fresh
        peaks_path = _save_peaks(
            form.get("peaks_text"),
            request.files.get("peaks_file"),
        )
    if not peaks_path:
        return jsonify({"error": "No peaks provided."}), 400

    output_dir = str(WORK_DIR / str(uuid.uuid4()))
    os.makedirs(output_dir, exist_ok=True)

    kwargs = _build_solver_kwargs(form, peaks_path, output_dir,
                                  candidates_only=False)

    # ---- Force-remove selected candidate blocks from the exclude list ----
    # Named candidate blocks that the user checked in the candidates panel
    # must NOT be excluded, even if their block-table Exclude checkbox was
    # still on in the form data (DOM sync can be unreliable).
    selected_candidates = form.get("selected_candidates", "")
    if selected_candidates and kwargs.get("exclude"):
        selected_set = {
            c.strip().lower()
            for c in selected_candidates.split(",")
            if c.strip()
        }
        exclude_parts = [e.strip() for e in kwargs["exclude"].split(",")]
        exclude_parts = [
            e for e in exclude_parts if e.lower() not in selected_set
        ]
        kwargs["exclude"] = ",".join(exclude_parts) if exclude_parts else None

    # ---- Add deselected mass-only candidates to exclude list ----
    existing_exclude = kwargs.get("exclude") or ""
    deselected = form.get("deselected_candidates", "")
    if deselected:
        extras = []
        for token in deselected.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                float(token)
                extras.append(token)
            except ValueError:
                continue
        if extras:
            if existing_exclude:
                kwargs["exclude"] = existing_exclude + "," + ",".join(extras)
            else:
                kwargs["exclude"] = ",".join(extras)

    # ---- Build protect list from selected candidates ----
    if selected_candidates:
        kwargs["protect"] = [
            c.strip() for c in selected_candidates.split(",") if c.strip()
        ]

    job_id = str(uuid.uuid4())
    # Derive a safe job name for the output ZIP
    raw_name = form.get("job_name", "").strip()
    salt = uuid.uuid4().hex[:8]
    if raw_name:
        # Sanitise: keep only alphanumeric, dash, underscore, dot
        safe_name = "".join(
            c if (c.isalnum() or c in "-_.") else "_" for c in raw_name
        )
    else:
        safe_name = "results"
    job_label = f"{safe_name}_{salt}"

    cancel_event = threading.Event()

    def _should_cancel() -> bool:
        return cancel_event.is_set()

    def _run():
        from .solve_progressive import solve_progressive
        from .solve_progressive import SolverCancelledError
        old_stdout = sys.stdout
        sys.stdout = _JobLogWriter(job_id)
        try:
            # Log the effective solver parameters for debugging
            print(f"[web] Known blocks (names): {kwargs.get('names', '(none)')}")
            print(f"[web] Exclude list: {kwargs.get('exclude') or '(none)'}")
            sel_cands = form.get("selected_candidates", "")
            if sel_cands:
                print(f"[web] Selected candidates: {sel_cands}")
            kwargs["should_cancel"] = _should_cancel
            solve_progressive(**kwargs)
            status = "done"
        except SolverCancelledError as exc:
            print(f"\n[web] {exc}")
            status = "cancelled"
        except Exception as exc:
            status = "error"
            with _jobs_lock:
                if job_id in _jobs:
                    _jobs[job_id]["log"] = _jobs[job_id].get("log", "") + f"\n\nERROR: {exc}\n"
        finally:
            sys.stdout = old_stdout
        # Save the console log to the output directory so it's included in the ZIP
        with _jobs_lock:
            log_text = _jobs.get(job_id, {}).get("log", "")
        if log_text:
            log_path = os.path.join(output_dir, "console_log.txt")
            try:
                with open(log_path, "w") as f:
                    f.write(log_text)
            except Exception:
                pass
        with _jobs_lock:
            _jobs[job_id]["status"] = status

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "output_dir": output_dir,
            "job_label": job_label,
            "log": "",
            "cancel_event": cancel_event,
            "cancel_requested": False,
            "_runner": _run,
        }
    _job_queue.put(job_id)

    return jsonify({"job_id": job_id, "output_dir": output_dir})


@app.route("/job_status/<job_id>")
def job_status(job_id):
    """Poll for solve job completion."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404

    resp = {
        "status": job["status"],
        "log": job.get("log", ""),
        "cancel_requested": bool(job.get("cancel_requested", False)),
    }
    if job["status"] == "queued":
        resp["queue_position"] = _queue_position(job_id)
        resp["queue_length"] = _queue_length()
    if job["status"] in ("done", "error", "cancelled"):
        output_dir = job["output_dir"]

        # Results
        results_path = os.path.join(output_dir, "results.tsv")
        headers, rows = _parse_results_tsv(results_path)
        resp["results_headers"] = headers
        resp["results_rows"] = rows

        # Blocks
        blocks_path = os.path.join(output_dir, "blocks.tsv")
        resp["blocks"] = _parse_blocks_tsv(blocks_path)

        # Biosynthetic plausibility analysis
        biosyn_path = os.path.join(output_dir, "biosynthetic_summary.tsv")
        if os.path.exists(biosyn_path):
            from .biosynthetic import analyse_biosynthetic_paths
            try:
                resp["biosynthetic"] = analyse_biosynthetic_paths(
                    results_path, output_dir
                )
            except Exception:
                resp["biosynthetic"] = None

        # Block dependency tree
        dep_path = os.path.join(output_dir, "block_dependencies.tsv")
        if os.path.exists(dep_path):
            try:
                resp["block_dependencies"] = _parse_block_dependencies(dep_path)
            except Exception:
                resp["block_dependencies"] = None

        # Model diagnostics (residuals + block usage)
        from .diagnostics import run_diagnostics
        try:
            resp["diagnostics"] = run_diagnostics(results_path, output_dir)
        except Exception:
            resp["diagnostics"] = None

        resp["download_url"] = f"/download_results/{job_id}"

    return jsonify(resp)


@app.route("/stop_job/<job_id>", methods=["POST"])
def stop_job(job_id):
    """Request cooperative cancellation for a running solve job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"error": "Unknown job"}), 404

        if job.get("status") != "running":
            return jsonify({"status": job.get("status")}), 200

        cancel_event = job.get("cancel_event")
        if cancel_event is not None:
            cancel_event.set()
        job["cancel_requested"] = True
        job["status"] = "stopping"
        job["log"] = job.get("log", "") + "\n[web] Stop requested by user. Waiting for safe cancellation...\n"

    return jsonify({"status": "stopping"})


@app.route("/download_results/<job_id>")
def download_results(job_id):
    """Download all solver outputs for a completed job as a ZIP file."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    if job.get("status") not in ("done", "cancelled"):
        return jsonify({"error": "Job not finished yet"}), 400

    output_dir = Path(job["output_dir"])
    if not output_dir.exists() or not output_dir.is_dir():
        return jsonify({"error": "Output directory not found"}), 404

    job_label = job.get("job_label", f"results_{job_id[:8]}")
    zip_name = f"{job_label}_glycansolver.zip"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(output_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(output_dir))
    buf.seek(0)

    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=zip_name,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_web(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_web(debug=True)
