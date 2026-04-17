# Glycansolver

Glycansolver is a Python package for fitting MALDI glycan spectra with a shared common block, known differential blocks, and optionally discovered unknown blocks. It provides both a command-line interface and a web UI.

## Requirements

- Python 3.14
- A working Gurobi installation and license
- The dependencies declared in `pyproject.toml`

This repository uses a `src/` layout. The package can be run either after installation or by launching Gunicorn with `--pythonpath src`.

## Installation

If you are using an existing conda or mamba environment:

```bash
mamba create -n glycansolver python=3.14
mamba activate glycansolver
pip install -e .
```

If you prefer `uv` inside the active environment:

```bash
mamba activate glycansolver
uv pip install -e .
```

The package installs two console entrypoints:

- `glycansolver` for the CLI solver
- `glycansolver-web` for the Flask web app

## Input Data

The solver accepts:

- Plain-text peak files with one mass per line
- `.msd` files

Sample inputs are available under `src/glycansolver/data/`.

## Command-Line Usage

Show the full CLI help:

```bash
glycansolver --help
```

Minimal example using a plain-text peak list:

```bash
glycansolver -p src/glycansolver/data/example.txt
```

Example with explicit known blocks and up to two discovered blocks:

```bash
glycansolver \
	-p src/glycansolver/data/example.txt \
	-n "Hex,Fuc,HexNAc" \
	-u 2
```

Example using an MSD input file and a custom tolerance setup:

```bash
glycansolver \
	-p file.msd \
	-t 0.4 \
	-f 0.6 \
	-b 3
```

### Commonly Used CLI Options

- `-p, --peaks`: input peak file (`.txt` or `.msd`)
- `-o, --output`: output directory; if omitted, one is derived from the input filename
- `-n, --names`: comma-separated known block names
- `-u, --unknown`: maximum number of discovered blocks to test
- `-c, --common`: common block mass; if omitted, a default is computed from mode and matrix
- `-D, --mode`: ionization mode (`neg_h`, `pos_na`, `pos_h`)
- `-X, --matrix`: matrix / reduction mode (`2aa_nr`, `2aa_r`)
- `-d, --blocks-dict`: path to the block dictionary, default `src/glycansolver/data/blocks.txt`
- `-e, --exclude`: block names or masses to exclude from candidate discovery
- `--exhaustive`: exhaustive comparison level (`0`, `1`, or `2`)
- `-T, --timeout`: maximum runtime in minutes
- `-v, --verbose`: verbose console output
- `-C, --candidates_only`: stop after generating the candidates list

### CLI Output Files

The solver writes results into the selected output directory. Typical files include:

- `results.tsv`: per-peak reconstructions and formulas
- `blocks.tsv`: common block and differential blocks used in the final model
- `candidates.tsv`: candidate masses found during unknown-block discovery
- `diagnostics_report.txt`: residual and block-usage diagnostics
- `biosynthetic_summary.tsv`: biosynthetic plausibility summary
- `block_dependencies.tsv`: inferred block dependency report when exhaustive comparison is enabled

The exact set depends on the chosen options and whether the solve reaches the later analysis phases.

## Web App Usage

Start the built-in Flask server:

```bash
glycansolver-web
```

By default, it listens on `0.0.0.0:5000`.

For local development you can also run the module directly:

```bash
python -m glycansolver.web
```

For deployment behind Gunicorn after installation:

```bash
gunicorn --workers 1 -b 0.0.0.0:5000 glycansolver.web:app
```

If you want to run from the repository without installing the package first:

```bash
gunicorn --workers 1 --pythonpath src -b 0.0.0.0:5000 glycansolver.web:app
```

### Web Workflow

1. Upload a peak file or paste peaks into the text area.
2. Choose glycan type, label, mode, reduction, and block settings.
3. Run candidate discovery to inspect optional unknown blocks.
4. Run the full solve.
5. Download the ZIP archive containing all generated outputs.

The web app enforces a 20 MB upload size limit.

## Docker

Build the image:

```bash
docker build -t glycansolver .
```

Run the container:

```bash
docker run --rm -p 5000:5000 glycansolver
```

The image starts Gunicorn on port 5000.

To persist usage counters outside the container, bind-mount the database file at `/app/usage.db`:

```bash
touch usage.db
docker run --rm -p 5000:5000 \
	-v "$PWD/usage.db:/app/usage.db" \
	glycansolver
```

An empty file is fine. On startup, the app initializes the SQLite schema with `CREATE TABLE IF NOT EXISTS`, so the file will be populated automatically as events are recorded. The mounted file must be writable by the container.

The image also sets `GLYCANSOLVER_USAGE_DB=/app/usage.db` explicitly so the web app writes to the mounted file even when the package is installed under `site-packages`.

### Docker With Gurobi WLS

If you use a Gurobi Web License Service (WLS) license, you do **not** need to run `grbgetkey` or mount a `gurobi.lic` file.
Pass the WLS credentials as runtime environment variables:

```bash
docker run --rm -p 5000:5000 \
	-e GRB_WLSACCESSID="<your-access-id>" \
	-e GRB_WLSSECRET="<your-secret>" \
	-e GRB_LICENSEID="<your-license-id>" \
	glycansolver
```

Avoid putting these values in the `Dockerfile`. Inject them at runtime (or via Docker secrets / your orchestrator secret manager).

## Package Layout

- `src/glycansolver/cli.py`: command-line entrypoint
- `src/glycansolver/web.py`: Flask application and web routes
- `src/glycansolver/solve_progressive.py`: main progressive solver
- `src/glycansolver/utils.py`: I/O and reporting helpers
- `src/glycansolver/data/blocks.txt`: block dictionary used by the solver and web UI
- `src/glycansolver/data/labels.txt`: derivatization label definitions

## Notes

- The CLI can infer mode and matrix defaults from `.msd` filenames when the naming pattern matches the expected convention.
- The web app stores temporary uploads and outputs under the system temporary directory.
- `usage.db` is ignored by Git and stores local usage counters for the web UI.

## Usage Data

The web app records usage events in `usage.db` in the current working directory by default. You can override this with the `GLYCANSOLVER_USAGE_DB` environment variable. The database contains a single `events` table with these columns:

- `id`: autoincrement primary key
- `kind`: event type such as `visit`, `find_candidates`, or `solve`
- `ts`: UTC timestamp in ISO 8601 format

When running in Docker, this path is `/app/usage.db`. You can bind-mount a host file there to persist the counters across container restarts.

Useful queries:

```bash
# Total counts by event kind
sqlite3 usage.db "SELECT kind, COUNT(*) FROM events GROUP BY kind ORDER BY kind;"

# First event, last event, and total rows
sqlite3 usage.db "SELECT MIN(ts), MAX(ts), COUNT(*) FROM events;"

# Most recent 20 events
sqlite3 usage.db "SELECT kind, ts FROM events ORDER BY id DESC LIMIT 20;"

# Daily counts by event kind
sqlite3 usage.db "SELECT substr(ts, 1, 10) AS day, kind, COUNT(*) FROM events GROUP BY day, kind ORDER BY day DESC, kind;"

# Busiest days overall
sqlite3 usage.db "SELECT substr(ts, 1, 10) AS day, COUNT(*) AS total_events FROM events GROUP BY day ORDER BY total_events DESC, day DESC LIMIT 5;"
```
