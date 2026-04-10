from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

# Absolute path to the bundled data directory inside this package
DATA_DIR = Path(__file__).resolve().parent / "data"


def _read_local_version() -> str:
	pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
	with pyproject_path.open("rb") as fh:
		data = tomllib.load(fh)
	return data["project"]["version"]


try:
	__version__ = _read_local_version()
except FileNotFoundError:
	try:
		__version__ = package_version("glycansolver")
	except PackageNotFoundError:
		__version__ = "unknown"

