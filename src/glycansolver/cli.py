import argparse
import os
import sys
from typing import Literal, TypeAlias, cast, get_args

from . import DATA_DIR, __version__
from .solve_progressive import solve_progressive
from .utils import compute_common_mass

ModeType: TypeAlias = Literal["pos_na", "pos_h", "neg_h"]
MatrixType: TypeAlias = Literal["2aa_nr", "2aa_r"]

# Default common-block sugar composition (N-glycan core)
DEFAULT_COMMON_COMPOSITION: dict[str, int] = {"Hex": 3, "HexNAc": 2}

# Block masses needed for common-block computation (from blocks.txt defaults)
_BLOCK_MASSES: dict[str, float] = {
    "Hex": 162.052824,
    "HexNAc": 203.079374,
}

# Default label
_DEFAULT_LABEL_MASS = 137.047679   # 2-AA


# Function to extract mode and matrix from MSD filename
def extract_mode_matrix_from_filename(filename: str) -> tuple[str | None, str | None]:
    """
    Extracts mode and matrix information from an MSD filename.

    Mode detection:
    - "LN_" or "RN_" in capitals: negative mode
    - "LP_" or "RP_" in capitals: positive mode

    Matrix detection:
    - "_2aa" or "_2AA": matrix as 2aa_nr

    Returns a tuple of (mode, matrix) where:
    - mode is 'pos_na' or 'neg_h'
    - matrix is '2aa_nr' or '2aa_r'

    If a parameter cannot be determined, None is returned for that parameter.
    """
    basename = os.path.basename(filename)

    # Detect mode
    mode = None
    if "LN_" in basename or "RN_" in basename:
        mode = "neg_h"
    elif "LP_" in basename or "RP_" in basename:
        mode = "pos_na"

    # Detect matrix
    matrix = None
    if "_2aa" in basename.lower():
        matrix = "2aa_nr"

    return mode, matrix


def determine_common_block(
    mode: ModeType,
    matrix: MatrixType,
    composition: dict[str, int] | None = None,
    label_mass: float | None = None,
    blocks_dict_path: str | None = None,
) -> tuple[float, dict[str, int]]:
    """Return ``(mass, composition)`` for the common block.

    If *composition* is provided, the mass is computed from it using
    ``compute_common_mass``.  Otherwise the default N-glycan core
    (3Hex + 2HexNAc) is used.
    """
    if composition is None:
        composition = dict(DEFAULT_COMMON_COMPOSITION)
    if label_mass is None:
        label_mass = _DEFAULT_LABEL_MASS

    # Try to load block masses from the dictionary if available
    block_masses = dict(_BLOCK_MASSES)
    if blocks_dict_path and os.path.exists(blocks_dict_path):
        with open(blocks_dict_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    block_masses[parts[0].strip()] = float(parts[1].strip())

    # Derive polarity and reduction from mode/matrix
    polarity = mode                                     # "neg_h", "pos_na" or "pos_h"
    reduction = matrix.split("_")[-1] if "_" in matrix else "nr"  # "nr" or "r"

    mass = compute_common_mass(composition, block_masses, label_mass,
                               polarity, reduction)
    return mass, composition


def parse_arguments():
    description = (
        f"Glycan Solver {__version__}\n\n"
        "Glycan Solver: A tool to solve for building blocks in mass spectrometry data.\n"
        "The solver uses known building blocks and discovers "
        "unknown blocks that best explain the observed peak patterns.\n"
        "It uses gurobipy - a Python implementation of the Gurobi optimizer."
    )

    epilog = (
        "Examples:\n"
        '  glycansolver -p peaks.txt -n "Hex,Fuc,HexNAc" -u 2\n'
        "  glycansolver -p data.msd -c 1000.5 -b 3 -t 0.4 -f 0.6\n\n"
        "Output files will include:\n"
        "  - results.tsv: Contains all peak reconstructions and their formulas\n"
        "  - blocks.tsv: Contains the common block and all differential blocks\n"
        "  - candidates.tsv: Contains candidate values for unknown blocks"
    )

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # DATA-RELATED OPTIONS
    parser.add_argument(
        "-p",
        "--peaks",
        default="peaks.txt",
        help="File with peak observations (one per line); also supports .msd format",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory for results (default: derived from input filename)",
    )
    parser.add_argument(
        "-D",
        "--mode",
        choices=get_args(ModeType),
        default="neg_h",
        help=(
            "Ionization mode: positive sodium adduct (pos_na = [M+Na]+), "
            "positive proton adduct (pos_h = [M+H]+), or negative (neg_h = [M-H]-) "
            "(default: neg_h)"
        ),
    )
    parser.add_argument(
        "-X",
        "--matrix",
        choices=get_args(MatrixType),
        default="2aa_nr",
        help="MALDI matrix used, only 2-AA implemented so far, "
        "nonreductive or reductive: 2aa_nr or 2aa_r (default: 2aa_nr)",
    )
    parser.add_argument(
        "-d",
        "--blocks-dict",
        default=str(DATA_DIR / "blocks.txt"),
        help="Path to a dictionary file of known blocks in TSV format: name mass (default: bundled blocks.txt)",
    )
    parser.add_argument(
        "-c",
        "--common",
        type=float,
        default=None,
        help="Mass of the common block that appears in all observations (default depends on mode/matrix). "
        "neg_h/2aa_nr: 1028.357059, neg_h/2aa_r: 1026.341379, "
        "pos_na/2aa_nr: 1052.354668, pos_na/2aa_r: 1050.338988, "
        "pos_h/2aa_nr: 1030.372739, pos_h/2aa_r: 1028.357059. "
        "Use a negative value to estimate from minimum observation.",
    )
    parser.add_argument(
        "-n",
        "--names",
        default="Hex",
        help="Comma-separated list of names for known blocks (default: Hex)",
    )

    # ALGORITHM-RELATED OPTIONS
    parser.add_argument(
        "-u",
        "--unknown",
        type=int,
        default=3,
        help="Maximum number of unknown differential blocks to search for. "
        "The solver tries from 0 up to this many and uses BIC to pick the "
        "best model. (default: 3)",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=0.3,
        help="Maximum reconstruction error per observation during optimization (default: 0.3)",
    )
    parser.add_argument(
        "-f",
        "--final_tolerance",
        type=float,
        default=0.5,
        help="Maximum reconstruction error for convergence (default: 0.5)",
    )
    parser.add_argument(
        "-b",
        "--bad",
        type=int,
        default=0,
        help="Number of observations allowed to exceed the final tolerance (default: 0)",
    )
    parser.add_argument(
        "-A",
        "--postgoal",
        type=int,
        default=20,
        help="Number of additional iterations to perform after reaching the specified number of bad blocks (default:20)",
    )
    parser.add_argument(
        "-K",
        "--max_known",
        type=int,
        default=10,
        help="Maximum number of copies of a known block in any observation (default: 10)",
    )
    parser.add_argument(
        "-U",
        "--max_unknown",
        type=int,
        default=4,
        help="Maximum number of copies of an unknown/discovered block in any observation (default: 4)",
    )
    parser.add_argument(
        "-L",
        "--lower_bound",
        type=float,
        default=35.0,
        help="Lower bound for unknown blocks mass (default: 35)",
    )
    parser.add_argument(
        "-M",
        "--upper_bound",
        type=float,
        default=370.0,
        help="Upper bound for unknown blocks mass (default: 370)",
    )
    parser.add_argument(
        "-I",
        "--min_diff",
        type=float,
        default=40.0,
        help="Minimum difference threshold to consider when finding block candidates (default: 40)",
    )
#    parser.add_argument(
#        "-F",
#        "--first_unknown_as_known",
#        action="store_false",
#        help="Do not treat the first unknown block as a known block (default: True) UNUSED",
#    )
    parser.add_argument(
        "-C",
        "--candidates_only",
        action="store_true",
        help="Stop after generating and outputting the candidate list",
    )
    parser.add_argument(
        "-T",
        "--timeout",
        type=int,
        default=15,
        help="Maximum runtime in minutes after which the best solution found will be reported (default: 15 minutes)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Whether to print verbose output",
    )

    parser.add_argument(
        "-e",
        "--exclude",
        default=None,
        help="Comma-separated list of block names (from the dictionary) or "
        "masses to exclude from candidates. E.g. 'NeuGc' or '307.09'. "
        "Useful when you know certain blocks are absent from the sample.",
    )

    parser.add_argument(
        "--exhaustive",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Exhaustive model comparison level: "
        "0 = off (nested-only comparison), "
        "1 = test all combinations that include Hex (default), "
        "2 = test all combinations. "
        "(default: 1)",
    )
    parser.add_argument(
        "--glycan-type",
        choices=["native", "permethylated", "peracetylated"],
        default=None,
        help="Glycan derivatization type. Filters the blocks dictionary to "
        "only include blocks relevant for this type. "
        "(default: native)",
    )

    # Check if no arguments were provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # Determine filename-based defaults for mode and matrix
    filename_mode = None
    filename_matrix = None

    if args.peaks and args.peaks.lower().endswith(".msd"):
        filename_mode, filename_matrix = extract_mode_matrix_from_filename(args.peaks)
        if filename_mode or filename_matrix:
            print(
                f"Detected from filename: {', '.join(filter(None, [f'mode={filename_mode}' if filename_mode else None, f'matrix={filename_matrix}' if filename_matrix else None]))}"
            )

    # Apply detected defaults only if not specified on command line
    # If sys.argv doesn't contain the flag, it means user didn't specify it
    if filename_mode and "--mode" not in sys.argv and "-D" not in sys.argv:
        args.mode = filename_mode
        print(f"Using filename-derived mode: {args.mode}")

    if filename_matrix and "--matrix" not in sys.argv and "-X" not in sys.argv:
        args.matrix = filename_matrix
        print(f"Using filename-derived matrix: {args.matrix}")

    # Set default common block based on mode and matrix if not provided
    _common_composition = None
    if args.common is None:
        blocks_dict = getattr(args, "blocks_dict", str(DATA_DIR / "blocks.txt"))
        args.common, _common_composition = determine_common_block(
            cast(ModeType, args.mode),
            cast(MatrixType, args.matrix),
            blocks_dict_path=blocks_dict,
        )
    args._common_composition = _common_composition

    # Set default output directory if not specified
    if args.output is None:
        basename, ext = os.path.splitext(os.path.basename(args.peaks))
        ext = ext.lstrip('.')
        args.output = f"{basename}{f'.{ext}' if ext else ''}_glycansolver"

    return args


def cli():
    args = parse_arguments()
    args_dict = vars(args)

    exclude = args_dict.pop("exclude", None)
    exhaustive = args_dict.pop("exhaustive", 1)
    common_composition = args_dict.pop("_common_composition", None)
    glycan_type = args_dict.pop("glycan_type", None)

    try:
        solve_progressive(
            **args_dict,
            exclude=exclude,
            exhaustive=exhaustive,
            common_composition=common_composition,
            glycan_type=glycan_type,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
