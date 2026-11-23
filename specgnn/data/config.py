"""Central configuration for dataset preparation.

Every number that appears in the paper's dataset section (Table III) is derived
from the files by `scripts/01_prepare_data.py` using only the settings below, so the
table can always be regenerated from scratch.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Vulnerability scope
# ---------------------------------------------------------------------------
# The 17 SWC types covered by the specification library (Table I of the paper).
SWC_TYPES: list[str] = [
    "SWC-100", "SWC-101", "SWC-102", "SWC-103", "SWC-104", "SWC-105",
    "SWC-107", "SWC-108", "SWC-112", "SWC-113", "SWC-114", "SWC-115",
    "SWC-116", "SWC-120", "SWC-128", "SWC-131", "SWC-135",
]

SWC_NAMES: dict[str, str] = {
    "SWC-100": "Function Default Visibility",
    "SWC-101": "Integer Overflow and Underflow",
    "SWC-102": "Outdated Compiler Version",
    "SWC-103": "Floating Pragma",
    "SWC-104": "Unchecked Call Return Value",
    "SWC-105": "Unprotected Ether Withdrawal",
    "SWC-107": "Reentrancy",
    "SWC-108": "State Variable Default Visibility",
    "SWC-112": "Delegatecall to Untrusted Callee",
    "SWC-113": "DoS with Failed Call",
    "SWC-114": "Transaction Order Dependence",
    "SWC-115": "Authorization through tx.origin",
    "SWC-116": "Block values as a proxy for time",
    "SWC-120": "Weak Sources of Randomness from Chain Attributes",
    "SWC-128": "DoS With Block Gas Limit",
    "SWC-131": "Presence of unused variables",
    "SWC-135": "Code With No Effects",
}

# ---------------------------------------------------------------------------
# Label mappings from each dataset's own taxonomy to SWC.
# "exact": the source category corresponds to one SWC type.
# "approx": the source category is broader; mapped to the closest SWC set and
#           reported separately in evaluation.
# ---------------------------------------------------------------------------
RESOURCE2_MAP: dict[str, str] = {
    "reentrancy": "SWC-107",
    "timestamp": "SWC-116",
    "Integeroverflow": "SWC-101",
    "delegatecall": "SWC-112",
}

# SmartBugs-curated uses DASP-10 categories.
SMARTBUGS_MAP: dict[str, tuple[list[str], str]] = {
    "reentrancy": (["SWC-107"], "exact"),
    "arithmetic": (["SWC-101"], "exact"),
    "unchecked_low_level_calls": (["SWC-104"], "exact"),
    "time_manipulation": (["SWC-116"], "exact"),
    "bad_randomness": (["SWC-120"], "exact"),
    "front_running": (["SWC-114"], "exact"),
    "denial_of_service": (["SWC-113", "SWC-128"], "approx"),
    "access_control": (["SWC-105", "SWC-112", "SWC-115", "SWC-100"], "approx"),
    "short_addresses": ([], "out_of_scope"),
    "other": ([], "out_of_scope"),
}

# SolidiFI injected bug types.
SOLIDIFI_MAP: dict[str, tuple[list[str], str]] = {
    "Re-entrancy": (["SWC-107"], "exact"),
    "Overflow-Underflow": (["SWC-101"], "exact"),
    "Timestamp-Dependency": (["SWC-116"], "exact"),
    "TOD": (["SWC-114"], "exact"),
    "tx.origin": (["SWC-115"], "exact"),
    "Unchecked-Send": (["SWC-105"], "exact"),
    "Unhandled-Exceptions": (["SWC-104"], "exact"),
}

# Chaliasos et al. (ICSE'24) vulnerability titles (VulnerabilitiesMap table).
CHALIASOS_MAP: dict[str, tuple[list[str], str]] = {
    "Reentrancy": (["SWC-107"], "exact"),
    "Integer Overflow and Underflow": (["SWC-101"], "exact"),
    "Delegatecall to Untrusted Callee": (["SWC-112"], "exact"),
    "Transaction Order Dependence": (["SWC-114"], "exact"),
    "Timestamp Dependence": (["SWC-116"], "exact"),
    "Weak Randomness": (["SWC-120"], "exact"),
    "Floating Pragma": (["SWC-103"], "exact"),
    "Outdated compiler or solidity version": (["SWC-102"], "exact"),
    "Unhandled or mishandled exception": (["SWC-104"], "approx"),
    "Unbounded or gas costly operation": (["SWC-128"], "approx"),
    "Function/State Visibility Error": (["SWC-100", "SWC-108"], "approx"),
}

# Web3Bugs (Zhang et al., ICSE'23) machine-auditable classes.
WEB3BUGS_MAP: dict[str, tuple[list[str], str]] = {
    "L1": (["SWC-107"], "exact"),          # reentrancy
    "L7": (["SWC-101"], "exact"),          # integer overflow/underflow
    "L8": (["SWC-104"], "approx"),         # revert issues from low-level calls
    "L4": (["SWC-128"], "approx"),         # exceeding gas limit
    "LB": (["SWC-115"], "exact"),          # tx.origin
    "L6": (["SWC-112"], "approx"),         # arbitrary external call
}

# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------
# Path fragments that mark non-production code. Applied to every DAppSCAN file
# (labeled or not) so positives and negatives are filtered identically.
EXCLUDED_PATH_PARTS: tuple[str, ...] = (
    "/test/", "/tests/", "/mock/", "/mocks/", "/node_modules/", "/lib/forge-std/",
    "/script/", "/scripts/", "/echidna/", "/certora/", "/fuzzing/",
)
EXCLUDED_NAME_PATTERNS: tuple[str, ...] = ("mock", "test", ".t.sol")
MIN_NORMALIZED_TOKENS = 20  # drop near-empty files (e.g. bare pragma/import)

# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
SHINGLE_SIZE = 5            # token k-grams
MINHASH_PERMUTATIONS = 128
LSH_BANDS = 32              # 32 bands x 4 rows
NEAR_DUP_JACCARD = 0.90     # files above this similarity share a cluster
# Second, independent criterion (Allamanis, Onward! 2019; SourcererCC defaults): files
# are near-duplicates if the Jaccard similarity of their identifier/literal token SETS
# is >= 0.8 AND of their token MULTISETS is >= 0.7; files with fewer than 20 such
# tokens are not compared. Order-insensitive, so it catches edited/reformatted copies
# that break 5-gram shingles. A pair is a near-duplicate if either criterion fires.
IDENT_SET_JACCARD = 0.8
IDENT_MULTISET_JACCARD = 0.7
IDENT_MIN_TOKENS = 20

# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
N_FOLDS = 5
DEFAULT_SAFE_CAP = 2000
SAFE_CAP_SWEEP = (500, 1000, 2000, 4000, None)  # None = no cap

# Datasets used only for evaluation; never seen in training.
HELD_OUT_DATASETS = ("smartbugs_curated", "solidifi", "chaliasos")
# Priority when a near-duplicate cluster spans datasets: the copy in the
# higher-priority dataset is kept, copies elsewhere are dropped.
DEDUP_PRIORITY = ("smartbugs_curated", "chaliasos", "solidifi", "dappscan", "resource2")

# Location of each dataset's source files below the --raw directory
# (directory names produced by unzipping the upstream releases).
RAW_SUBDIRS = {
    "dappscan": "dappscan/contracts",
    "resource2": "resource2",
    "smartbugs_curated": "smartbugs-curated-main",
    "solidifi": "SolidiFI-benchmark-master",
    "chaliasos": "sc-defi-security-main/dataset",
}
