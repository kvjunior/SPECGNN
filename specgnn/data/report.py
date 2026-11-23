"""Generate every dataset table/number used in the paper from the processed files."""
from __future__ import annotations

import csv
import json
import os
from collections import Counter

from . import config as C
from .splits import apply_safe_cap

DISPLAY = {
    "dappscan": "DAppSCAN",
    "resource2": "Qian et al. (Resource 2)",
    "smartbugs_curated": "SmartBugs-curated",
    "solidifi": "SolidiFI",
    "chaliasos": "DeFi attacks (Chaliasos et al.)",
}
ROLE = {
    "dappscan": "train/val/test",
    "resource2": "train/val/test",
    "smartbugs_curated": "held-out test",
    "solidifi": "robustness test",
    "chaliasos": "real-world test",
}
GRANULARITY = {
    "dappscan": "file, line",
    "resource2": "contract",
    "smartbugs_curated": "file, line",
    "solidifi": "file, line",
    "chaliasos": "contract",
}


def load(processed: str, name: str) -> list[dict]:
    path = os.path.join(processed, f"{name}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _positive(r: dict) -> bool:
    return any(v == 1 for v in r["labels"].values())


def dataset_row(name: str, recs: list[dict]) -> dict:
    types_covered = [t for t in C.SWC_TYPES if any(r["labels"][t] == 1 for r in recs)]
    splits = Counter(r["split"] for r in recs)
    pos = sum(_positive(r) for r in recs)
    return {
        "Dataset": DISPLAY[name],
        "Role": ROLE[name],
        "Files": len(recs),
        "Groups": len({r["split_group"] for r in recs}),
        "Vulnerable": pos,
        "Pos. rate": f"{pos / len(recs):.1%}" if recs else "-",
        "Multi-label files": sum(sum(v == 1 for v in r["labels"].values()) > 1 for r in recs),
        "SWC types": len(types_covered),
        "Labels": GRANULARITY[name],
        "Train/Val/Test": (f"{splits.get('train', 0)}/{splits.get('val', 0)}/{splits.get('test', 0)}"
                           if name in ("dappscan", "resource2") else f"-/-/{len(recs)}"),
    }


def per_swc(datasets: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for t in C.SWC_TYPES:
        row = {"SWC": t, "Name": C.SWC_NAMES[t]}
        for name, recs in datasets.items():
            row[DISPLAY[name]] = sum(r["labels"][t] == 1 for r in recs)
        if "dappscan" in datasets:
            d = datasets["dappscan"]
            row["DAppSCAN test"] = sum(r["labels"][t] == 1 and r["split"] == "test" for r in d)
        rows.append(row)
    return rows


def _md(rows: list[dict]) -> str:
    if not rows:
        return "_(empty)_\n"
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _latex_table3(rows: list[dict]) -> str:
    cols = ["Dataset", "Role", "Files", "Groups", "Vulnerable", "Pos. rate", "SWC types",
            "Labels", "Train/Val/Test"]
    esc = lambda s: str(s).replace("%", r"\%").replace("&", r"\&")
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Dataset statistics after filtering, cross-dataset deduplication and "
             r"grouped splitting. Groups = units that are never split across train/val/test "
             r"(repositories, base contracts, near-duplicate clusters).}",
             r"\label{tab:datasets}", r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{ll" + "r" * 5 + "ll}", r"\toprule",
             " & ".join(cols) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(r[c]) for c in cols) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def generate(processed: str, reports: str, web3bugs: dict | None = None,
             safe_cap: int | None = C.DEFAULT_SAFE_CAP) -> str:
    os.makedirs(reports, exist_ok=True)
    full = {n: load(processed, n) for n in DISPLAY}
    full = {n: v for n, v in full.items() if v}
    capped = {n: (apply_safe_cap(v, safe_cap) if n == "dappscan" else v) for n, v in full.items()}

    table3 = [dataset_row(n, v) for n, v in capped.items()]
    swc = per_swc(capped)
    _csv(table3, os.path.join(reports, "table3_datasets.csv"))
    _csv(swc, os.path.join(reports, "table_per_swc.csv"))
    with open(os.path.join(reports, "table3_datasets.tex"), "w") as fh:
        fh.write(_latex_table3(table3))

    # safe-cap sensitivity (Reviewer 3, comment 6)
    sweep = []
    d = full.get("dappscan", [])
    for cap in C.SAFE_CAP_SWEEP:
        sub = apply_safe_cap(d, cap)
        sp = Counter(r["split"] for r in sub)
        pos = sum(_positive(r) for r in sub)
        sweep.append({"Safe cap": "none" if cap is None else cap, "Files": len(sub),
                      "Vulnerable": pos, "Pos. rate": f"{pos / max(len(sub), 1):.1%}",
                      "Train/Val/Test": f"{sp['train']}/{sp['val']}/{sp['test']}"})
    if sweep:
        _csv(sweep, os.path.join(reports, "table_safe_cap_sweep.csv"))

    # split balance check
    balance = []
    for n in ("dappscan", "resource2"):
        recs = capped.get(n, [])
        for t in C.SWC_TYPES:
            tot = sum(r["labels"][t] == 1 for r in recs)
            if tot:
                c = Counter(r["split"] for r in recs if r["labels"][t] == 1)
                balance.append({"Dataset": DISPLAY[n], "SWC": t, "Total": tot,
                                "Train": c["train"], "Val": c["val"], "Test": c["test"]})
    if balance:
        _csv(balance, os.path.join(reports, "table_split_balance.csv"))

    # leakage verification: no group and no near-duplicate cluster spans splits
    leak_lines = []
    for n in ("dappscan", "resource2"):
        recs = full.get(n, [])
        for key in ("split_group", "cluster", "group_key", "norm_sha256"):
            seen: dict = {}
            for r in recs:
                seen.setdefault(r[key], set()).add(r["split"])
            bad = sum(len(s) > 1 for s in seen.values())
            leak_lines.append(f"| {DISPLAY[n]} | {key} | {len(seen)} | {bad} |")

    with open(os.path.join(processed, "manifest.json")) as fh:
        manifest = json.load(fh)

    md = ["# Dataset report (generated by scripts/01_prepare_data.py)\n",
          f"Safe-sample cap for DAppSCAN: **{safe_cap}** (sweep below).\n",
          "## Table III — datasets\n", _md(table3),
          "## Vulnerable files per SWC type\n", _md(swc)]
    if sweep:
        md += ["## DAppSCAN safe-sample cap sensitivity\n",
               "Caps are nested: every smaller cap is a subset of the larger one, and each "
               "kept file stays in the same split.\n", _md(sweep)]
    if balance:
        md += ["## Positives per split\n", _md(balance)]
    md += ["## Leakage check (must be 0 in the last column)\n",
           "| Dataset | Key | Distinct values | Values spanning >1 split |", "|---|---|---|---|",
           *leak_lines, "",
           "## Filtering and deduplication\n",
           "```json", json.dumps({k: manifest[k] for k in (
               "loaded_files", "exclusion_counts", "excluded_positives",
               "within_dataset_exact_duplicates", "clustering", "cross_repository_near_duplicates",
               "cross_dataset_dropped",
               "final_files")}, indent=1), "```\n"]
    if web3bugs:
        md += ["## Scope analysis on Web3Bugs (real audit findings)\n",
               f"Total bugs: {web3bugs['total_bugs']}. Bugs whose class maps onto one of "
               f"the 17 SpecGNN types: **{web3bugs['mapped_total']} "
               f"({web3bugs['mapped_total'] / web3bugs['total_bugs']:.1%})**.\n",
               _md([{"Family": k, "Bugs": v} for k, v in web3bugs["by_family"].items()]),
               _md([{"Web3Bugs class": k, "SWC": ", ".join(C.WEB3BUGS_MAP[k][0]),
                     "Mapping": C.WEB3BUGS_MAP[k][1], "Bugs": v}
                    for k, v in sorted(web3bugs["mapped_to_specgnn"].items())])]
        with open(os.path.join(reports, "web3bugs_scope.json"), "w") as fh:
            json.dump(web3bugs, fh, indent=1)
    text = "\n".join(md)
    with open(os.path.join(reports, "dataset_report.md"), "w") as fh:
        fh.write(text)
    return text
