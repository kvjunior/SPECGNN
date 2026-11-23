"""Dataset loaders.

Each loader returns (records, exclusions). A record is a plain dict:

    uid           unique id "<dataset>:<relative path>"
    dataset       dataset name
    rel_path      path relative to the dataset's raw root
    abs_path      absolute path on this machine (not written to the release files)
    tokens        normalized token list (dropped before writing)
    norm_sha256   hash of the normalized token stream
    raw_sha256    hash of the raw file
    n_lines, n_tokens, pragma
    group_key     dataset-specific grouping key (project / repository / base contract)
    labels        {SWC id: 1 | 0 | None}   None = unknown (not annotated)
    label_quality {SWC id: "annotated" | "assumed_negative" | "approx" | "unknown"}
    meta          dataset-specific provenance (original categories, lines, ...)

Exclusions are (rel_path, reason, was_positive) tuples, written to the audit report.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sqlite3
from collections import defaultdict

from . import config as C
from .normalize import (
    is_interface_only,
    normalized_hash,
    normalized_tokens,
    pragma_version,
    raw_hash,
    read_source,
)


def _base_record(dataset: str, root: str, abs_path: str) -> dict | None:
    src = read_source(abs_path)
    toks = normalized_tokens(src)
    return {
        # uid from the normalized path: identical whichever unzip tool produced the
        # files ('#Uxxxx' escapes vs. real Unicode names); rel_path is the on-disk path
        "uid": f"{dataset}:{_decode_zip_name(os.path.relpath(abs_path, root))}",
        "dataset": dataset,
        "rel_path": os.path.relpath(abs_path, root),
        "abs_path": abs_path,
        "tokens": toks,
        "norm_sha256": normalized_hash(toks),
        "raw_sha256": raw_hash(src),
        "n_lines": src.count("\n") + 1,
        "n_tokens": len(toks),
        "pragma": pragma_version(src),
    }


def _labels(positive: set[str], negative_policy: str,
            approx: set[str] = frozenset()) -> tuple[dict, dict]:
    """Build label + quality dicts over the 17 in-scope types.

    negative_policy: "annotated"         -> absent types are true negatives (0)
                     "assumed_negative"  -> absent types are 0 but flagged
                     "unknown"           -> absent types are None
    """
    labels, quality = {}, {}
    for t in C.SWC_TYPES:
        if t in positive:
            labels[t], quality[t] = 1, ("approx" if t in approx else "annotated")
        elif negative_policy == "unknown":
            labels[t], quality[t] = None, "unknown"
        else:
            labels[t], quality[t] = 0, negative_policy
    return labels, quality


def _path_excluded(rel_path: str) -> str | None:
    p = "/" + rel_path.replace("\\", "/").lower()
    for part in C.EXCLUDED_PATH_PARTS:
        if part in p:
            return f"path contains '{part.strip('/')}'"
    name = os.path.basename(p)
    for pat in C.EXCLUDED_NAME_PATTERNS:
        if pat in name:
            return f"file name contains '{pat}'"
    return None


# ---------------------------------------------------------------------------
# DAppSCAN (Zheng et al., IEEE TSE 2024)
# ---------------------------------------------------------------------------
def _repo_key(url: str | None) -> str | None:
    if not url:
        return None
    m = re.search(r"github\.com/([^/\s;]+)/([^/\s;#?]+)", url, re.I)
    if not m:
        return None
    return f"gh:{m.group(1)}/{m.group(2)}".lower().removesuffix(".git")


def _parse_lines(spec) -> list[list[int]]:
    """'L11' -> [[11, 11]]; 'L139-199' -> [[139, 199]]; 'L3,L7-9' -> [[3, 3], [7, 9]]."""
    out = []
    for a, b in re.findall(r"L?(\d+)(?:\s*-\s*L?(\d+))?", str(spec or "")):
        out.append([int(a), int(b or a)])
    return out


def _decode_zip_name(path: str) -> str:
    """Some unzip tools write non-ASCII characters as '#Uxxxx'; undo that so
    contract paths match the paths recorded in SWCsource/*.json."""
    return re.sub(r"#U([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), path)


def load_dappscan(root: str) -> tuple[list[dict], list[tuple]]:
    """root contains contracts/, SWCsource/ and Audit_and_Repository_link.xlsx."""
    import openpyxl

    # repository grouping: audits of the same GitHub repository share a group
    wb = openpyxl.load_workbook(os.path.join(root, "Audit_and_Repository_link.xlsx"),
                                read_only=True)
    dir_info: dict[str, dict] = {}
    for row in list(wb.active.iter_rows(values_only=True))[1:]:
        fname, company, project, _report, repo = (list(row) + [None] * 5)[:5]
        if fname:
            dir_info[_decode_zip_name(str(fname).strip())] = {
                "company": company, "project": (project or "").strip(),
                "repo": _repo_key(str(repo) if repo else None)}

    # labels: SWCsource/<dir>/.../File.json  <->  contracts/<dir>/.../File.sol
    file_labels: dict[str, list[dict]] = {}
    for jf in glob.glob(os.path.join(root, "SWCsource", "**", "*.json"), recursive=True):
        with open(jf, encoding="utf-8") as fh:
            entry = json.load(fh)
        rel = _decode_zip_name(entry["filePath"].split("contracts/", 1)[1])
        file_labels[rel] = entry["SWCs"]

    records, exclusions = [], []
    contracts_root = os.path.join(root, "contracts")
    seen_labeled = set()
    for path in glob.glob(os.path.join(contracts_root, "**", "*.sol"), recursive=True):
        rel = _decode_zip_name(os.path.relpath(path, contracts_root))
        swcs = file_labels.get(rel)
        is_labeled = swcs is not None
        if is_labeled:
            seen_labeled.add(rel)
        all_types = sorted({s["category"][:7] for s in (swcs or [])})
        in_scope = {t for t in all_types if t in C.SWC_TYPES}
        positive = bool(in_scope)

        reason = _path_excluded(rel)
        if reason:
            exclusions.append((rel, reason, positive))
            continue
        rec = _base_record("dappscan", contracts_root, path)
        if rec["n_tokens"] < C.MIN_NORMALIZED_TOKENS:
            exclusions.append((rel, f"fewer than {C.MIN_NORMALIZED_TOKENS} tokens", positive))
            continue
        if not is_labeled and is_interface_only(rec["tokens"]):
            exclusions.append((rel, "interface-only file", False))
            continue

        top = rel.split(os.sep, 1)[0]
        info = dir_info.get(top, {})
        project = info.get("project") or top.split("-", 1)[-1]
        rec["group_key"] = info.get("repo") or "proj:" + re.sub(r"[^a-z0-9]", "", project.lower())
        rec["labels"], rec["label_quality"] = _labels(in_scope, "annotated")
        rec["meta"] = {
            "dapp_dir": top,
            "audit_company": info.get("company"),
            "has_swc_annotation": is_labeled,
            "all_swc": all_types,
            "out_of_scope_swc": [t for t in all_types if t not in C.SWC_TYPES],
            "swc_annotations": [
                {"swc": s["category"][:7], "lines": _parse_lines(s.get("lineNumber")),
                 "function": None if s.get("function") in (None, "N/A") else s.get("function")}
                for s in (swcs or [])],
        }
        records.append(rec)

    for rel in sorted(set(file_labels) - seen_labeled):
        exclusions.append((rel, "label file has no matching source file", True))
    return records, exclusions


# ---------------------------------------------------------------------------
# Qian et al. four-vulnerability dataset (Messi-Q Smart-Contract-Dataset,
# Resource 2; WWW 2023). Contract-level labels; a contract is annotated only
# for the vulnerability folders it appears in.
# ---------------------------------------------------------------------------
def load_resource2(root: str) -> tuple[list[dict], list[tuple]]:
    by_key: dict[tuple[str, str], dict] = {}
    exclusions = []
    for folder, swc in C.RESOURCE2_MAP.items():
        lv = folder.lower()
        names_f = os.path.join(root, folder, f"final_{lv}_name_fixed.txt")
        labels_f = os.path.join(root, folder, f"final_{lv}_label_fixed.txt")
        if not os.path.exists(names_f):                 # delegatecall has no _fixed
            names_f = os.path.join(root, folder, f"final_{lv}_name.txt")
            labels_f = os.path.join(root, folder, f"final_{lv}_label.txt")
        names = open(names_f).read().split()
        labs = open(labels_f).read().split()
        if len(names) != len(labs):
            raise ValueError(f"{folder}: {len(names)} names vs {len(labs)} labels")
        for name, lab in zip(names, labs, strict=True):
            path = os.path.join(root, folder, "sourcecode", name)
            if not os.path.exists(path):
                exclusions.append((f"{folder}/{name}", "source file missing", lab == "1"))
                continue
            rec = _base_record("resource2", root, path)
            key = (name, rec["norm_sha256"])
            if key not in by_key:
                rec["uid"] = f"resource2:{name}:{rec['norm_sha256'][:8]}"
                rec["group_key"] = "esc:" + name
                rec["labels"] = {t: None for t in C.SWC_TYPES}
                rec["label_quality"] = {t: "unknown" for t in C.SWC_TYPES}
                rec["meta"] = {"folders": [], "contract_name": name}
                by_key[key] = rec
            r = by_key[key]
            r["labels"][swc] = int(lab)
            r["label_quality"][swc] = "annotated"
            r["meta"]["folders"].append(folder)
    records = []
    for r in by_key.values():
        if r["n_tokens"] < C.MIN_NORMALIZED_TOKENS:
            exclusions.append((r["rel_path"], f"fewer than {C.MIN_NORMALIZED_TOKENS} tokens",
                               any(v == 1 for v in r["labels"].values())))
        else:
            records.append(r)
    return records, exclusions


# ---------------------------------------------------------------------------
# SmartBugs-curated (Durieux et al., ICSE 2020) -- held-out test set
# ---------------------------------------------------------------------------
def load_smartbugs_curated(root: str) -> tuple[list[dict], list[tuple]]:
    with open(os.path.join(root, "vulnerabilities.json"), encoding="utf-8") as fh:
        entries = json.load(fh)
    records, exclusions = [], []
    for e in entries:
        path = os.path.join(root, e["path"])
        cats = sorted({v["category"] for v in e["vulnerabilities"]})
        pos, approx = set(), set()
        for c in cats:
            swcs, kind = C.SMARTBUGS_MAP[c]
            pos |= set(swcs)
            if kind == "approx":
                approx |= set(swcs)
        if not pos:
            exclusions.append((e["path"], "only out-of-scope categories: " + ",".join(cats), False))
            continue
        rec = _base_record("smartbugs_curated", root, path)
        rec["group_key"] = "sb:" + e["path"]
        rec["labels"], rec["label_quality"] = _labels(pos, "assumed_negative", approx)
        rec["meta"] = {"dasp_categories": cats,
                       "swc_annotations": [
                           {"swc": swc, "lines": [[ln, ln] for ln in v["lines"]], "function": None}
                           for v in e["vulnerabilities"] for swc in C.SMARTBUGS_MAP[v["category"]][0]],
                       "upstream_source": e.get("source")}
        records.append(rec)
    return records, exclusions


# ---------------------------------------------------------------------------
# SolidiFI (Ghaleb & Pattabiraman, ISSTA 2020) -- robustness test set
# ---------------------------------------------------------------------------
def load_solidifi(root: str) -> tuple[list[dict], list[tuple]]:
    records, exclusions = [], []
    bug_root = os.path.join(root, "buggy_contracts")
    for folder, (swcs, _kind) in C.SOLIDIFI_MAP.items():
        for path in sorted(glob.glob(os.path.join(bug_root, folder, "buggy_*.sol"))):
            idx = re.search(r"buggy_(\d+)\.sol$", path).group(1)
            log = os.path.join(bug_root, folder, f"BugLog_{idx}.csv")
            lines = []
            if os.path.exists(log):
                with open(log, encoding="utf-8", errors="replace") as fh:
                    for row in csv.DictReader(fh):
                        loc, length = row.get("loc", ""), row.get("length", "1")
                        if loc.isdigit():
                            n = int(length) if length.isdigit() else 1
                            lines.append([int(loc), int(loc) + max(n, 1) - 1])
            rec = _base_record("solidifi", root, path)
            # the same 50 base contracts are reused for all 7 bug types
            rec["group_key"] = "solidifi-base:" + idx
            rec["labels"], rec["label_quality"] = _labels(set(swcs), "assumed_negative")
            rec["meta"] = {"bug_type": folder, "base_contract": int(idx), "n_injected": len(lines),
                           "swc_annotations": [{"swc": s, "lines": lines, "function": None}
                                               for s in swcs]}
            records.append(rec)
    return records, exclusions


# ---------------------------------------------------------------------------
# Chaliasos et al. (ICSE 2024) real-world DeFi victim contracts -- real-world test
# ---------------------------------------------------------------------------
def load_chaliasos(root: str) -> tuple[list[dict], list[tuple]]:
    """root = sc-defi-security-main/dataset"""
    db = sqlite3.connect(os.path.join(root, "database", "retro.db"))
    q = """SELECT lower(vc.contract_address), lower(vc.chain), vc.contract_name,
                  v.incident_id, i.incident_title, vm.vuln_title
           FROM VulnerableContract vc
           JOIN Victim v ON vc.victim_id = v.victim_id
           JOIN Incident i ON i.incident_id = v.incident_id
           LEFT JOIN IncidentCause ic ON ic.incident_id = v.incident_id
           LEFT JOIN Cause ca ON ca.cause_id = ic.cause_id
           LEFT JOIN VulnerabilitiesMap vm ON vm.vuln_map_id = ca.vuln_map_id
           WHERE vc.has_source = 1"""
    info: dict[tuple[str, str], dict] = defaultdict(lambda: {"vulns": set(), "incidents": set()})
    for addr, chain, name, inc_id, inc_title, vuln in db.execute(q):
        d = info[(chain, addr)]
        d["name"] = name
        d["incidents"].add((inc_id, inc_title))
        if vuln:
            d["vulns"].add(vuln)

    records, exclusions = [], []
    src_root = os.path.join(root, "data", "contracts", "sources")
    for path in sorted(glob.glob(os.path.join(src_root, "*", "sol", "*.sol"))):
        chain = path.split(os.sep)[-3]
        addr = os.path.basename(path)[:-4].lower()
        d = info.get((chain, addr))
        rel = os.path.relpath(path, root)
        if d is None:
            exclusions.append((rel, "no incident record in retro.db", False))
            continue
        pos, approx = set(), set()
        for v in d["vulns"]:
            swcs, kind = C.CHALIASOS_MAP.get(v, ([], "out_of_scope"))
            pos |= set(swcs)
            if kind == "approx":
                approx |= set(swcs)
        rec = _base_record("chaliasos", root, path)
        incidents = sorted(d["incidents"])
        rec["group_key"] = "incident:" + str(incidents[0][0])
        rec["labels"], rec["label_quality"] = _labels(pos, "assumed_negative", approx)
        rec["meta"] = {"chain": chain, "address": addr, "contract_name": d["name"],
                       "incidents": [t for _, t in incidents],
                       "vulnerability_titles": sorted(d["vulns"]),
                       "in_scope": bool(pos)}
        records.append(rec)
    return records, exclusions


# ---------------------------------------------------------------------------
# Web3Bugs (Zhang et al., ICSE 2023) -- scope analysis only (no samples)
# ---------------------------------------------------------------------------
def web3bugs_scope(root: str) -> dict:
    with open(os.path.join(root, "results", "bugs.csv"), encoding="utf-8") as fh:
        rows = list(csv.reader(fh, skipinitialspace=True))[1:]
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        if len(r) > 2:
            counts[r[2].strip()] += 1
    total = sum(counts.values())
    mapped = {c: n for c, n in counts.items() if c in C.WEB3BUGS_MAP}
    per_swc: dict[str, int] = defaultdict(int)
    for c, n in mapped.items():
        for s in C.WEB3BUGS_MAP[c][0]:
            per_swc[s] += n

    def family(c: str) -> str:
        if c.startswith("O"):
            return "O (excluded by Web3Bugs authors)"
        if c.startswith("L"):
            return "L (machine-auditable)"
        if c.startswith("S") or c == "SC":
            return "S/SE/SC (functional / logic)"
        return "other"

    fam: dict[str, int] = defaultdict(int)
    for c, n in counts.items():
        fam[family(c)] += n
    return {"total_bugs": total, "by_class": dict(sorted(counts.items())),
            "by_family": dict(fam), "mapped_to_specgnn": dict(mapped),
            "mapped_total": sum(mapped.values()), "per_swc": dict(per_swc)}
