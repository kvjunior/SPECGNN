#!/usr/bin/env python3
"""How closely do the specifications reproduce a dataset's labelling rule?

Qian et al. (WWW 2023) labelled their four-type dataset by pattern-filtering suspicious
functions and having experts verify them, with patterns resembling our specifications
(e.g. timestamp: TDInvocation AND (TDAssign OR TDContaminate)). If a specification
nearly reproduces the labelling rule, specification feedback can look effective on that
dataset merely by imitating the labeller. This data-only analysis (no model involved)
reports, per type and label, the share of contracts with at least one CONFIRMED node and
with all applicable nodes REFUTED.

    python scripts/08_label_rule_agreement.py --dataset resource2
"""
import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from specgnn.data.config import SWC_TYPES  # noqa: E402
from specgnn.feedback.verdicts import CONF, NA, REF  # noqa: E402
from specgnn.training import load_samples  # noqa: E402

TYPES = {"resource2": ("SWC-101", "SWC-107", "SWC-112", "SWC-116")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="resource2")
    ap.add_argument("--graphs", default="data/graphs")
    ap.add_argument("--verdicts", default="data/verdicts")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    samples = load_samples(args.graphs, args.dataset, safe_cap=None)
    with open(os.path.join(args.verdicts, f"{args.dataset}.pkl"), "rb") as fh:
        verdicts = pickle.load(fh)
    out = {}
    for t in TYPES.get(args.dataset, SWC_TYPES):
        j = SWC_TYPES.index(t)
        out[t] = {}
        for lab, name in ((1, "positive"), (0, "negative")):
            rows = [s for s in samples if s["y_mask"][j] == 1 and s["y"][j] == lab]
            conf = allref = 0
            for s in rows:
                v = verdicts[s["uid"]][:, j]
                v = v[v != NA]
                if (v == CONF).any():
                    conf += 1
                elif len(v) and (v == REF).all():
                    allref += 1
            n = len(rows)
            out[t][name] = {"n": n, "any_confirmed": conf / n if n else None,
                            "all_refuted": allref / n if n else None}
    path = os.path.join(args.out, f"label_rule_agreement_{args.dataset}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    for t, d in out.items():
        print(f"{t}: positives CONFIRMED {d['positive']['any_confirmed']:.1%} (n={d['positive']['n']}), "
              f"negatives CONFIRMED {d['negative']['any_confirmed']:.1%} (n={d['negative']['n']})")


if __name__ == "__main__":
    main()
