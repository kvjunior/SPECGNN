# SpecGNN datasets

All dataset statistics reported in the paper are produced by one command:

```bash
pip install -r requirements.txt
python scripts/01_prepare_data.py --raw /path/to/raw --out data/processed --reports reports
python -m pytest -q tests        # normalization, deduplication and split invariants
```

The run takes about 20 seconds and is deterministic (seed 42): two runs produce
byte-identical files in `data/processed/`.

## 1. Upstream sources

Source code is **not** redistributed; download the upstream releases and unzip
them under `--raw` with the directory names below.

| Directory under `--raw` | Dataset | Download | Cite |
|---|---|---|---|
| `dappscan/` (`contracts/`, `SWCsource/`, `Audit_and_Repository_link.xlsx`) | DAppSCAN-source | https://github.com/InPlusLab/DAppSCAN | Zheng et al., IEEE TSE 50(6), 2024 |
| `resource2/` (the unzipped `dataset_preprocessing_for_vulnerabilities`) | Four-vulnerability dataset ("Resource 2") | https://github.com/Messi-Q/Smart-Contract-Dataset (Google Drive link in README) | Qian et al., WWW 2023 |
| `smartbugs-curated-main/` | SmartBugs-curated | https://github.com/smartbugs/smartbugs-curated | Durieux et al., ICSE 2020 |
| `SolidiFI-benchmark-master/` | SolidiFI | https://github.com/DependableSystemsLab/SolidiFI-benchmark | Ghaleb & Pattabiraman, ISSTA 2020 |
| `sc-defi-security-main/dataset/` | DeFi attack victim contracts | https://github.com/StefanosChaliasos/sc-defi-security | Chaliasos et al., ICSE 2024 |
| `Web3Bugs-main/` | Web3Bugs (scope analysis only) | https://github.com/ZhangZhuoSJTU/Web3Bugs | Zhang et al., ICSE 2023 |

Not used, and why:
- **ESC raw contracts** (Messi-Q Resource 1): no published label file for the
  3-type task described in the original manuscript.
- **GNNSCVulDetector `train_data/`**: of 6,200 reentrancy samples only 140 are
  distinct and 99% of validation samples are identical to training samples
  (timestamp: 15 distinct of 5,013; integer overflow: train and valid files are
  identical).
- **EFEVD `data/`** (xawmx/MTLContractdetection): 1,155 precomputed feature
  vectors from 336 contracts, no source code; the 9,742-sample set is not released.

## 2. Roles

| Dataset | Role |
|---|---|
| DAppSCAN | main benchmark, all 17 SWC types; train / val / test + 5 folds |
| Resource 2 | four-type benchmark (SWC-101/107/112/116); train / val / test + 5 folds |
| SmartBugs-curated | held-out, hand-annotated test set (never used for training) |
| SolidiFI | robustness test (injected bugs) |
| DeFi attacks | real-world test (victims of 127 exploits) |
| Web3Bugs | measures what fraction of real audit findings the 17 SWC types cover |

## 3. What the pipeline does

1. **Loading and label mapping** (`specgnn/data/loaders.py`, mappings in
   `config.py`). Labels are multi-label over the 17 SWC types with three values:
   `1` vulnerable, `0` not vulnerable, `null` not annotated. `label_quality`
   records whether each value was annotated, assumed negative, or comes from an
   approximate category mapping (e.g. DASP "access control" → SWC-105/112/115/100).
2. **Filtering (DAppSCAN)**, applied identically to labeled and unlabeled files:
   test/mock/script/formal-verification directories and file names, files with
   fewer than 20 tokens, and interface-only files (unlabeled files only). Every
   excluded file and the reason are listed in `exclusions.json`.
3. **Exact deduplication** within each dataset on a hash of the comment- and
   whitespace-free token stream. When copies carry different labels, labels are
   merged by OR (an audit that flagged a file outranks one that did not mention it).
4. **Near-duplicate clustering** across all datasets: Jaccard similarity ≥ 0.90
   on token 5-gram shingles (MinHash LSH candidates, verified exactly).
5. **Cross-dataset deduplication**: a cluster present in several datasets is kept
   only in the highest-priority one (held-out test sets first), so no test
   contract has a near-copy in any training set.
6. **Cross-repository deduplication** (DAppSCAN, Resource 2): library files copied
   into many repositories (SafeMath, Address, ERC20, ...) are kept in one
   repository only; otherwise they would either leak across splits or chain
   unrelated repositories into one giant split group.
7. **Grouped, stratified splitting**: groups (same GitHub repository, same DApp
   project, same near-duplicate cluster) never cross train/val/test or folds;
   within that constraint, groups are placed to match 70/15/15 for every SWC
   type and for the overall size.
8. **Safe-sample cap** (DAppSCAN): unlabeled files receive a fixed random rank;
   a cap C keeps the C lowest ranks. Caps are nested and do not change any
   file's split, so the cap-sensitivity study compares like with like.

The report's leakage table verifies that no repository, cluster or hash spans
two splits.

## 4. Output format

`data/processed/<dataset>.jsonl`, one object per file:

```
uid, dataset, rel_path, norm_sha256, raw_sha256, n_lines, n_tokens, pragma,
group_key, cluster, split_group, split ("train"|"val"|"test"), fold (0-4 | null),
safe_rank (int | null), labels {SWC-xxx: 1|0|null}, label_quality {...}, meta {...}
```

`meta` keeps provenance: original categories, annotated line numbers and
functions (DAppSCAN, SmartBugs, SolidiFI), incident and chain (DeFi attacks).
`manifest.json` stores the full configuration and every filtering count.

## 5. Known limitations (reported in the paper)

- DAppSCAN files without an SWC annotation are treated as negatives. Audits do not
  certify the absence of every weakness, so some negatives may be unreported
  positives.
- SWC-112 (8 files) and SWC-115 (7 files) are too rare in DAppSCAN for stable
  per-type test estimates (SWC-112 has no validation positive, SWC-115 no test
  positive); per-type results for them are reported from 5-fold cross-validation.
- Resource 2 labels are contract-level and partial: a contract is annotated only
  for the vulnerability folders it appears in (`null` elsewhere).
- Only 22 of the 126 DeFi victim contracts and 61 of 492 Web3Bugs findings fall in
  the 17 SWC types; most real exploits are logic or economic bugs.
