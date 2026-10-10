"""Summary table of outputs/hmm_results_ar*.csv: Baum-Welch runs averaged over seeds

Usage: python scripts/summarize_hmm.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

rows = {}
for path in sorted(Path("outputs").glob("hmm_results*.csv")):
    if "quick" in path.name or path.name == "hmm_results.csv":
        continue  # quick runs stop after 20 iterations, older unsplit files are ignored too
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["model"]] = r  # baselines appear in every file with the same values

groups = defaultdict(list)
for name, r in rows.items():
    key = name.split(" seed ")[0]
    groups[key].append(r)


def fmt(values, scale=1.0, digits=3):
    v = np.array([float(x) * scale for x in values if x not in ("", None)])
    if len(v) == 0:
        return ""
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{v.mean():.{digits}f} +- {v.std(ddof=1):.{digits}f}"


print("| model | runs | test bits per note | chord accuracy % |")
print("|---|---|---|---|")
for key, rs in groups.items():
    print(f"| {key} | {len(rs)} | {fmt([r['test_bits'] for r in rs])} | {fmt([r['chord_acc'] for r in rs], 100, 1)} |")
