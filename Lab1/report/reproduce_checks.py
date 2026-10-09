"""Reproduce data checks and LOO sorting diagnostics without changing exp_1.py."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


root = Path(__file__).resolve().parents[1]
data_path = root / "data" / "semeion.data"
raw = np.loadtxt(data_path)
X = raw[:, :256]
y = raw[:, 256:].argmax(axis=1)
ks = (1, 3, 5)
correct = np.zeros(3, dtype=int)
stable_correct = np.zeros(3, dtype=int)
boundary_ties = np.zeros(3, dtype=int)
vote_ties = np.zeros(3, dtype=int)

for i in range(len(y)):
    training = np.delete(X, i, axis=0)
    labels = np.delete(y, i)
    distances = np.sqrt(((training - X[i]) ** 2).sum(axis=1))
    default_order = np.argsort(distances)
    stable_order = np.argsort(distances, kind="stable")
    for j, k in enumerate(ks):
        counts = np.bincount(labels[default_order[:k]], minlength=10)
        stable_counts = np.bincount(labels[stable_order[:k]], minlength=10)
        correct[j] += int(counts.argmax() == y[i])
        stable_correct[j] += int(stable_counts.argmax() == y[i])
        boundary_ties[j] += int(
            distances[default_order[k - 1]] == distances[default_order[k]]
        )
        vote_ties[j] += int(np.count_nonzero(counts == counts.max()) > 1)

joined = np.vstack([
    np.loadtxt(root / "data" / "semeion_train.txt"),
    np.loadtxt(root / "data" / "semeion_test.txt"),
])
rows, multiplicities = np.unique(raw, axis=0, return_counts=True)
joined_rows, joined_multiplicities = np.unique(joined, axis=0, return_counts=True)
results = {
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "data_md5": hashlib.md5(data_path.read_bytes()).hexdigest(),
    "raw_shape": list(raw.shape),
    "split_matches_full_multiset": bool(
        np.array_equal(rows, joined_rows)
        and np.array_equal(multiplicities, joined_multiplicities)
    ),
    "initial_zero_labels": int(np.flatnonzero(y != 0)[0]),
    "unique_first_six_images": len(np.unique(X[:6], axis=0)),
    "k": list(ks),
    "correct_default": correct.tolist(),
    "correct_stable": stable_correct.tolist(),
    "boundary_tie_samples": boundary_ties.tolist(),
    "vote_tie_samples": vote_ties.tolist(),
}
output = Path(__file__).with_name("reproduction_checks.json")
output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
print(json.dumps(results, indent=2))
