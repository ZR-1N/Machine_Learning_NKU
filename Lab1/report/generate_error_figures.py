"""Plot local k=1,3,5 LOO confusion counts and k=1 misclassified images."""

from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import runpy
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parents[1]
figures = root / "report" / "figures"
figures.mkdir(parents=True, exist_ok=True)
# The original module uses a relative data path; run this script from the lab root.
with redirect_stdout(StringIO()):
    source = runpy.run_path(str(root / "exp_1.py"))
X, y = source["X"], source["y"]
predict = source["knn_predict"]
predictions_by_k = {}
matrices = {}
for k in (1, 3, 5):
    predicted = np.empty(len(y), dtype=int)
    for i in range(len(y)):
        predicted[i] = predict(np.delete(X, i, axis=0), np.delete(y, i), X[i], k)
    counts = np.zeros((10, 10), dtype=int)
    np.add.at(counts, (y, predicted), 1)
    assert counts.sum() == len(y)
    assert np.array_equal(counts.sum(axis=1), np.bincount(y, minlength=10))
    assert counts.trace() == np.count_nonzero(predicted == y)
    predictions_by_k[k] = predicted
    matrices[k] = counts
    np.savetxt(root / "report" / f"local_predictions_k{k}.csv",
               np.column_stack((np.arange(1, len(y) + 1), y, predicted)),
               fmt="%d", delimiter=",", header="sample_row,true_digit,predicted_digit",
               comments="")
    np.savetxt(root / "report" / f"local_confusion_k{k}.csv", counts,
               fmt="%d", delimiter=",")

predictions = predictions_by_k[1]
matrix = matrices[1]
errors = np.flatnonzero(predictions != y)
assert matrix.sum() == len(y)
assert np.array_equal(matrix.sum(axis=1), np.bincount(y, minlength=10))
assert matrix.trace() + len(errors) == len(y)

pairs = sorted(
    [(int(matrix[a, b]), a, b) for a in range(10) for b in range(10)
     if a != b and matrix[a, b] > 0],
    key=lambda item: (-item[0], item[1], item[2]),
)
# Pick the first erroneous row from each of the six most frequent confusion pairs.
selected = [int(np.flatnonzero((y == a) & (predictions == b))[0])
            for _, a, b in pairs[:6]]
assert len(selected) == 6 and len(set(selected)) == 6
assert np.all(predictions[selected] != y[selected])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
for k, counts in matrices.items():
    fig, ax = plt.subplots(figsize=(8.4, 7.0), layout="constrained")
    im = ax.imshow(counts, cmap="Blues", vmin=0, vmax=160)
    for a in range(10):
        for b in range(10):
            ax.text(b, a, str(counts[a, b]), ha="center", va="center",
                    fontsize=10, color="white" if counts[a, b] > 80 else "#222222")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit",
           ylabel="True digit", title=f"Local kNN confusion matrix (k={k}, LOO)")
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(im, ax=ax, shrink=0.86, label="Number of samples")
    fig.savefig(figures / f"local_confusion_k{k}.png", dpi=240)
    plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.8), layout="constrained")
for ax, i in zip(axes.flat, selected):
    ax.imshow(X[i].reshape(16, 16), cmap="gray", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_title(f"Sample #{i + 1}\nTrue: {y[i]}   Predicted: {predictions[i]}",
                 fontsize=12, pad=8)
    ax.axis("off")
fig.savefig(figures / "local_errors_k1.png", dpi=240)
plt.close(fig)

summary = {
    "python": sys.version.split()[0], "numpy": np.__version__,
    "matplotlib": matplotlib.__version__, "k": 1, "evaluation": "LOO",
    "source_sha256": hashlib.sha256((root / "exp_1.py").read_bytes()).hexdigest(),
    "data_md5": hashlib.md5((root / "data" / "semeion.data").read_bytes()).hexdigest(),
    "correct": int(matrix.trace()), "errors": len(errors),
    "accuracy": float(matrix.trace() / len(y)),
    "recall": (matrix.diagonal() / matrix.sum(axis=1)).tolist(),
    "top_confusion_pairs": [{"true": a, "predicted": b, "count": n}
                            for n, a, b in pairs[:6]],
    "selected_samples": [{"sample_row": i + 1, "true": int(y[i]),
                          "predicted": int(predictions[i])} for i in selected],
    "all_k_results": [{"k": k, "correct": int(counts.trace()),
                       "errors": int(len(y) - counts.trace()),
                       "accuracy": float(counts.trace() / len(y)),
                       "recall": (counts.diagonal() / counts.sum(axis=1)).tolist()}
                      for k, counts in matrices.items()],
}
(root / "report" / "error_figure_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
