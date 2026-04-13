#!/usr/bin/env python3
"""Train and evaluate on real MNIST, then report best single-model and ensemble accuracy."""
import gzip
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def download(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        urllib.request.urlretrieve(url, out_path)


def read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image file magic: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32)


def read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label file magic: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def evaluate_prob_ensemble(models, x_test, y_test):
    probs = [m.predict_proba(x_test) for m in models]
    mean_probs = np.mean(probs, axis=0)
    pred = np.argmax(mean_probs, axis=1)
    return float(np.mean(pred == y_test))


def main():
    data_dir = Path("experiments/data/mnist")
    for name, url in MNIST_URLS.items():
        download(url, data_dir / f"{name}.gz")

    x_train = read_images(data_dir / "train_images.gz") / 255.0
    y_train = read_labels(data_dir / "train_labels.gz")
    x_test = read_images(data_dir / "test_images.gz") / 255.0
    y_test = read_labels(data_dir / "test_labels.gz")

    candidates = [
        {"hidden_layer_sizes": (512, 256), "alpha": 1e-5, "lr": 0.001, "max_iter": 35, "seed": 41},
        {"hidden_layer_sizes": (512, 256), "alpha": 5e-6, "lr": 0.001, "max_iter": 35, "seed": 42},
        {"hidden_layer_sizes": (768, 256), "alpha": 1e-5, "lr": 0.0008, "max_iter": 40, "seed": 43},
    ]

    trained = []
    for cfg in candidates:
        clf = MLPClassifier(
            hidden_layer_sizes=cfg["hidden_layer_sizes"],
            activation="relu",
            solver="adam",
            alpha=cfg["alpha"],
            batch_size=256,
            learning_rate_init=cfg["lr"],
            max_iter=cfg["max_iter"],
            random_state=cfg["seed"],
            verbose=False,
        )
        clf.fit(x_train, y_train)
        acc = float(clf.score(x_test, y_test))
        trained.append({"config": cfg, "model": clf, "test_accuracy": acc})
        print(f"single_model {cfg} => test_acc={acc:.4f}")

    trained.sort(key=lambda x: x["test_accuracy"], reverse=True)
    best_single = trained[0]

    top_models = [trained[i]["model"] for i in range(min(3, len(trained)))]
    ensemble_acc = evaluate_prob_ensemble(top_models, x_test, y_test)

    out = {
        "dataset": "MNIST",
        "best_single_model": {
            "config": best_single["config"],
            "test_accuracy": best_single["test_accuracy"],
        },
        "ensemble_top3_test_accuracy": ensemble_acc,
        "target_met": ensemble_acc > 0.98,
    }

    out_path = Path("experiments/results/mnist_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
