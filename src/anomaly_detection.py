# src/anomaly_detection.py
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from src import config
from src.feature_extraction import normalize


def load_norm_stats():
    with open(config.NORM_STATS_PATH) as f:
        s = json.load(f)
    return s["mean"], s["std"]


def reconstruction_errors(model, X_raw, mean, std, batch_size=64):
    """Per-clip MSE between the normalized spectrogram and its reconstruction."""
    X = normalize(X_raw, mean, std)[..., np.newaxis].astype(np.float32)
    recon = model.predict(X, batch_size=batch_size, verbose=0)
    return np.mean((X - recon) ** 2, axis=(1, 2, 3))


def select_threshold(val_errors, percentile=config.THRESHOLD_PERCENTILE):
    """Threshold from NORMAL validation clips only (never from test labels)."""
    return float(np.percentile(val_errors, percentile))


def main():
    model = load_model(config.MODEL_PATH)
    mean, std = load_norm_stats()

    val_raw = np.load(config.FEATURES_DIR / "val.npy")
    test_raw = np.load(config.FEATURES_DIR / "test.npy")
    val_meta = pd.read_csv(config.FEATURES_DIR / "val_meta.csv", dtype={"snr_db": str})
    test_meta = pd.read_csv(config.FEATURES_DIR / "test_meta.csv", dtype={"snr_db": str})

    val_err = reconstruction_errors(model, val_raw, mean, std)
    test_err = reconstruction_errors(model, test_raw, mean, std)

    # A: one global threshold (as before)
    global_thr = select_threshold(val_err)

    # B: one threshold + score scale per machine id, from its NORMAL validation clips
    per_id = {}
    for mid in sorted(val_meta["machine_id"].unique()):
        e = val_err[(val_meta["machine_id"] == mid).values]
        per_id[mid] = {"threshold": select_threshold(e),
                       "val_mean": float(e.mean()),
                       "val_std": float(e.std())}

    mid_col = test_meta["machine_id"]
    thr_id = mid_col.map(lambda m: per_id[m]["threshold"])
    mu = mid_col.map(lambda m: per_id[m]["val_mean"])
    sd = mid_col.map(lambda m: per_id[m]["val_std"])

    test_meta["score"] = test_err
    test_meta["pred"] = (test_err > global_thr).astype(int)       # global threshold
    test_meta["pred_id"] = (test_err > thr_id).astype(int)        # per-machine threshold
    test_meta["score_z"] = (test_err - mu) / sd                   # per-machine scaled score

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    test_meta.to_csv(config.SCORES_CSV, index=False)
    with open(config.THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": global_thr,
                   "percentile": config.THRESHOLD_PERCENTILE,
                   "per_machine": per_id}, f, indent=2)

    print(f"Global threshold: {global_thr:.5f}")
    for mid, v in per_id.items():
        print(f"{mid} threshold: {v['threshold']:.5f} (val mean error {v['val_mean']:.5f})")


if __name__ == "__main__":
    main()