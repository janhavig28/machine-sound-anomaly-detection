# src/evaluation.py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, precision_score,
                             recall_score, f1_score, confusion_matrix)
from tensorflow.keras.models import load_model
from src import config
from src.anomaly_detection import load_norm_stats
from src.feature_extraction import normalize


def compute_metrics(y_true, scores, preds):
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),                 # threshold-independent
        "precision": float(precision_score(y_true, preds, zero_division=0)),  # threshold-dependent
        "recall": float(recall_score(y_true, preds, zero_division=0)),        # threshold-dependent
        "f1": float(f1_score(y_true, preds, zero_division=0)),                # threshold-dependent
    }


def plot_roc(y_true, scores, auc, path):
    fpr, tpr, _ = roc_curve(y_true, scores)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"Autoencoder (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_error_distribution(df, threshold, path):
    plt.figure(figsize=(7, 4))
    bins = np.linspace(df["score"].min(), df["score"].quantile(0.995), 60)
    plt.hist(df.loc[df["is_anomaly"] == 0, "score"], bins=bins, alpha=0.6, label="Normal")
    plt.hist(df.loc[df["is_anomaly"] == 1, "score"], bins=bins, alpha=0.6, label="Abnormal")
    plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
    plt.xlabel("Reconstruction error (MSE)")
    plt.ylabel("Number of clips")
    plt.title("Reconstruction error distribution (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_reconstructions(df, path):
    """Original vs reconstructed spectrogram for one normal and one abnormal clip."""
    model = load_model(config.MODEL_PATH)
    mean, std = load_norm_stats()
    test_raw = np.load(config.FEATURES_DIR / "test.npy")

    rng = np.random.default_rng(config.SEED)
    i_norm = int(rng.choice(np.where(df["is_anomaly"] == 0)[0]))
    i_abn = int(rng.choice(np.where(df["is_anomaly"] == 1)[0]))

    X = normalize(test_raw[[i_norm, i_abn]], mean, std)[..., np.newaxis].astype(np.float32)
    recon = model.predict(X, verbose=0)

    vmin, vmax = X.min(), X.max()                       # same color scale for all panels
    rows = [("Normal", i_norm, 0), ("Abnormal", i_abn, 1)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for r, (name, idx, k) in enumerate(rows):
        err = df.loc[idx, "score"]
        axes[r, 0].imshow(X[k, :, :, 0], origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        axes[r, 0].set_title(f"{name} - original")
        axes[r, 1].imshow(recon[k, :, :, 0], origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        axes[r, 1].set_title(f"{name} - reconstructed (error {err:.4f})")
    for ax in axes.ravel():
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Mel bins")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

def confusion(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def main():
    df = pd.read_csv(config.SCORES_CSV, dtype={"snr_db": str})
    with open(config.THRESHOLD_PATH) as f:
        threshold = json.load(f)["threshold"]
    y = df["is_anomaly"].values

    # A: one global threshold on raw scores
    res_a = {**compute_metrics(y, df["score"].values, df["pred"].values),
             **confusion(y, df["pred"].values)}
    # B: per-machine thresholds, with scores scaled per machine (validation data only)
    res_b = {**compute_metrics(y, df["score_z"].values, df["pred_id"].values),
             **confusion(y, df["pred_id"].values)}

    rows = []
    for mid, g in df.groupby("machine_id"):
        gy = g["is_anomaly"].values
        m_a = compute_metrics(gy, g["score"].values, g["pred"].values)
        m_b = compute_metrics(gy, g["score"].values, g["pred_id"].values)
        rows.append({"machine_id": mid, "roc_auc": m_a["roc_auc"],
                     "prec_global": m_a["precision"], "prec_id": m_b["precision"],
                     "rec_global": m_a["recall"], "rec_id": m_b["recall"],
                     "f1_global": m_a["f1"], "f1_id": m_b["f1"]})
    per_id = pd.DataFrame(rows)

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_DIR / "metrics.json", "w") as f:
        json.dump({"global_threshold": res_a, "per_machine_threshold": res_b}, f, indent=2)
    per_id.to_csv(config.METRICS_DIR / "per_machine_metrics.csv", index=False)

    # plots stay on the global-threshold version
    plot_roc(y, df["score"].values, res_a["roc_auc"], config.PLOTS_DIR / "roc_curve.png")
    plot_error_distribution(df, threshold, config.PLOTS_DIR / "error_distribution.png")
    plot_reconstructions(df, config.PLOTS_DIR / "reconstructions.png")

    print("--- Pooled over all machines ---")
    print(f"{'':26s}{'ROC-AUC':>9s}{'Prec':>8s}{'Recall':>8s}{'F1':>8s}")
    for name, r in [("One global threshold", res_a), ("Per-machine thresholds", res_b)]:
        print(f"{name:26s}{r['roc_auc']:9.4f}{r['precision']:8.4f}{r['recall']:8.4f}{r['f1']:8.4f}")
        print(f"{'':26s}TN={r['tn']} FP={r['fp']} FN={r['fn']} TP={r['tp']}")
    print("\nPer machine id:")
    print(per_id.round(4).to_string(index=False))

if __name__ == "__main__":
    main()