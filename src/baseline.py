# src/baseline.py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from src import config


def clip_features(X):
    """(N, 64, 312) log-Mel -> (N, 128): mean and std over time for each Mel band."""
    return np.concatenate([X.mean(axis=2), X.std(axis=2)], axis=1)


def main():
    train = np.load(config.FEATURES_DIR / "train.npy")
    val = np.load(config.FEATURES_DIR / "val.npy")
    test = np.load(config.FEATURES_DIR / "test.npy")
    val_meta = pd.read_csv(config.FEATURES_DIR / "val_meta.csv", dtype={"snr_db": str})
    test_meta = pd.read_csv(config.FEATURES_DIR / "test_meta.csv", dtype={"snr_db": str})
    y = test_meta["is_anomaly"].values

    # trained on NORMAL training clips only, same as the autoencoder
    forest = IsolationForest(n_estimators=200, random_state=config.SEED, n_jobs=-1)
    forest.fit(clip_features(train))

    # higher score = more abnormal
    val_s = -forest.score_samples(clip_features(val))
    test_s = -forest.score_samples(clip_features(test))

    # same rules as the autoencoder: 95th percentile of NORMAL validation scores
    pct = config.THRESHOLD_PERCENTILE
    global_thr = np.percentile(val_s, pct)
    thr_id = {m: np.percentile(val_s[(val_meta["machine_id"] == m).values], pct)
              for m in val_meta["machine_id"].unique()}
    pred_global = (test_s > global_thr).astype(int)
    pred_id = (test_s > test_meta["machine_id"].map(thr_id).values).astype(int)

    print(f"--- Isolation Forest baseline (pooled) ---")
    print(f"ROC-AUC: {roc_auc_score(y, test_s):.4f}")
    for name, p in [("Global threshold", pred_global), ("Per-machine thresholds", pred_id)]:
        print(f"{name:24s} Prec {precision_score(y, p, zero_division=0):.4f} | "
              f"Recall {recall_score(y, p, zero_division=0):.4f} | "
              f"F1 {f1_score(y, p, zero_division=0):.4f}")

    # per machine id, side by side with the autoencoder
    rows = []
    for m, g in test_meta.groupby("machine_id"):
        idx, gy = g.index.values, y[g.index.values]
        rows.append({"machine_id": m,
                     "if_auc": roc_auc_score(gy, test_s[idx]),
                     "if_f1_global": f1_score(gy, pred_global[idx], zero_division=0),
                     "if_f1_id": f1_score(gy, pred_id[idx], zero_division=0)})
    ae = pd.read_csv(config.METRICS_DIR / "per_machine_metrics.csv")[
        ["machine_id", "roc_auc", "f1_global", "f1_id"]
    ].rename(columns={"roc_auc": "ae_auc", "f1_global": "ae_f1_global", "f1_id": "ae_f1_id"})
    table = pd.DataFrame(rows).merge(ae, on="machine_id")
    table.to_csv(config.METRICS_DIR / "baseline_vs_autoencoder.csv", index=False)
    print("\nPer machine id (if = Isolation Forest, ae = autoencoder):")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()