# src/train.py
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src import config
from src.feature_extraction import compute_stats, normalize
from src.model import build_autoencoder


def prepare(X, mean, std):
    """Normalize and add the channel axis: (N, 64, 312) -> (N, 64, 312, 1)."""
    return normalize(X, mean, std)[..., np.newaxis].astype(np.float32)


def plot_history(history, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(history["loss"], label="train")
    plt.plot(history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def main():
    tf.keras.utils.set_random_seed(config.SEED)
    for d in [config.MODELS_DIR, config.METRICS_DIR, config.PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    train_raw = np.load(config.FEATURES_DIR / "train.npy")
    val_raw = np.load(config.FEATURES_DIR / "val.npy")

    # normalization numbers come from TRAINING data only
    mean, std = compute_stats(train_raw)
    with open(config.NORM_STATS_PATH, "w") as f:
        json.dump({"mean": mean, "std": std}, f)
    print(f"train mean = {mean:.2f}, std = {std:.2f}")

    X_train = prepare(train_raw, mean, std)
    X_val = prepare(val_raw, mean, std)

    model = build_autoencoder()
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=config.PATIENCE,
                      restore_best_weights=True),
        ModelCheckpoint(config.MODEL_PATH, monitor="val_loss", save_best_only=True),
    ]

    # autoencoder: the input is also the target
    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
    )

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(config.METRICS_DIR / "history.json", "w") as f:
        json.dump(hist, f)
    plot_history(hist, config.PLOTS_DIR / "loss_curves.png")
    print("Best val loss:", min(hist["val_loss"]))
    print("Model saved to", config.MODEL_PATH)


if __name__ == "__main__":
    main()