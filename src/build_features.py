# src/build_features.py
import numpy as np
import pandas as pd
from tqdm import tqdm
from src import config
from src.preprocessing import load_audio
from src.feature_extraction import wav_to_logmel


def extract_split(rows):
    """Load every clip in this split and turn it into a (64, 312) log-Mel array."""
    specs = []
    for path in tqdm(rows["filepath"], desc="clips"):
        y = load_audio(config.ROOT / path)
        spec = wav_to_logmel(y)[:, :config.N_FRAMES]   # crop 313 -> 312 frames
        specs.append(spec)
    return np.stack(specs).astype(np.float32)


def main():
    splits = pd.read_csv(config.SPLITS_CSV, dtype={"snr_db": str})
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    for name in ["train", "val", "test"]:
        rows = splits[splits["split"] == name].reset_index(drop=True)
        print(f"\n{name}: {len(rows)} clips")
        X = extract_split(rows)
        assert np.isfinite(X).all(), f"{name} contains NaN or inf values"

        np.save(config.FEATURES_DIR / f"{name}.npy", X)
        # same row order as X, so row i of the CSV describes X[i]
        rows.to_csv(config.FEATURES_DIR / f"{name}_meta.csv", index=False)
        print(f"saved {X.shape} | min {X.min():.1f}, max {X.max():.1f}, mean {X.mean():.1f}")


if __name__ == "__main__":
    main()