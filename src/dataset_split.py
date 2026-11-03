# src/dataset_split.py
import pandas as pd
from src import config

VAL_FRACTION = 0.10   # share of the leftover normal clips used for validation


def make_splits(df, seed=config.SEED):
    """Add a 'split' column (train / val / test), separately for each machine id."""
    df = df.copy().reset_index(drop=True)
    df["split"] = ""

    for (snr, mid), g in df.groupby(["snr_db", "machine_id"]):
        normal = g[g["label"] == "normal"].sample(frac=1.0, random_state=seed)  # shuffled
        abnormal = g[g["label"] == "abnormal"]

        # test set: all abnormal clips + the same number of unseen normal clips
        n_test_normal = min(len(abnormal), len(normal) // 2)
        test_normal = normal.iloc[:n_test_normal]

        # remaining normal clips -> train and validation
        rest = normal.iloc[n_test_normal:]
        n_val = int(len(rest) * VAL_FRACTION)
        val, train = rest.iloc[:n_val], rest.iloc[n_val:]

        df.loc[train.index, "split"] = "train"
        df.loc[val.index, "split"] = "val"
        df.loc[test_normal.index, "split"] = "test"
        df.loc[abnormal.index, "split"] = "test"

    return df


def check_splits(df):
    """Safety checks: fail loudly if anything looks wrong."""
    assert (df["split"] != "").all(), "Some files got no split"
    assert df["filepath"].is_unique, "A file appears more than once"
    train_val_labels = set(df.loc[df["split"].isin(["train", "val"]), "label"])
    assert train_val_labels == {"normal"}, "Train/val must contain normal clips only"


if __name__ == "__main__":
    meta = pd.read_csv(config.METADATA_CSV, dtype={"snr_db": str})
    meta = meta[meta["machine"] == config.MACHINE_TYPE]
    out = make_splits(meta)
    check_splits(out)
    out.to_csv(config.SPLITS_CSV, index=False)
    print(out.groupby(["machine_id", "split", "label"]).size().unstack(fill_value=0))