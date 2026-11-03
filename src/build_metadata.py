# src/build_metadata.py
import re
import pandas as pd
from src import config


def parse_path(wav_path):
    """Read machine, id, noise level and label from the file's folder path."""
    rel = wav_path.relative_to(config.DATASET_DIR).as_posix().lower()

    machine = re.search(r"(?<![a-z])(fan|pump|slider|valve)(?![a-z])", rel)
    machine_id = re.search(r"id_\d+", rel)
    snr = re.search(r"(-?\d+)_db", rel)

    # check "abnormal" first, because "normal" is inside that word
    if "/abnormal/" in rel:
        label = "abnormal"
    elif "/normal/" in rel:
        label = "normal"
    else:
        label = None

    return (
        machine.group(1) if machine else None,
        machine_id.group(0) if machine_id else None,
        snr.group(1) if snr else None,
        label,
    )


def build_metadata():
    wav_files = sorted(config.DATASET_DIR.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(
            f"No .wav files found in {config.DATASET_DIR}. "
            "Did you extract the dataset inside the dataset/ folder?"
        )

    rows = []
    for f in wav_files:
        machine, machine_id, snr, label = parse_path(f)
        if label is None:          # skip anything that isn't normal/abnormal
            continue
        rows.append({
            "filepath": f.relative_to(config.ROOT).as_posix(),
            "machine": machine,
            "machine_id": machine_id,
            "snr_db": snr,
            "label": label,
            "is_anomaly": int(label == "abnormal"),   # 1 = abnormal, 0 = normal
        })

    df = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.METADATA_CSV, index=False)
    return df


if __name__ == "__main__":
    df = build_metadata()
    print(f"Total files: {len(df)}")
    print(df.groupby(["machine", "snr_db", "machine_id", "label"])
            .size().unstack(fill_value=0))