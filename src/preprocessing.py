# src/preprocessing.py
import librosa
from src import config

TARGET_LEN = config.SAMPLE_RATE * config.DURATION   # 160000 samples


def load_audio(path):
    """Load a wav file as mono, at the target sample rate, exactly 10 s long."""
    try:
        y, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
    except Exception as e:
        raise RuntimeError(f"Could not load {path}: {e}")
    # pad with zeros if shorter, cut if longer
    return librosa.util.fix_length(y, size=TARGET_LEN)