# src/feature_extraction.py
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from src import config


def wav_to_logmel(y):
    """Audio array -> log-Mel spectrogram of shape (N_MELS, frames)."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        power=2.0,
    )
    # ref=1.0 and top_db=None keep each clip's real loudness (no per-clip rescaling)
    return librosa.power_to_db(mel, ref=1.0, top_db=None).astype(np.float32)


def compute_stats(specs):
    """Mean and std from TRAINING spectrograms only (used in a later stage)."""
    arr = np.stack(specs)
    return float(arr.mean()), float(arr.std())


def normalize(spec, mean, std):
    """Scale a spectrogram using the training mean and std."""
    return (spec - mean) / (std + 1e-8)


def plot_logmel(spec, title, save_path):
    """Save a spectrogram picture to disk."""
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(
        spec, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
        x_axis="time", y_axis="mel", ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)