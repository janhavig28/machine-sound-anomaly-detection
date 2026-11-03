# src/config.py
from pathlib import Path

SEED = 42

# ---- Paths ----
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = ROOT / "models"
METADATA_CSV = RESULTS_DIR / "metadata.csv"
SPLITS_CSV = RESULTS_DIR / "splits.csv"
FEATURES_DIR = ROOT / "features"
# ---- Data selection ----
MACHINE_TYPE = "fan"      # change later to pump / valve / slider

# ---- Audio (used from the next stage) ----
SAMPLE_RATE = 16000
DURATION = 10             # seconds per clip
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

# ---- Model ----
N_FRAMES = 312            # 313 frames cropped to 312 so pooling/upsampling match
BOTTLENECK_CHANNELS = 4   # smaller = tighter bottleneck

# ---- Training ----
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 5
MODEL_PATH = MODELS_DIR / "autoencoder.keras"
NORM_STATS_PATH = MODELS_DIR / "norm_stats.json"

# ---- Anomaly detection ----
THRESHOLD_PERCENTILE = 95   # percentile of validation reconstruction errors
SCORES_CSV = METRICS_DIR / "test_scores.csv"
THRESHOLD_PATH = METRICS_DIR / "threshold.json"