# src/model.py
import numpy as np
from tensorflow.keras import layers, models
from src import config


def to_model_input(spec):
    """(N_MELS, frames) -> (N_MELS, N_FRAMES, 1): crop extra frames, add channel axis."""
    return spec[:, :config.N_FRAMES, np.newaxis]


def build_autoencoder(bottleneck_channels=config.BOTTLENECK_CHANNELS):
    inp = layers.Input(shape=(config.N_MELS, config.N_FRAMES, 1))

    # ---- Encoder: compress the spectrogram ----
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D(2)(x)                    # 32 x 156
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)                    # 16 x 78
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)                    # 8 x 39

    # ---- Bottleneck: 8 x 39 x 4 = 1248 values vs 19968 in the input ----
    z = layers.Conv2D(bottleneck_channels, 3, padding="same",
                      activation="relu", name="bottleneck")(x)

    # ---- Decoder: rebuild the spectrogram ----
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(z)
    x = layers.UpSampling2D(2)(x)                    # 16 x 78
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.UpSampling2D(2)(x)                    # 32 x 156
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.UpSampling2D(2)(x)                    # 64 x 312
    out = layers.Conv2D(1, 3, padding="same", activation="linear")(x)

    model = models.Model(inp, out, name="cnn_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model