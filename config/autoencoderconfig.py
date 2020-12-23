# Config for autoencoder model
import os

# LR = 1e-2
LR = 1e-3
EPOCHS = 60
BATCH = 32

# Encoder unit size
LATENT_DIM = 192

IMG_WIDTH = 256
IMG_HEIGHT = 256

IMAGES_PATH = "dataset"

OUTPUT_PATH = "output"

MODEL_PLOT_PATH = os.path.sep.join([OUTPUT_PATH, "autoencoder_model.png"])

ENCODER_PLOT_PATH = os.path.sep.join([OUTPUT_PATH, "encoder_model.png"])

SAVED_MODEL_PATH = os.path.sep.join([OUTPUT_PATH, "autoencoder.h5"])

FINAL_MODEL_PATH = os.path.sep.join([OUTPUT_PATH, "final_model.h5"])

FIG_PATH = os.path.sep.join([OUTPUT_PATH, "autoencoder_loss.png"])

FINAL_FIG_PATH = os.path.sep.join([OUTPUT_PATH, "final_model_loss.png"])

NUM_SAMPLES = 20