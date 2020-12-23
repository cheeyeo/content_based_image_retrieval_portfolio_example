import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import config.autoencoderconfig as config

def plot_model_loss(H, figpath=config.FIG_PATH):
    N = np.arange(0, config.EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(figpath)