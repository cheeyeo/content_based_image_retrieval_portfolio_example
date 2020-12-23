import config.autoencoderconfig as config
from autoencoder.model import ConvAutoencoder
from utils.plot import plot_model_loss
from callbacks.learningrates import poly_decay
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from imutils import paths

if __name__ == "__main__":
    images_paths = list(paths.list_images(config.IMAGES_PATH))

    data = []
    for img_path in images_paths:
        img = load_img(img_path)
        img = img_to_array(img)
        data.append(img)

    # Normalize dataset
    data = np.array(data)
    data = data.astype("float32") / 255.0
    trainX = np.array(data)

    print("[INFO] Building model...")
    model = ConvAutoencoder.build(
        width=config.IMG_WIDTH, 
        height=config.IMG_HEIGHT,
        depth=3,
        latent_dim=config.LATENT_DIM)

    opt = Adam(lr=config.LR)
    # Using MSE as loss function
    model.compile(optimizer=opt, loss="mse")

    model.summary()
    plot_model(model, to_file=config.MODEL_PLOT_PATH, show_shapes=True, show_layer_names=True)

    print("[INFO] Training model")
    callbacks = [LearningRateScheduler(poly_decay)]
    H = model.fit(
        trainX, trainX, 
        epochs=config.EPOCHS,
        batch_size=config.BATCH,
        callbacks=callbacks)

    print("[INFO] Saving final model...")
    model.save(config.FINAL_MODEL_PATH)