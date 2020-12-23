# Builds the CBIR system by indexing the images using the encoder
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import config.autoencoderconfig as config
import numpy as np
from imutils import paths
import argparse
import pickle

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path of encoder model")
    ap.add_argument("--index", required=True, help="Path to store index/database file..")
    args = vars(ap.parse_args())

    print("[INFO] Loading dataset...")
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

    print("[INFO] Loading encoder model...")
    autoencoder = load_model(args["model"])
    encoder = Model(inputs=autoencoder.inputs, outputs=autoencoder.get_layer("encoder").output)

    plot_model(encoder, to_file=config.ENCODER_PLOT_PATH, show_shapes=True, show_layer_names=True)

    print("[INFO] Building CBIR database...")
    features = encoder.predict(trainX)
    indexes = list(range(0, trainX.shape[0]))
    data = {"indexes": indexes, "features": features}
    with open(args["index"], "wb") as f:
        f.write(pickle.dumps(data))