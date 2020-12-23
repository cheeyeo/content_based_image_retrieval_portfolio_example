# Using MSE to evaluate autoencoder
import config.autoencoderconfig as config
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from utils.visual import visualize_predictions
import cv2
import argparse
from imutils import paths
import random
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis", help="visualization plot")
    ap.add_argument("--model", help="path to autoencoder model")
    args = vars(ap.parse_args())

    # TODO: How to load test set
    print("[INFO] Loading test data...")
    images_paths = list(paths.list_images(config.IMAGES_PATH))

    random.shuffle(images_paths)

    data = []
    for img_path in images_paths:
        img = load_img(img_path)
        img = img_to_array(img)
        data.append(img)

    # Normalize dataset
    data = np.array(data)
    data = data.astype("float32") / 255.0
    _, testX = train_test_split(data, test_size=0.25, random_state=42)

    print(testX.shape)


    # TODO: Load the encoder....
    print("[INFO] Loading model...")
    autoencoder = load_model(args["model"])
    autoencoder.summary()

    print("[INFO] Evaluating model...")
    decoded = autoencoder.predict(testX)
    reconstruction_loss = np.mean((testX - decoded) ** 2)
    print("Reconstruction Loss: {:.6f}".format(reconstruction_loss))

    print("[INFO] Making predictions...")
    samples = testX[:config.NUM_SAMPLES]
    vis = visualize_predictions(decoded, testX, samples=config.NUM_SAMPLES)
    cv2.imwrite(args["vis"], vis)