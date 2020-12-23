# Uses the encoder to search for input images matching the encoded features 
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from imutils import build_montages
from imutils import paths
from sklearn.model_selection import train_test_split
import config.autoencoderconfig as config
import numpy as np
import argparse
import pickle
import cv2
import random

def euclidean(a, b):
    return np.linalg.norm(a-b)

def perform_search(features, index, max_results=64):
    results = []

    for i in range(0, len(index["features"])):
        d = euclidean(features, index["features"][i])
        results.append((d, i))

    results = sorted(results)[:max_results]
    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, type=str, help="path to trained autoencoder")
    ap.add_argument("-i", "--index", required=True, type=str, help="path to index of features")
    ap.add_argument("-s", "--sample", type=int, default=10, help="number of testing queries to perform")

    args = vars(ap.parse_args())

    print("[INFO] Loading dataset...")
    images_paths = list(paths.list_images(config.IMAGES_PATH))

    data = []
    for img_path in images_paths:
        img = load_img(img_path)
        img = img_to_array(img)
        data.append(img)

    # Normalize dataset
    data = np.asarray(data)
    data = data.astype("float32") / 255.0
    trainX = np.asarray(data)
    _, testX = train_test_split(data, test_size=0.25, random_state=42)

    print("[INFO] Loading encoder...")
    autoencoder = load_model(args["model"])
    encoder = Model(inputs=autoencoder.inputs, outputs=autoencoder.get_layer("encoder").output)

    print("[INFO] Loading image features index...")
    with open(args["index"], "rb") as f:
        index = pickle.loads(f.read())

    print("[INFO] Encoding testing images...")
    features = encoder.predict(testX)

    # Randomly sample from test set for query
    query_idxs = list(range(0, testX.shape[0]))
    query_idxs = np.random.choice(query_idxs, size=args["sample"], replace=False)

    for i in query_idxs:
        # take features for current image, find similar images, init list of current images
        query_features = features[i]
        results = perform_search(query_features, index, max_results=10)

        images = []
        for (d,j) in results:
            # grab result image, convert to [0, 255]
            image = (trainX[j] * 255).astype("uint8")
            images.append(image)

        query = (testX[i] * 255).astype("uint8")
        cv2.imwrite("query/query_{}.jpg".format(i), query)
        montage = build_montages(images, (256, 256), (2, 5))[0]
        cv2.imwrite("query/results_{}.jpg".format(i), montage)