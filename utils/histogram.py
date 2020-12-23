from imutils import paths
import numpy as np
import cv2

def quantify_image(image, bins=[4, 6, 3]):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    # print(hist)
    # print(hist.shape)
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_histogram(image, bins=64):
    """
    Take histogram info from image on per channel basis

    Uses 64 bins for each channel as default
    """
    res = []
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        res.extend(hist)

    res = np.asarray(res)
    return res


if __name__ == "__main__":
    import os
    path = os.path.sep.join([os.getcwd(), "dataset", "coast_arnat59.jpg"])
    print(path)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(quantify_image(img, bins=[4, 6, 8]))

    # print("Testing per channel...")
    # print(extract_histogram(img, bins=192))