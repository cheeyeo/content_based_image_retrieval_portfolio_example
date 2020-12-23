import numpy as np

def visualize_predictions(decoded, gt, samples=10):
    outputs = None

    for i in range(0, samples):
        orig = (gt[i] * 255).astype("uint8")
        recon = (decoded[i] * 255).astype("uint8")

        output = np.hstack([orig, recon])

        if outputs is None:
            outputs = output
        else:
            outputs = np.vstack([outputs, output])

    return outputs