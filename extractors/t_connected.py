import cv2
import numpy as np

from extractors.base import BaseExtractor


class TConnectedExtractor(BaseExtractor):
    """
    T-connected pixel features (connected-component structure).

    NOTE: placeholder implementation using OpenCV connected components
    on a binarized image. Replace with whatever exact method produced
    t_connected_image_features.csv so results line up with training data.
    """

    name = "t_connected_pixel"

    def extract(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        num_labels, labels = cv2.connectedComponents(binary)
        return labels.flatten().astype(np.float32)
