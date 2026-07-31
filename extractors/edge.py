import cv2
import numpy as np

from extractors.base import BaseExtractor


class EdgeDetectionExtractor(BaseExtractor):
    """Canny edge detection features."""

    name = "edge_detection"

    def extract(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return edges.flatten().astype(np.float32)
