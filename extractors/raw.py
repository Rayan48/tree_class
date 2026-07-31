import cv2
import numpy as np

from extractors.base import BaseExtractor


class RawImageExtractor(BaseExtractor):
    """Raw pixel values, resized to a fixed shape."""

    name = "raw_image"

    def __init__(self, size=(64, 64)):
        self.size = size

    def extract(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, self.size)
        return resized.flatten().astype(np.float32) / 255.0
