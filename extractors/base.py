from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from exceptions import FeatureExtractionError, InvalidImageError


class BaseExtractor(ABC):
    """
    Abstract base class (Interface + shared implementation).

    Every concrete extractor MUST implement extract(). The shared
    load_image() / run() logic here is the "Abstract Class" concept
    in action: common behavior lives once, specific behavior is
    pushed down to subclasses.
    """

    name = "base"  # override in each subclass

    def load_image(self, image_path_or_file) -> np.ndarray:
        try:
            img = Image.open(image_path_or_file).convert("RGB")
            return np.array(img)
        except Exception as e:
            raise InvalidImageError(f"Could not open image: {e}")

    def run(self, image_path_or_file) -> np.ndarray:
        """Template method: shared error handling wraps each subclass's extract()."""
        img = self.load_image(image_path_or_file)
        try:
            return self.extract(img)
        except Exception as e:
            raise FeatureExtractionError(self.name, e)

    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Subclasses implement their specific feature logic here."""
        raise NotImplementedError

