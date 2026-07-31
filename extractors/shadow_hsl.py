import cv2
import numpy as np

from extractors.base import BaseExtractor


class ShadowHSLExtractor(BaseExtractor):
    """
    HSL-based shadow feature extraction.

    NOTE: this is a starting-point implementation — replace the body
    with whatever produced hsl_shadow_features.csv so results match
    your training data exactly.
    """

    name = "shadow_hsl"

    def extract(self, image: np.ndarray) -> np.ndarray:
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        lightness = hls[:, :, 1]
        # crude shadow mask: pixels darker than a threshold
        shadow_mask = (lightness < 80).astype(np.float32)
        return shadow_mask.flatten()
