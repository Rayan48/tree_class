import numpy as np
from sklearn.decomposition import PCA

from extractors.base import BaseExtractor


class PCAExtractor(BaseExtractor):
    """
    PCA-based dimensionality-reduced features.

    NOTE: for consistent results across runs, a PCA extractor should
    normally be *fit once* on your training set and then reused here
    (e.g. loaded from a pickled sklearn PCA object) rather than fit
    fresh on every single uploaded image. Replace this placeholder
    once you export the fitted PCA object used to build
    pca_image_features_with_labels.csv.
    """

    name = "pca"

    def __init__(self, n_components: int = 50):
        self.n_components = n_components

    def extract(self, image: np.ndarray) -> np.ndarray:
        gray = image.mean(axis=2)  # flatten RGB to grayscale for simplicity
        pca = PCA(n_components=min(self.n_components, min(gray.shape)))
        transformed = pca.fit_transform(gray)
        return transformed.flatten().astype(np.float32)
