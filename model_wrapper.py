import tensorflow as tf

from exceptions import ModelLoadError


class TreeClassifier:
    """Wraps the trained .keras model behind a simple predict() interface."""

    def __init__(self, model_path: str = "plant_model.keras"):
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            raise ModelLoadError(f"Could not load model from {model_path}: {e}")

    def predict(self, features):
        features = features.reshape(1, -1)
        preds = self.model.predict(features)
        return preds
