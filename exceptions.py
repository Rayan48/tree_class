class TreeClassAppError(Exception):
    """Base exception for the app — lets the frontend catch one type broadly."""
    pass


class InvalidImageError(TreeClassAppError):
    """Raised when an uploaded file isn't a usable image."""
    pass


class FeatureExtractionError(TreeClassAppError):
    """Raised when a specific extractor fails on a given image."""

    def __init__(self, extractor_name, original_error):
        self.extractor_name = extractor_name
        self.original_error = original_error
        super().__init__(f"{extractor_name} failed: {original_error}")


class ModelLoadError(TreeClassAppError):
    """Raised when the .keras model can't be loaded."""
    pass
