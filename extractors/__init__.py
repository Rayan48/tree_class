from extractors.edge import EdgeDetectionExtractor
from extractors.shadow_hsl import ShadowHSLExtractor
from extractors.pca import PCAExtractor
from extractors.raw import RawImageExtractor
from extractors.t_connected import TConnectedExtractor

# Polymorphism payoff: the frontend/pipeline never checks type,
# it just calls EXTRACTOR_REGISTRY[choice].run(image).
EXTRACTOR_REGISTRY = {
    "Edge Detection": EdgeDetectionExtractor(),
    "Shadow (HSL)": ShadowHSLExtractor(),
    "PCA": PCAExtractor(),
    "Raw Image": RawImageExtractor(),
    "T-Connected Pixel": TConnectedExtractor(),
}
