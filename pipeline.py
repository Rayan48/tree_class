from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from exceptions import FeatureExtractionError
from extractors import EXTRACTOR_REGISTRY


class ClassificationPipeline:
    """Orchestrates extractor + classifier. Demonstrates multithreading + collections."""

    def __init__(self, classifier):
        self.classifier = classifier

    def run_single(self, image_file, method_name: str):
        extractor = EXTRACTOR_REGISTRY[method_name]
        features = extractor.run(image_file)
        return self.classifier.predict(features)

    def run_all(self, image_file) -> "OrderedDict":
        """Run every extractor concurrently, collect results in a fixed order."""
        results = OrderedDict()
        with ThreadPoolExecutor(max_workers=len(EXTRACTOR_REGISTRY)) as pool:
            futures = {
                pool.submit(self.run_single, image_file, name): name
                for name in EXTRACTOR_REGISTRY
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except FeatureExtractionError as e:
                    results[name] = f"Error: {e}"
        return results
