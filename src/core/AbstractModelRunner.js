/**
 * Abstract Base Class for ML Model Runners
 * Ensures all classifiers share a standard lifecycle interface.
 */
export class AbstractModelRunner {
  constructor(modelUrl) {
    if (new.target === AbstractModelRunner) {
      throw new TypeError("Cannot instantiate abstract class AbstractModelRunner directly.");
    }
    this.modelUrl = modelUrl;
    this.isLoaded = false;
  }

  // Common method shared across all models
  async initialize() {
    try {
      await this.loadModel();
      this.isLoaded = true;
      console.log(`[Model Readiness]: ${this.constructor.name} loaded successfully.`);
    } catch (error) {
      this.handleError(error);
    }
  }

  // Abstract methods — subclasses MUST override these
  async loadModel() {
    throw new Error("Method 'loadModel()' must be implemented by subclass.");
  }

  async predict(inputData) {
    throw new Error("Method 'predict()' must be implemented by subclass.");
  }

  handleError(error) {
    throw new Error("Method 'handleError()' must be implemented by subclass.");
  }
}
