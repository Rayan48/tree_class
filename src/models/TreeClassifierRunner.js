// Correct relative path from src/main.js:
import { TreeClassifierRunner } from './models/TreeClassifierRunner.js';

/**
 * Concrete Subclass implementing the abstract methods for Hugging Face Inference
 */
export class TreeClassifierRunner extends AbstractModelRunner {
  constructor(modelUrl) {
    super(modelUrl); // Call parent constructor
    this.apiEndpoint = modelUrl;
  }

  // 1. Implementation of abstract method loadModel()
  async loadModel() {
    console.log(`Connecting to Hugging Face Space: ${this.apiEndpoint}`);
    
    // Ping the space to check if it's live
    const response = await fetch(this.apiEndpoint);
    if (!response.ok) {
      throw new Error(`Failed to reach Hugging Face Space. Status: ${response.status}`);
    }
  }

  // 2. Implementation of abstract method predict()
  async predict(imageData) {
    if (!this.isLoaded) {
      throw new Error("Cannot run prediction: Model is not initialized yet.");
    }

    console.log("Processing image and sending request to Hugging Face...");
    
    // Mock response structure (Replace with actual Gradio / Hugging Face Client call)
    return {
      prediction: "Tree Detected",
      confidence: 0.94,
      timestamp: new Date().toISOString()
    };
  }

  // 3. Implementation of abstract method handleError()
  handleError(error) {
    console.error("🚨 [TreeClassifier Error]:", error.message);
    
    // Update the UI if an error element exists
    const statusElement = document.getElementById("model-status");
    if (statusElement) {
      statusElement.innerText = `Error: ${error.message}`;
      statusElement.style.color = "red";
    }
  }
}
