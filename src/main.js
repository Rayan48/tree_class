import { TreeClassifierRunner } from './models/TreeClassifierRunner.js';

document.addEventListener('DOMContentLoaded', async () => {
  console.log("Initializing ML Web App...");

  // Instantiate the child class (OOP Polymorphism / Inheritance)
  const treeClassifier = new TreeClassifierRunner(
    'https://huggingface.co/spaces/Rayan563/tree_test'
  );

  // Initialize the model
  await treeClassifier.initialize();

  // Test running a prediction if model loaded successfully
  if (treeClassifier.isLoaded) {
    const result = await treeClassifier.predict(null);
    console.log("Prediction Result:", result);
  }
});
