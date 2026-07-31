import { TreeClassifierRunner } from './models/TreeClassifierRunner.js';

// Initialize when the page loads
document.addEventListener('DOMContentLoaded', async () => {
  console.log("App initializing...");

  // Instantiate your subclass
  const modelRunner = new TreeClassifierRunner(
    'https://huggingface.co/spaces/Rayan563/tree_test'
  );

  // Initialize and check status
  await modelRunner.initialize();
});
