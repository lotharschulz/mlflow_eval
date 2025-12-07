import mlflow
import pandas as pd
from mlflow.deployments import get_deploy_client

# Step 1: Create a new MLflow Experiment
mlflow.set_experiment("Ollama Model Evaluation")

# Step 2: Prepare evaluation data
eval_data = pd.DataFrame({
    "inputs": [
        "What is MLflow?",
        "What is the capital of France?",
        "Explain machine learning in simple terms.",
        "Write a Python function to calculate factorial.",
    ],
    "ground_truth": [
        "MLflow is an open-source platform for managing the machine learning lifecycle.",
        "Paris",
        "Machine learning is a way for computers to learn patterns from data without being explicitly programmed.",
        "A factorial function multiplies a number by all positive integers less than it.",
    ]
})

# Step 3: Define the model using Ollama
# You can use Ollama directly via its API endpoint
def ollama_model(inputs):
    """Wrapper function for Ollama model"""
    import requests
    
    responses = []
    for prompt in inputs["inputs"]:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            }
        )
        responses.append(response.json()["response"])
    
    return responses

# Alternative: Use MLflow's deployment client (if you prefer)
# client = get_deploy_client("ollama")
# This would require setting up Ollama as an MLflow deployment target

# Step 4: Run evaluation with MLflow
with mlflow.start_run(run_name="llama3.2-evaluation"):
    
    # Log parameters
    mlflow.log_param("model", "llama3.2")
    mlflow.log_param("provider", "ollama")
    mlflow.log_param("num_samples", len(eval_data))
    
    # Evaluate the model
    results = mlflow.evaluate(
        model=ollama_model,
        data=eval_data,
        targets="ground_truth",
        model_type="text",
        evaluators="default",
    )
    
    # Log additional metrics
    print("\n=== Evaluation Results ===")
    print(f"Metrics: {results.metrics}")
    
    # View results in the UI
    print(f"\nView results at: http://localhost:5000")
    print(f"Experiment: {mlflow.get_experiment_by_name('Ollama Model Evaluation').experiment_id}")

print("\nEvaluation complete! Check the MLflow UI for detailed results.")