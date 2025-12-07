import mlflow
import pandas as pd
from langchain_ollama import OllamaLLM

# Set experiment
mlflow.set_experiment("Ollama Model Evaluation")

# Prepare data
eval_data = pd.DataFrame({
    "inputs": [
        "What is MLflow?",
        "What is the capital of Spain?",
        "Explain machine learning in simple terms.",
    ],
    "ground_truth": [
        "MLflow is an open-source platform for managing the machine learning lifecycle.",
        "Madrid is the capital of Spain.",
        "Machine learning is a way for computers to learn patterns from data without being explicitly programmed.",
    ]
})

# Initialize model
llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")

# Create wrapper
def model_fn(inputs):
    if isinstance(inputs, pd.DataFrame):
        return [llm.invoke(text) for text in inputs["inputs"]]
    return [llm.invoke(inp) for inp in inputs]

# Run evaluation
with mlflow.start_run(run_name="llama3.2-eval"):
    mlflow.log_param("model", "llama3.2")
    mlflow.log_param("provider", "ollama")
    
    # Use mlflow.models.evaluate instead
    results = mlflow.models.evaluate(
        model=model_fn,
        data=eval_data,
        targets="ground_truth",
        model_type="text",
    )
    
    print(f"\nEvaluation Metrics:")
    for metric, value in results.metrics.items():
        print(f"  {metric}: {value}")
    
    print(f"\nView detailed results at: http://localhost:5000")