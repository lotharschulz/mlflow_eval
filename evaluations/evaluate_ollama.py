import mlflow
import pandas as pd
from langchain_ollama import OllamaLLM

# Set experiment
mlflow.set_experiment("Ollama Model Evaluation - Simple")

# Prepare data
eval_data = pd.DataFrame({
    "question": [
        "What is MLflow?",
        "What is the capital of Spain?",
        "Explain machine learning in simple terms.",
        "What is 2+2?",
        "Who wrote Romeo and Juliet?",
    ],
    "ground_truth": [
        "MLflow is an open-source platform for managing the machine learning lifecycle.",
        "Madrid",
        "Machine learning is a way for computers to learn patterns from data.",
        "4",
        "William Shakespeare",
    ]
})

# Initialize model and generate predictions
llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")

print("Generating predictions...")
predictions = []
for idx, question in enumerate(eval_data["question"], 1):
    print(f"[{idx}/{len(eval_data)}] Processing: {question[:60]}...")
    response = llm.invoke(question)
    predictions.append(response)

eval_data["prediction"] = predictions

# Run evaluation
with mlflow.start_run(run_name="llama3.2-simple-eval"):
    mlflow.log_param("model", "llama3.2")
    mlflow.log_param("provider", "ollama")
    mlflow.log_param("num_samples", len(eval_data))
    
    # Log the evaluation data as a table
    mlflow.log_table(data=eval_data, artifact_file="eval_results.json")
    
    # Calculate simple metrics manually
    from difflib import SequenceMatcher
    
    similarities = []
    for pred, truth in zip(eval_data["prediction"], eval_data["ground_truth"]):
        similarity = SequenceMatcher(None, pred.lower(), truth.lower()).ratio()
        similarities.append(similarity)
    
    avg_similarity = sum(similarities) / len(similarities)
    
    mlflow.log_metric("avg_similarity", avg_similarity)
    mlflow.log_metric("num_predictions", len(predictions))
    
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Average Similarity: {avg_similarity:.4f}")
    print(f"Number of samples: {len(predictions)}")
    
    print(f"\n{'='*60}")
    print("Sample Results:")
    print(f"{'='*60}")
    for i in range(min(3, len(eval_data))):
        print(f"\nQuestion: {eval_data.iloc[i]['question']}")
        print(f"Ground Truth: {eval_data.iloc[i]['ground_truth']}")
        print(f"Prediction: {eval_data.iloc[i]['prediction'][:150]}...")
        print(f"Similarity: {similarities[i]:.4f}")
    
    print(f"\n{'='*60}")
    print(f"🔗 View full results at: http://localhost:5000")
    print(f"{'='*60}")