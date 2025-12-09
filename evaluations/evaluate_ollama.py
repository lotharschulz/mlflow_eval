import mlflow
import pandas as pd
from langchain_ollama import OllamaLLM
from difflib import SequenceMatcher
import os

# Set experiment
mlflow.set_experiment("Ollama Model Evaluation - Simple")

# MLflow tracking URI (for MLflow server)
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
print(f"Using MLflow Tracking URI: {mlflow_tracking_uri}")

# Ollama base URL (for Ollama server)
ollama_base_url = os.getenv("OLLAMA_BASE_URI", "http://localhost:11434")
print(f"Using Ollama Base URL: {ollama_base_url}")

# Prepare data
eval_data = pd.DataFrame({
    "question": [
        "What is MLflow?",
        "What is the capital of Spain?",
        "Explain machine learning in simple terms.",
        "What is 2+2+2?",
        "Who wrote Romeo and Juliet?",
        "How many vowels are in Alabama?",
        "Which city is meant in the song \"We built this city\" by the group Starship?",
    ],
    "ground_truth": [
        "MLflow is an open-source platform for managing the machine learning lifecycle.",
        "The capital city of Spain is Madrid",
        "Machine learning is a way for computers to learn patterns from data.",
        "6",
        "William Shakespeare",
        "4",
        "Two cities are referenced: San Francisco and Los Angeles.",
    ]
})

# List of models to evaluate
models_to_evaluate = [
    "llama3.2:3b",
    "llama3.2:1b",
    "mistral:7b",
    "dolphin3:8b",
    "deepseek-r1:7b"
]

# Store results for final comparison
results_summary = []

# Evaluate each model
for model_name in models_to_evaluate:
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_name}")
    print(f"{'='*60}\n")
    
    try:
        # Initialize model and generate predictions
        llm = OllamaLLM(model=model_name, base_url=ollama_base_url)
        
        print("Generating predictions...")
        predictions = []
        for idx, question in enumerate(eval_data["question"], 1):
            print(f"[{idx}/{len(eval_data)}] Processing: {question[:60]}...")
            response = llm.invoke(question)
            predictions.append(response)
        
        # Create a copy of eval_data for this model
        model_eval_data = eval_data.copy()
        model_eval_data["prediction"] = predictions
        
        # Calculate simple metrics manually
        similarities = []
        for pred, truth in zip(predictions, eval_data["ground_truth"]):
            similarity = SequenceMatcher(None, pred.lower(), truth.lower()).ratio()
            similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities)
        min_similarity = min(similarities)
        max_similarity = max(similarities)
        
        # Run evaluation
        with mlflow.start_run(run_name=f"{model_name}-simple-eval"):
            mlflow.log_param("model", model_name)
            mlflow.log_param("provider", "ollama")
            mlflow.log_param("num_samples", len(eval_data))
            
            # Log metrics
            mlflow.log_metric("avg_similarity", avg_similarity)
            mlflow.log_metric("min_similarity", min_similarity)
            mlflow.log_metric("max_similarity", max_similarity)
            mlflow.log_metric("num_predictions", len(predictions))
            
            # Log the evaluation data as a table
            model_eval_data["similarity"] = similarities
            mlflow.log_table(data=model_eval_data, artifact_file="eval_results.json")
            
            print(f"\n{'='*60}")
            print(f"Evaluation Complete for {model_name}!")
            print(f"{'='*60}")
            print(f"Average Similarity: {avg_similarity:.4f}")
            print(f"Min Similarity: {min_similarity:.4f}")
            print(f"Max Similarity: {max_similarity:.4f}")
            print(f"Number of samples: {len(predictions)}")
            
            print(f"\n{'='*60}")
            print("Sample Results:")
            print(f"{'='*60}")
            for i in range(min(3, len(model_eval_data))):
                print(f"\nQuestion: {model_eval_data.iloc[i]['question']}")
                print(f"Ground Truth: {model_eval_data.iloc[i]['ground_truth']}")
                print(f"Prediction: {model_eval_data.iloc[i]['prediction'][:150]}...")
                print(f"Similarity: {similarities[i]:.4f}")
        
        # Store results for comparison
        results_summary.append({
            "model": model_name,
            "avg_similarity": avg_similarity,
            "min_similarity": min_similarity,
            "max_similarity": max_similarity,
            "status": "✓ Success"
        })
        
    except Exception as e:
        print(f"\n✗ Error evaluating {model_name}: {e}")
        results_summary.append({
            "model": model_name,
            "avg_similarity": 0.0,
            "min_similarity": 0.0,
            "max_similarity": 0.0,
            "status": f"✗ Failed: {str(e)[:50]}"
        })
        continue

# Display final comparison
print(f"\n\n{'='*80}")
print("FINAL MODEL COMPARISON")
print(f"{'='*80}\n")

summary_df = pd.DataFrame(results_summary)
# Sort by avg_similarity descending
summary_df = summary_df.sort_values("avg_similarity", ascending=False)

# Format for display
display_df = summary_df.copy()
display_df['avg_similarity'] = display_df['avg_similarity'].apply(lambda x: f"{x:.4f}")
display_df['min_similarity'] = display_df['min_similarity'].apply(lambda x: f"{x:.4f}")
display_df['max_similarity'] = display_df['max_similarity'].apply(lambda x: f"{x:.4f}")

print(display_df.to_string(index=False))

# Determine winner
successful_models = summary_df[summary_df['status'] == '✓ Success']
if len(successful_models) > 0:
    winner = successful_models.iloc[0]
    print(f"\n{'='*80}")
    print(f"🏆 WINNER: {winner['model'].upper()}")
    print(f"{'='*80}")
    print(f"   Average Similarity: {winner['avg_similarity']:.4f}")
    print(f"   Min Similarity: {winner['min_similarity']:.4f}")
    print(f"   Max Similarity: {winner['max_similarity']:.4f}")
    print(f"{'='*80}")
    
    # Show top 3 if available
    if len(successful_models) >= 3:
        print(f"\n🥇 1st Place: {successful_models.iloc[0]['model']} ({successful_models.iloc[0]['avg_similarity']:.4f})")
        print(f"🥈 2nd Place: {successful_models.iloc[1]['model']} ({successful_models.iloc[1]['avg_similarity']:.4f})")
        print(f"🥉 3rd Place: {successful_models.iloc[2]['model']} ({successful_models.iloc[2]['avg_similarity']:.4f})")
else:
    print(f"\n{'='*80}")
    print("⚠️  No models completed successfully")
    print(f"{'='*80}")

print(f"\n🔗 View detailed results at: {mlflow_tracking_uri}")
print(f"{'='*80}\n")