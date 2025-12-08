# MLflow Model Evaluation with Ollama

This project demonstrates how to evaluate multiple Ollama LLM models using MLflow for experiment tracking and comparison.

## Prerequisites

- Python 3.x
- Docker and Docker Compose
- Git
- Ollama

## Setup Instructions

### 1. Create and Activate Virtual Environment

```bash
python -m venv env
source env/bin/activate
```

To deactivate the virtual environment later:
```bash
deactivate
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up MLflow Tracking Server

Clone the MLflow repository with sparse checkout to get only the docker-compose files:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/mlflow/mlflow.git
cd mlflow
git sparse-checkout set docker-compose
cd docker-compose
cp .env.dev.example .env
```

Start the MLflow tracking server using Docker Compose:

```bash
docker compose up -d
```

Verify the setup:
```bash
docker-compose config
docker-compose logs -f mlflow
```

To stop the Docker setup:
```bash
docker-compose down -v  # The -v flag removes volumes storing data
```

### 4. Configure Environment Variables

```bash
cd ../../
export MLFLOW_TRACKING_URI="http://localhost:5000"
echo $MLFLOW_TRACKING_URI

export OLLAMA_BASE_URI="http://localhost:11434"
echo $OLLAMA_BASE_URI
```

### 5. Prepare Ollama Models

Navigate to the evaluations directory and run the preparation script:

```bash
cd evaluations/
./prepare_ollama_evaluation.sh
```

This script will:
- Check if Ollama is installed (install if not found)
- Check for the latest Ollama version and update if needed
- Pull the required models

## Running Evaluations

### Test Tracking URI

```bash
./tracking_uri_test.sh
```

### Run Model Evaluation

```bash
./evaluate_ollama.sh
```

## Evaluation Metrics

The evaluation script measures:
- **Average Similarity**: Average similarity score between predictions and ground truth
- **Min/Max Similarity**: Range of similarity scores
- **Number of Predictions**: Total predictions made

## Viewing Results

After running the evaluation, view the results in the MLflow UI on http://localhost:5000

## Project Structure

```
mlflow_eval/
├── env/                           # Virtual environment
├── mlflow/                        # MLflow docker-compose files
├── evaluations/
│   ├── evaluate_ollama.py        # Main evaluation script
│   ├── evaluate_ollama.sh        # Execution wrapper script
│   ├── prepare_ollama_evaluation.sh  # Ollama setup script
│   └── tracking_uri_test.sh      # URI verification script
├── requirements.txt              # Python dependencies
├── notes.txt                     # Development notes
└── README.md                     # This file
```

## Dependencies

See `requirements.txt` for the complete list. 

## Troubleshooting

### Python not found
Make sure the virtual environment is activated:
```bash
source env/bin/activate
```

### Ollama connection errors
Verify Ollama is running:
```bash
ollama list
```

### MLflow tracking issues
Check the MLflow server is running:
```bash
docker-compose logs -f mlflow
```

Verify the tracking URI:
```bash
echo $MLFLOW_TRACKING_URI
```
