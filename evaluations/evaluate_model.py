import mlflow
# no need for mlflow.set_tracking_uri()
# it reads from environment variable MLFLOW_TRACKING_URI

with mlflow.start_run():
    results = mlflow.evaluate(
        model=your_model,
        data=eval_data
    )