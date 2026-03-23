import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os
import pickle

def load_features(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['toss_win_match_win'])
    y = df['toss_win_match_win']
    print(f"✅ Features loaded: {X.shape}")
    return X, y

def train(X_train, y_train, params):
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    print(f"✅ Model trained!")
    return model

def save_model(model, path="models/model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to: {path}")

if __name__ == "__main__":
    # Config
    FEATURES_PATH = "data/processed/features.csv"
    MODEL_PATH    = "models/model.pkl"

    # Hyperparameters
    params = {
        "n_estimators": 50,
        "max_depth": 3,
        "random_state": 42
    }

    # MLflow experiment
    mlflow.set_experiment("cricket-outcome-predictor")

    with mlflow.start_run():
        # Load data
        X, y = load_features(FEATURES_PATH)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"✅ Train size: {X_train.shape} | Test size: {X_test.shape}")

        # Train
        model = train(X_train, y_train, params)

        # Evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "random_forest_model")

        # Save model locally
        save_model(model, MODEL_PATH)

        print(f"✅ Accuracy : {acc:.4f}")
        print(f"✅ F1 Score : {f1:.4f}")
        print(f"✅ MLflow Run ID: {mlflow.active_run().info.run_id}")