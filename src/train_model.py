import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from data_preprocessing import load_data, split_data, get_preprocessor


# If XGBoost is not installed, code will still run without it
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


DATA_PATH = "data/processed/feature_engineered_churn_data.csv"
MODEL_PATH = "models/churn_model.pkl"


def evaluate_model(model, X_test, y_test):
    """
    This function checks model performance.
    It returns accuracy, precision, recall, f1-score and roc-auc.
    """

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    return accuracy, precision, recall, f1, roc_auc, y_pred


def train():
    # Load dataset
    df = load_data(DATA_PATH)

    # Split data into train and test
    X_train, X_test, y_train, y_test = split_data(df)

    # Get preprocessing pipeline
    preprocessor = get_preprocessor(X_train)

    # Models dictionary
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    }

    # Add XGBoost only if installed
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )

    results = []
    best_model = None
    best_model_name = None
    best_f1_score = 0

    # Train each model one by one
    for model_name, classifier in models.items():
        print("\n" + "=" * 50)
        print(f"Training Model: {model_name}")
        print("=" * 50)

        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier)
            ]
        )

        # Train model
        model_pipeline.fit(X_train, y_train)

        # Evaluate model
        accuracy, precision, recall, f1, roc_auc, y_pred = evaluate_model(
            model_pipeline,
            X_test,
            y_test
        )

        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1 Score:", round(f1, 4))
        print("ROC AUC:", round(roc_auc, 4))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        results.append({
            "Model": model_name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1 Score": round(f1, 4),
            "ROC AUC": round(roc_auc, 4)
        })

        # Select best model based on F1 score
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model = model_pipeline
            best_model_name = model_name

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 50)
    print("Final Model Comparison")
    print("=" * 50)
    print(results_df)

    # Save comparison report
    os.makedirs("reports", exist_ok=True)
    results_df.to_csv("reports/model_comparison.csv", index=False)

    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    print("\nBest Model:", best_model_name)
    print("Best F1 Score:", round(best_f1_score, 4))
    print("Best model saved at:", MODEL_PATH)


if __name__ == "__main__":
    train()