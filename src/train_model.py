import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , f1_score , recall_score , roc_auc_score , precision_score , classification_report

from data_preprocessing import load_data , split_data , get_preprocessor 

DATA_PATH = "data/processed/feature_engineered_churn_data.csv"
MODEL_PATH = "models/churn_model.pkl"

def train():

    df = load_data(DATA_PATH) 

    X_train , X_test , y_train , y_test = split_data(df) 

    preprocessor = get_preprocessor(X_train)

    model = Pipeline(
        steps=[
            ("preprocessor",preprocessor),
            ("classifier",LogisticRegression(max_iter=1000))
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print("Model saved successfully")


if __name__ == "__main__":
    train()