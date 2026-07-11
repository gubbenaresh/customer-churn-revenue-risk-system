import joblib
import pandas as pd

MODEL_PATH = "models/churn_model.pkl"

model = joblib.load(MODEL_PATH)


def get_risk_level(probability):
    if probability >= 0.70:
        return "High Risk"
    elif probability >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"


def get_recommendation(probability):
    if probability >= 0.70:
        return "Call customer and offer retention discount"
    elif probability >= 0.40:
        return "Send personalized offer and monitor usage"
    else:
        return "No immediate action required"


def predict_churn(input_data):
    df = pd.DataFrame([input_data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 3),
        "risk_level": get_risk_level(probability),
        "recommendation": get_recommendation(probability),
        "monthly_revenue_at_risk": input_data.get("MonthlyCharges", 0) if prediction == 1 else 0
    }