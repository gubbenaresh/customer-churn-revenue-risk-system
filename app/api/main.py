from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.predict import predict_churn
from app.api.schemas import CustomerInput

app = FastAPI(
    title="Customer Churn Revenue Risk API",
    description="Predict customer churn probability, risk level and revenue at risk",
    version="1.0.0"
)


class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    AvgChargesPerTenure: float
    TenureGroup: str
    RiskRevenue: float


@app.get("/")
def home():
    return {"message": "Customer Churn Revenue Risk API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(customer: CustomerInput):
    result = predict_churn(customer.dict())
    return result