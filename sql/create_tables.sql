CREATE DATABASE ChurnDB;
GO

USE ChurnDB;
GO

CREATE TABLE customer_churn (
    gender VARCHAR(20),
    SeniorCitizen INT,
    Partner VARCHAR(20),
    Dependents VARCHAR(20),
    tenure INT,
    PhoneService VARCHAR(20),
    MultipleLines VARCHAR(50),
    InternetService VARCHAR(50),
    OnlineSecurity VARCHAR(50),
    OnlineBackup VARCHAR(50),
    DeviceProtection VARCHAR(50),
    TechSupport VARCHAR(50),
    StreamingTV VARCHAR(50),
    StreamingMovies VARCHAR(50),
    Contract VARCHAR(50),
    PaperlessBilling VARCHAR(20),
    PaymentMethod VARCHAR(100),
    MonthlyCharges FLOAT,
    TotalCharges FLOAT,
    Churn INT
);