USE ChurnDB;
GO

 -- Total customers
SELECT COUNT(*) AS total_customers
FROM customer_churn;

-- Churn rate
SELECT 
    ROUND(AVG(CAST(Churn AS FLOAT)) * 100 , 2)  AS churn_rate_percentage
FROM customer_churn;

-- Revenue at risk
SELECT 
      ROUND(SUM(MonthlyCharges),2) as monthly_revenue_at_risk
FROM customer_churn
where Churn=1; 

-- Churn by contract
SELECT 
    Contract,
    COUNT(*) AS total_customers,
    SUM(CAST(Churn AS FLOAT)) AS churned_customers,
    ROUND(AVG(CAST(Churn AS FLOAT)) * 100, 2) AS churn_rate
FROM customer_churn
GROUP BY Contract
ORDER BY churn_rate DESC;


-- Churn by payment method
SELECT 
    PaymentMethod,
    COUNT(*) AS total_customers,
    ROUND(AVG(CAST(Churn AS FLOAT)) * 100, 2) AS churn_rate
FROM customer_churn
GROUP BY PaymentMethod
ORDER BY churn_rate DESC;