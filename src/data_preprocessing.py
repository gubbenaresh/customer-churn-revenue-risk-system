import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , StandardScaler 

def load_data(path):
    return pd.read_csv(path) 

def split_data(df , target_col="Churn"):
    X = df.drop(columns=[target_col])
    y = df[target_col] 

    return train_test_split(
        X , y , test_size=0.2 , random_state=42 , stratify=y  
    )

def get_preprocessor(X):

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist() 

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",StandardScaler(),numeric_features),
            ("cat",OneHotEncoder(handle_unknown="ignore"),categorical_features)
        ]
    )

    return preprocessor 


