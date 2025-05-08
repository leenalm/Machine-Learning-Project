import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

def preprocess_data(df):
    """
    Preprocess the input DataFrame by handling missing values, encoding categorical variables, and scaling numeric features.
    Args:
        df: pandas DataFrame containing input features.

    Returns:
        Preprocessed DataFrame ready for prediction.
    """
    # Assuming 'Attrition' column has been removed earlier
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Encoding categorical columns with LabelEncoder or OneHotEncoder
    # For simplicity, we use label encoding for categorical features
    for col in categorical_columns:
        df[col] = df[col].astype('category').cat.codes

    # Scaling numerical features using StandardScaler
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Save the scaler for later use
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return df



