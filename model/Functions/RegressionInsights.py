import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import numpy as np

def RegressionInsights(df, target):
    response = {}

    """# Missing Value Analysis"""
    missing_values = df.isnull().sum()
    all_zeroes = all(element == 0 for element in missing_values)
    if(all_zeroes):
        print("No missing values")
        response["1missing_values"] = "No missing values"
        response["2missing_value_filling"] = "Not Required"
    else:
        print("Missing values")
        response["1missing_values"] = "Missing values"
        object_columns = df.select_dtypes(include=['object']).columns
        numeric_columns = df.select_dtypes(include=['int', 'float']).columns
        for col in object_columns:
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)
        for col in numeric_columns:
            mean_value = df[col].mean()
            df[col].fillna(mean_value, inplace=True)
        response["2missing_value_filling"] = "Done"

    x = df.drop(target, axis=1)
    y = df[target]

    """# Encoding"""
    le = LabelEncoder()
    for col in x.select_dtypes(include=['object']).columns:
        x[col] = le.fit_transform(x[col])
    response["3encoding"] = "Label Encoding Done"

    scaler = MinMaxScaler()
    x_normalized = scaler.fit_transform(x)
    x_normalized[0:5]
    response["4normalisation"] = "Min Max Normalization Done"

    """# Various Models"""
    X_train, X_test, y_train, y_test = train_test_split(x_normalized, y, test_size=0.2, random_state=42)

    # Define various regression models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
        "Support Vector Regressor": SVR()
    }

    # Train and evaluate each model
    model_response = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name}: MSE = {mse:.2f}, MAE = {mae:.2f}, R2 = {r2:.2f}")
        model_response[name] = {
            "mean_squared_error": mse,
            "mean_absolute_error": mae,
            "r2_score": r2,
        }
        response["5model_response"] = model_response

    return response
