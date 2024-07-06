
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def ClassificationInsights(df,target):
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

    """# Class Imbalance"""

    target_count = df[target].value_counts()
    target_count

    num_classes = len(df[target].unique())
    response["5number_of_classes"] = num_classes
    if num_classes > 1:
        target_count = df[target].value_counts()
        majority_class_count = max(target_count)
        minority_class_count = min(target_count)
        imbalance_ratio = majority_class_count / minority_class_count
        response["6imbalance_ratio"] = imbalance_ratio
        response["7imbalance_status"] = "SMOTE Required" if imbalance_ratio > 1.5 else "Smote Not Required"
        if imbalance_ratio > 1.5:
            # Perform SMOTE if class imbalance is present
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            x_resampled, y_resampled = smote.fit_resample(x_normalized, y)
            print("SMOTE applied")
            response["8smote_status"] = "SMOTE Applied"
        else:
            # No need for SMOTE
            x_resampled = x_normalized
            y_resampled = y
            response["8smote_status"] = "SMOTE Not Applied"
    else:
        # Only one class is present, SMOTE not applicable
        x_resampled = x_normalized
        y_resampled = y
        response["6imbalance_ratio"] = 1
        response["7imbalance_status"] = "Single Class Dataset"
        response["8smote_status"] = "SMOTE Not Applied"

    """# Various Models"""

    X_train, X_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

    # Define various classification models
    models = {
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC()
    }

    # Train and evaluate each model
    model_response = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name}: Accuracy = {accuracy:.2f}")
        print(classification_report(y_test, y_pred))
        model_response[name] = {
            "accuracy": accuracy,
            "classification_report": classification_report(y_test, y_pred),
        }
        response["9model_response"] = model_response


    # prompt: Plot Accuracy

    model_names = list(models.keys())
    accuracy_values = [accuracy_score(y_test, model.predict(X_test)) for model in models.values()]

    return response
