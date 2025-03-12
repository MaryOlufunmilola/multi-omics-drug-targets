This script defines and trains a machine learning model using the preprocessed multi-omics data. For simplicity, we'll use a Random Forest classifier for the demonstration, but this could be expanded to more complex models like deep neural networks.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# Load the merged and preprocessed data
def load_data(file_path):
    return pd.read_csv(file_path, index_col=0)

# Split data into features (X) and target (y)
def split_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

# Train Random Forest model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

# Perform cross-validation
def cross_validate_model(model, X, y):
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Average CV Score: {cv_scores.mean()}")

def main():
    # Load preprocessed data
    data = load_data('data/merged_data.csv')

    # Split data into features and target
    target_column = 'target'  # Replace with the actual target column name in the dataset
    X, y = split_data(data, target_column)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Perform cross-validation
    cross_validate_model(model, X, y)

if __name__ == '__main__':
    main()
