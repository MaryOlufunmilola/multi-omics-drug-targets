# This script handles the prediction and evaluation of the trained model. It allows for the prediction of new samples and evaluation metrics.

import pandas as pd
import joblib
from sklearn.metrics import classification_report

# Load the model
def load_model(model_path):
    return joblib.load(model_path)

# Load the new data for prediction
def load_data(file_path):
    return pd.read_csv(file_path, index_col=0)

# Predict using the trained model
def predict(model, X):
    return model.predict(X)

# Evaluate model performance
def evaluate_model(y_true, y_pred):
    print(classification_report(y_true, y_pred))

def main():
    # Load the trained model
    model = load_model('models/random_forest_model.pkl')

    # Load the new data for prediction
    data = load_data('data/new_samples.csv')  # Replace with actual prediction data
    X = data.drop(columns=['target'])  # Drop the target column if present in the prediction data
    y_true = data['target']  # True labels, if available

    # Predict using the trained model
    y_pred = predict(model, X)

    # Evaluate the model
    evaluate_model(y_true, y_pred)

if __name__ == '__main__':
    main()
