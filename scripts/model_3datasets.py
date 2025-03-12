import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load integrated multi-omics data and labels
def load_data(data_file, labels_file):
    data = pd.read_csv(data_file, index_col=0)
    labels = pd.read_csv(labels_file, index_col=0)
    return data, labels

# Train model (Random Forest)
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

# Save trained model
def save_model(model, model_file):
    joblib.dump(model, model_file)

def main():
    # Load data
    data, labels = load_data('data/processed/integrated_omics_data.csv', 'data/processed/disease_labels.csv')

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Save the model
    save_model(model, 'models/drug_target_discovery_model.pkl')

if __name__ == '__main__':
    main()
