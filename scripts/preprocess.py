# This script handles merging and preprocessing of RNA-Seq and proteomics data. It involves normalization, handling missing values, and merging both datasets.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from fancyimpute import IterativeImputer

# Load RNA-Seq and proteomics data
def load_data(rna_file, proteomics_file):
    rna_data = pd.read_csv(rna_file, index_col=0)  # RNA-Seq data
    proteomics_data = pd.read_csv(proteomics_file, index_col=0)  # Proteomics data
    return rna_data, proteomics_data

# Preprocess the data (e.g., normalization, imputation)
def preprocess_data(rna_data, proteomics_data):
    # Handle missing values using iterative imputation
    imputer = IterativeImputer(max_iter=10, random_state=42)
    rna_data_imputed = pd.DataFrame(imputer.fit_transform(rna_data), columns=rna_data.columns, index=rna_data.index)
    proteomics_data_imputed = pd.DataFrame(imputer.fit_transform(proteomics_data), columns=proteomics_data.columns, index=proteomics_data.index)

    # Standardize the data (z-score normalization)
    scaler = StandardScaler()
    rna_data_scaled = pd.DataFrame(scaler.fit_transform(rna_data_imputed), columns=rna_data.columns, index=rna_data.index)
    proteomics_data_scaled = pd.DataFrame(scaler.fit_transform(proteomics_data_imputed), columns=proteomics_data.columns, index=proteomics_data.index)

    # Merge the datasets (inner join on samples)
    merged_data = pd.concat([rna_data_scaled, proteomics_data_scaled], axis=1)
    return merged_data

# Save preprocessed data
def save_preprocessed_data(merged_data, output_file):
    merged_data.to_csv(output_file)

def main():
    # Load data
    rna_file = 'data/rna_seq_data.csv'
    proteomics_file = 'data/proteomics_data.csv'
    rna_data, proteomics_data = load_data(rna_file, proteomics_file)

    # Preprocess data
    merged_data = preprocess_data(rna_data, proteomics_data)

    # Save preprocessed data
    save_preprocessed_data(merged_data, 'data/merged_data.csv')

if __name__ == '__main__':
    main()
