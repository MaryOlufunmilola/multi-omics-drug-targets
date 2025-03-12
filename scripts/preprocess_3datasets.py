import pandas as pd
import numpy as np
from fancyimpute import IterativeImputer
from sklearn.preprocessing import StandardScaler

# Load the datasets
def load_data(rna_seq_file, proteomics_file, metabolomics_file, labels_file):
    rna_seq_data = pd.read_csv(rna_seq_file, index_col=0)
    proteomics_data = pd.read_csv(proteomics_file, index_col=0)
    metabolomics_data = pd.read_csv(metabolomics_file, index_col=0)
    labels = pd.read_csv(labels_file, index_col=0)
    return rna_seq_data, proteomics_data, metabolomics_data, labels

# Preprocess the datasets (imputation, scaling)
def preprocess_data(rna_seq_data, proteomics_data, metabolomics_data):
    # Impute missing values
    imputer = IterativeImputer(max_iter=10, random_state=42)
    rna_seq_data_imputed = pd.DataFrame(imputer.fit_transform(rna_seq_data), columns=rna_seq_data.columns, index=rna_seq_data.index)
    proteomics_data_imputed = pd.DataFrame(imputer.fit_transform(proteomics_data), columns=proteomics_data.columns, index=proteomics_data.index)
    metabolomics_data_imputed = pd.DataFrame(imputer.fit_transform(metabolomics_data), columns=metabolomics_data.columns, index=metabolomics_data.index)

    # Standardize/Normalize the data
    scaler = StandardScaler()
    rna_seq_data_scaled = pd.DataFrame(scaler.fit_transform(rna_seq_data_imputed), columns=rna_seq_data.columns, index=rna_seq_data.index)
    proteomics_data_scaled = pd.DataFrame(scaler.fit_transform(proteomics_data_imputed), columns=proteomics_data.columns, index=proteomics_data.index)
    metabolomics_data_scaled = pd.DataFrame(scaler.fit_transform(metabolomics_data_imputed), columns=metabolomics_data.columns, index=metabolomics_data.index)

    return rna_seq_data_scaled, proteomics_data_scaled, metabolomics_data_scaled

# Save the preprocessed data
def save_preprocessed_data(rna_seq_data, proteomics_data, metabolomics_data, labels, output_data_dir):
    rna_seq_data.to_csv(f'{output_data_dir}/processed_rna_seq.csv')
    proteomics_data.to_csv(f'{output_data_dir}/processed_proteomics.csv')
    metabolomics_data.to_csv(f'{output_data_dir}/processed_metabolomics.csv')
    labels.to_csv(f'{output_data_dir}/processed_labels.csv')

def main():
    # Load the raw data
    rna_seq_file = 'data/rna_seq_data.csv'
    proteomics_file = 'data/proteomics_data.csv'
    metabolomics_file = 'data/metabolomics_data.csv'
    labels_file = 'data/disease_labels.csv'

    rna_seq_data, proteomics_data, metabolomics_data, labels = load_data(rna_seq_file, proteomics_file, metabolomics_file, labels_file)

    # Preprocess the data
    rna_seq_data_scaled, proteomics_data_scaled, metabolomics_data_scaled = preprocess_data(rna_seq_data, proteomics_data, metabolomics_data)

    # Save the preprocessed data
    save_preprocessed_data(rna_seq_data_scaled, proteomics_data_scaled, metabolomics_data_scaled, labels, 'data/processed')

if __name__ == '__main__':
    main()
