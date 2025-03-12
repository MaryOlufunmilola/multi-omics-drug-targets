import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load preprocessed data
def load_preprocessed_data(rna_seq_file, proteomics_file, metabolomics_file):
    rna_seq_data = pd.read_csv(rna_seq_file, index_col=0)
    proteomics_data = pd.read_csv(proteomics_file, index_col=0)
    metabolomics_data = pd.read_csv(metabolomics_file, index_col=0)
    return rna_seq_data, proteomics_data, metabolomics_data

# Integrate the omics data (e.g., using concatenation or multi-view learning)
def integrate_data(rna_seq_data, proteomics_data, metabolomics_data):
    # Example: Concatenate data horizontally (on common features/genes/proteins)
    integrated_data = pd.concat([rna_seq_data, proteomics_data, metabolomics_data], axis=1)
    
    # Optional: Apply PCA for dimensionality reduction if needed
    pca = PCA(n_components=50)
    integrated_data_reduced = pca.fit_transform(integrated_data)
    
    return pd.DataFrame(integrated_data_reduced)

# Save the integrated data
def save_integrated_data(integrated_data, output_file):
    integrated_data.to_csv(output_file)

def main():
    # Load preprocessed omics data
    rna_seq_file = 'data/processed/processed_rna_seq.csv'
    proteomics_file = 'data/processed/processed_proteomics.csv'
    metabolomics_file = 'data/processed/processed_metabolomics.csv'

    rna_seq_data, proteomics_data, metabolomics_data = load_preprocessed_data(rna_seq_file, proteomics_file, metabolomics_file)

    # Integrate the omics data
    integrated_data = integrate_data(rna_seq_data, proteomics_data, metabolomics_data)

    # Save the integrated data
    save_integrated_data(integrated_data, 'data/processed/integrated_omics_data.csv')

if __name__ == '__main__':
    main()
