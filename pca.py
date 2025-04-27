'''
Replicating regular PCA/UMAP encoding

'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap

def load_data(expr_path, meta_path, n_genes=1000):
    """Load and prepare expression matrix and metadata."""
    # Load expression data
    expr = pd.read_csv(expr_path, nrows=n_genes)
    expr = expr.set_index('Unnamed: 0')
    expr = expr.T  # Now rows = cells, columns = genes

    # Load metadata
    meta = pd.read_csv(meta_path)
    meta = meta.set_index('transcriptomics_sample_id')
    
    # Align metadata to expression cells
    meta = meta.loc[expr.index]

    return expr, meta

def compute_umap(expr, n_pcs=50, n_neighbors=15, min_dist=0.1):
    """Standardize, PCA, and UMAP on expression matrix."""
    # Standardize
    scaler = StandardScaler()
    expr_scaled = scaler.fit_transform(expr)
    
    # PCA
    pca = PCA(n_components=n_pcs)
    pca_result = pca.fit_transform(expr_scaled)

    # UMAP
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    umap_result = umap_model.fit_transform(pca_result)

    return umap_result

def plot_umap(umap_result, labels, title='UMAP of Cells by Cell Type'):
    """Plot UMAP embedding colored by labels."""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1],
                           c=pd.factorize(labels)[0],
                           cmap='tab20', s=10, alpha=0.8)
    plt.title(title, fontsize=14)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.show()

def main():
    # Set your file paths here
    expr_path = 'path_to/20200513_Mouse_PatchSeq_Release_cpm.v2.csv'
    meta_path = 'path_to/20200625_patchseq_metadata_mouse.csv'

    # Load data
    expr, meta = load_data(expr_path, meta_path)

    # Compute UMAP
    umap_result = compute_umap(expr)

    # Plot
    plot_umap(umap_result, meta['corresponding_AIT2.3.1_alias'])

if __name__ == '__main__':
    main()
