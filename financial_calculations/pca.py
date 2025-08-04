import numpy as np

def PCA(sample_rates, use_diff: bool = False, n_components: int = 3):
    """
    Performs a Principal Component Analysis (PCA) manually using linear algebra on a matrix of interest rates.

    Parameters:
    - sample_rates (np.ndarray): 2D array of shape (n_samples, n_features), where each column is a rate series.
    - use_diff (bool): If True, use first differences instead of levels.
    - n_components (int): Number of principal components to return.

    Returns:
    - loadings (np.ndarray): Array of shape (n_components, n_features), each row is a principal component.
    - explained_variance (np.ndarray): Array of shape (n_components,), variance ratio of each PC.
    - scores (np.ndarray): Array of shape (n_samples, n_components), projection of sample_rates onto PCs.
    """
    if use_diff:
        sample_rates = np.diff(sample_rates, axis=0)

    # Center the data
    centered_rates = sample_rates - np.mean(sample_rates, axis=0)

    # Covariance matrix (features x features)
    cov_matrix = np.cov(centered_rates, rowvar=False)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # symmetric matrix -> eigh is more stable

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select top components
    selected_vectors = eigenvectors[:, :n_components].T         # shape (n_components, n_features)
    selected_values = eigenvalues[:n_components]

    # Explained variance ratio
    explained_variance = selected_values / np.sum(eigenvalues)

    # Project data onto components
    scores = centered_rates @ selected_vectors.T  # shape (n_samples, n_components)

    return selected_vectors, explained_variance, scores
