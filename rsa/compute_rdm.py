"""Compute Representational Dissimilarity Matrices (RDMs) for RSA.

Implements RDM computation using various distance metrics.
"""

import numpy as np
from typing import Optional, Union, List
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


def compute_rdm(features: np.ndarray,
               metric: str = 'correlation',
               normalize: bool = True) -> np.ndarray:
    """Compute Representational Dissimilarity Matrix from feature vectors.
    
    Parameters
    ----------
    features : ndarray
        Feature matrix (n_conditions, n_features)
        Each row represents features for one condition/stimulus
    metric : str
        Distance metric:
        - 'correlation': 1 - Pearson correlation
        - 'euclidean': Euclidean distance
        - 'cosine': Cosine distance
        - 'mahalanobis': Mahalanobis distance
    normalize : bool
        Whether to z-score normalize features before computing distances
    
    Returns
    -------
    rdm : ndarray
        Representational Dissimilarity Matrix (n_conditions, n_conditions)
        Symmetric matrix with dissimilarities between all condition pairs
    """
    if normalize:
        features = (features - np.mean(features, axis=1, keepdims=True)) / \
                   (np.std(features, axis=1, keepdims=True) + 1e-10)
    
    # Compute pairwise distances
    distances = pdist(features, metric=metric)
    
    # Convert to square form
    rdm = squareform(distances)
    
    print(f"Computed RDM with shape {rdm.shape} using {metric} distance")
    return rdm


def compute_rdm_across_time(epochs_data: np.ndarray,
                           time_points: np.ndarray,
                           metric: str = 'correlation',
                           sliding_window: Optional[float] = None) -> np.ndarray:
    """Compute time-resolved RDMs.
    
    Parameters
    ----------
    epochs_data : ndarray
        Epoched data (n_epochs, n_channels, n_times)
    time_points : ndarray
        Time points corresponding to each sample
    metric : str
        Distance metric for RDM computation
    sliding_window : float, optional
        Window size in seconds. If None, compute at each time point
    
    Returns
    -------
    rdms : ndarray
        Time-resolved RDMs (n_times, n_epochs, n_epochs)
    """
    n_epochs, n_channels, n_times = epochs_data.shape
    rdms = np.zeros((n_times, n_epochs, n_epochs))
    
    for t in range(n_times):
        # Get features at this time point (all channels)
        features = epochs_data[:, :, t]  # (n_epochs, n_channels)
        rdms[t] = compute_rdm(features, metric=metric, normalize=True)
    
    print(f"Computed {n_times} time-resolved RDMs")
    return rdms


def compute_rdm_crossvalidated(features: np.ndarray,
                              labels: np.ndarray,
                              n_folds: int = 5,
                              metric: str = 'correlation') -> np.ndarray:
    """Compute cross-validated RDM to reduce noise.
    
    Parameters
    ----------
    features : ndarray
        Feature matrix (n_trials, n_features)
    labels : ndarray
        Condition labels for each trial (n_trials,)
    n_folds : int
        Number of cross-validation folds
    metric : str
        Distance metric
    
    Returns
    -------
    rdm : ndarray
        Cross-validated RDM (n_conditions, n_conditions)
    """
    from sklearn.model_selection import KFold
    
    unique_labels = np.unique(labels)
    n_conditions = len(unique_labels)
    
    # Initialize RDM accumulator
    rdm_sum = np.zeros((n_conditions, n_conditions))
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for train_idx, test_idx in kf.split(features):
        # Compute condition averages in test set
        test_features = features[test_idx]
        test_labels = labels[test_idx]
        
        condition_features = np.zeros((n_conditions, features.shape[1]))
        for i, label in enumerate(unique_labels):
            condition_mask = test_labels == label
            if np.any(condition_mask):
                condition_features[i] = np.mean(test_features[condition_mask], axis=0)
        
        # Compute RDM for this fold
        rdm_fold = compute_rdm(condition_features, metric=metric, normalize=True)
        rdm_sum += rdm_fold
    
    # Average across folds
    rdm = rdm_sum / n_folds
    
    print(f"Computed cross-validated RDM with {n_folds} folds")
    return rdm


def compute_condition_rdm(epochs_data: np.ndarray,
                         condition_labels: np.ndarray,
                         metric: str = 'correlation') -> np.ndarray:
    """Compute RDM from epoched data with condition labels.
    
    Parameters
    ----------
    epochs_data : ndarray
        Epoched data (n_epochs, n_channels, n_times)
    condition_labels : ndarray
        Condition label for each epoch (n_epochs,)
    metric : str
        Distance metric
    
    Returns
    -------
    rdm : ndarray
        RDM between conditions (n_conditions, n_conditions)
    """
    # Average data across time to get trial-level features
    features = np.mean(epochs_data, axis=2)  # (n_epochs, n_channels)
    
    # Average trials within each condition
    unique_conditions = np.unique(condition_labels)
    n_conditions = len(unique_conditions)
    condition_features = np.zeros((n_conditions, features.shape[1]))
    
    for i, condition in enumerate(unique_conditions):
        condition_mask = condition_labels == condition
        condition_features[i] = np.mean(features[condition_mask], axis=0)
    
    # Compute RDM
    rdm = compute_rdm(condition_features, metric=metric, normalize=True)
    
    return rdm


def visualize_rdm(rdm: np.ndarray,
                 labels: Optional[List[str]] = None,
                 title: str = "Representational Dissimilarity Matrix",
                 cmap: str = "viridis",
                 save_path: Optional[str] = None) -> None:
    """Visualize RDM as a heatmap.
    
    Parameters
    ----------
    rdm : ndarray
        Representational Dissimilarity Matrix (n_conditions, n_conditions)
    labels : list, optional
        Labels for each condition
    title : str
        Plot title
    cmap : str
        Colormap name
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(rdm, annot=False, cmap=cmap, square=True,
               xticklabels=labels, yticklabels=labels,
               cbar_kws={'label': 'Dissimilarity'}, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved RDM visualization to {save_path}")
    
    plt.show()


def compare_rdms(rdm1: np.ndarray,
                rdm2: np.ndarray,
                method: str = 'spearman') -> float:
    """Compute similarity between two RDMs.
    
    Parameters
    ----------
    rdm1, rdm2 : ndarray
        Two RDMs to compare (must be same shape)
    method : str
        Correlation method ('pearson' or 'spearman')
    
    Returns
    -------
    similarity : float
        Correlation coefficient between RDMs
    """
    # Get upper triangular indices (excluding diagonal)
    n = rdm1.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    
    # Extract upper triangular values
    rdm1_vec = rdm1[triu_idx]
    rdm2_vec = rdm2[triu_idx]
    
    # Compute correlation
    if method == 'pearson':
        similarity, _ = pearsonr(rdm1_vec, rdm2_vec)
    else:  # spearman
        similarity, _ = spearmanr(rdm1_vec, rdm2_vec)
    
    return similarity


if __name__ == "__main__":
    # Example usage
    pass
