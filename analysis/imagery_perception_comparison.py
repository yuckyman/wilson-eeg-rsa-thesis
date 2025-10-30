"""Compare imagery and perception representations using RSA.

Implements the core analysis pipeline following Wilson et al. methodology.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
import sys
sys.path.append('..')

from rsa.compute_rdm import compute_rdm, visualize_rdm, compare_rdms
from rsa.model_rdms import create_imagery_perception_model
from rsa.statistics import (
    rsa_correlation, permutation_test, 
    compare_two_conditions, compare_models
)


def load_epoched_data(data_path: str) -> Dict:
    """Load preprocessed epoched data.
    
    Parameters
    ----------
    data_path : str
        Path to saved epochs file
    
    Returns
    -------
    data_dict : dict
        Dictionary containing imagery and perception epochs
    """
    import mne
    
    # Load epochs
    imagery_epochs = mne.read_epochs(f"{data_path}/imagery_epochs-epo.fif")
    perception_epochs = mne.read_epochs(f"{data_path}/perception_epochs-epo.fif")
    
    data_dict = {
        'imagery': imagery_epochs,
        'perception': perception_epochs
    }
    
    return data_dict


def extract_condition_features(epochs: 'mne.Epochs', 
                              feature_type: str = 'erp') -> np.ndarray:
    """Extract features for each condition.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data
    feature_type : str
        Type of features ('erp', 'spectral', 'all')
    
    Returns
    -------
    features : ndarray
        Feature matrix (n_conditions, n_features)
    """
    from features.extract_features import extract_all_features
    
    # Extract features
    features_dict = extract_all_features(epochs, 
                                        include_erp=(feature_type in ['erp', 'all']),
                                        include_spectral=(feature_type in ['spectral', 'all']),
                                        normalize=True)
    
    return features_dict[feature_type]


def compute_condition_rdms(imagery_features: np.ndarray,
                          perception_features: np.ndarray,
                          condition_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute RDMs for imagery and perception conditions.
    
    Parameters
    ----------
    imagery_features : ndarray
        Feature matrix for imagery trials
    perception_features : ndarray
        Feature matrix for perception trials
    condition_labels : ndarray
        Stimulus category labels (e.g., 0=face, 1=house)
    
    Returns
    -------
    rdm_imagery : ndarray
        RDM for imagery condition
    rdm_perception : ndarray
        RDM for perception condition
    """
    # Average features within each stimulus category
    unique_labels = np.unique(condition_labels)
    n_conditions = len(unique_labels)
    
    imagery_cond = np.zeros((n_conditions, imagery_features.shape[1]))
    perception_cond = np.zeros((n_conditions, perception_features.shape[1]))
    
    for i, label in enumerate(unique_labels):
        mask = condition_labels == label
        imagery_cond[i] = np.mean(imagery_features[mask], axis=0)
        perception_cond[i] = np.mean(perception_features[mask], axis=0)
    
    # Compute RDMs
    rdm_imagery = compute_rdm(imagery_cond, metric='correlation')
    rdm_perception = compute_rdm(perception_cond, metric='correlation')
    
    return rdm_imagery, rdm_perception


def test_shared_representations(rdm_imagery: np.ndarray,
                               rdm_perception: np.ndarray,
                               n_permutations: int = 10000) -> Dict:
    """Test whether imagery and perception share representational structure.
    
    Parameters
    ----------
    rdm_imagery : ndarray
        RDM for imagery
    rdm_perception : ndarray
        RDM for perception
    n_permutations : int
        Number of permutations for significance testing
    
    Returns
    -------
    results : dict
        Statistical test results
    """
    print("\n=== Testing Shared Representations ===")
    print("H0: Imagery and perception have independent representations")
    print("H1: Imagery and perception share representational structure\n")
    
    # Direct correlation between imagery and perception RDMs
    corr, p_parametric = rsa_correlation(rdm_imagery, rdm_perception, method='spearman')
    print(f"Direct correlation: r = {corr:.3f}, p = {p_parametric:.4f}")
    
    # Permutation test for robustness
    _, p_perm, null_dist = permutation_test(rdm_imagery, rdm_perception,
                                           n_permutations=n_permutations)
    
    results = {
        'correlation': corr,
        'p_value_parametric': p_parametric,
        'p_value_permutation': p_perm,
        'null_distribution': null_dist
    }
    
    return results


def compare_with_models(rdm_imagery: np.ndarray,
                       rdm_perception: np.ndarray,
                       stimulus_categories: list) -> Dict:
    """Compare data RDMs with theoretical model RDMs.
    
    Parameters
    ----------
    rdm_imagery : ndarray
        RDM for imagery
    rdm_perception : ndarray
        RDM for perception
    stimulus_categories : list
        List of stimulus category names
    
    Returns
    -------
    results : dict
        Model comparison results
    """
    print("\n=== Model Comparison ===")
    
    # Create condition and stimulus type lists
    n_stim = len(stimulus_categories)
    condition_types = ['imagery'] * n_stim + ['perception'] * n_stim
    stimulus_types = stimulus_categories * 2
    
    # Generate model RDMs
    model_rdms = create_imagery_perception_model(condition_types, stimulus_types)
    
    # Compare imagery RDM with models
    print("\nImagery condition:")
    imagery_results = compare_models(rdm_imagery, model_rdms)
    
    # Compare perception RDM with models
    print("\nPerception condition:")
    perception_results = compare_models(rdm_perception, model_rdms)
    
    results = {
        'imagery': imagery_results,
        'perception': perception_results,
        'models': model_rdms
    }
    
    return results


def visualize_results(rdm_imagery: np.ndarray,
                     rdm_perception: np.ndarray,
                     stimulus_labels: list,
                     save_dir: Optional[str] = None) -> None:
    """Create visualization of RDM comparison results.
    
    Parameters
    ----------
    rdm_imagery : ndarray
        RDM for imagery
    rdm_perception : ndarray
        RDM for perception
    stimulus_labels : list
        Labels for each stimulus condition
    save_dir : str, optional
        Directory to save figures
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot imagery RDM
    sns.heatmap(rdm_imagery, annot=True, fmt='.2f', cmap='viridis',
               xticklabels=stimulus_labels, yticklabels=stimulus_labels,
               cbar_kws={'label': 'Dissimilarity'}, ax=axes[0], square=True)
    axes[0].set_title('Imagery RDM', fontsize=14, fontweight='bold')
    
    # Plot perception RDM
    sns.heatmap(rdm_perception, annot=True, fmt='.2f', cmap='viridis',
               xticklabels=stimulus_labels, yticklabels=stimulus_labels,
               cbar_kws={'label': 'Dissimilarity'}, ax=axes[1], square=True)
    axes[1].set_title('Perception RDM', fontsize=14, fontweight='bold')
    
    # Plot correlation scatter
    n = rdm_imagery.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    imagery_vec = rdm_imagery[triu_idx]
    perception_vec = rdm_perception[triu_idx]
    
    axes[2].scatter(imagery_vec, perception_vec, alpha=0.6, s=50)
    axes[2].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Identity line')
    axes[2].set_xlabel('Imagery Dissimilarity', fontsize=12)
    axes[2].set_ylabel('Perception Dissimilarity', fontsize=12)
    axes[2].set_title('RDM Correlation', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    # Add correlation value
    from scipy.stats import spearmanr
    corr, p = spearmanr(imagery_vec, perception_vec)
    axes[2].text(0.05, 0.95, f'r = {corr:.3f}\np = {p:.4f}',
                transform=axes[2].transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/rdm_comparison.png", dpi=300, bbox_inches='tight')
        print(f"\nSaved figure to {save_dir}/rdm_comparison.png")
    
    plt.show()


def run_full_analysis(imagery_epochs: 'mne.Epochs',
                     perception_epochs: 'mne.Epochs',
                     stimulus_labels: list,
                     feature_type: str = 'erp',
                     save_dir: Optional[str] = None) -> Dict:
    """Run complete imagery-perception RSA analysis pipeline.
    
    Parameters
    ----------
    imagery_epochs : mne.Epochs
        Epochs for imagery condition
    perception_epochs : mne.Epochs
        Epochs for perception condition
    stimulus_labels : list
        Labels for stimulus categories
    feature_type : str
        Type of features to extract
    save_dir : str, optional
        Directory to save results
    
    Returns
    -------
    results : dict
        Complete analysis results
    """
    print("="*60)
    print("EEG-RSA ANALYSIS: IMAGERY vs PERCEPTION")
    print("="*60)
    
    # Extract features
    print("\n1. Extracting features...")
    imagery_features = extract_condition_features(imagery_epochs, feature_type)
    perception_features = extract_condition_features(perception_epochs, feature_type)
    
    # Create condition labels
    n_trials = len(imagery_epochs)
    condition_labels = np.repeat(np.arange(len(stimulus_labels)), 
                                n_trials // len(stimulus_labels))
    
    # Compute RDMs
    print("\n2. Computing RDMs...")
    rdm_imagery, rdm_perception = compute_condition_rdms(
        imagery_features, perception_features, condition_labels
    )
    
    # Test shared representations
    print("\n3. Testing shared representations...")
    shared_rep_results = test_shared_representations(rdm_imagery, rdm_perception)
    
    # Model comparison
    print("\n4. Comparing with theoretical models...")
    model_results = compare_with_models(rdm_imagery, rdm_perception, stimulus_labels)
    
    # Visualize results
    print("\n5. Creating visualizations...")
    visualize_results(rdm_imagery, rdm_perception, stimulus_labels, save_dir)
    
    # Compile results
    results = {
        'rdm_imagery': rdm_imagery,
        'rdm_perception': rdm_perception,
        'shared_representations': shared_rep_results,
        'model_comparison': model_results,
        'stimulus_labels': stimulus_labels
    }
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    return results


if __name__ == "__main__":
    # Example usage
    pass
