"""Statistical tests for RSA analysis.

Implements permutation tests, bootstrap confidence intervals,
and model comparison for representational similarity analysis.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from scipy.stats import spearmanr, pearsonr, ttest_rel, ttest_1samp
from scipy.spatial.distance import squareform
import warnings


def rsa_correlation(rdm1: np.ndarray,
                   rdm2: np.ndarray,
                   method: str = 'spearman') -> Tuple[float, float]:
    """Compute correlation between two RDMs.
    
    Parameters
    ----------
    rdm1, rdm2 : ndarray
        Two RDMs to correlate
    method : str
        Correlation method ('spearman' or 'pearson')
    
    Returns
    -------
    correlation : float
        Correlation coefficient
    p_value : float
        Two-tailed p-value
    """
    # Extract upper triangular values (excluding diagonal)
    n = rdm1.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    
    rdm1_vec = rdm1[triu_idx]
    rdm2_vec = rdm2[triu_idx]
    
    # Compute correlation
    if method == 'spearman':
        correlation, p_value = spearmanr(rdm1_vec, rdm2_vec)
    else:  # pearson
        correlation, p_value = pearsonr(rdm1_vec, rdm2_vec)
    
    return correlation, p_value


def permutation_test(rdm_data: np.ndarray,
                    rdm_model: np.ndarray,
                    n_permutations: int = 10000,
                    method: str = 'spearman',
                    random_state: Optional[int] = None) -> Tuple[float, float, np.ndarray]:
    """Permutation test for RDM correlation significance.
    
    Parameters
    ----------
    rdm_data : ndarray
        Data RDM
    rdm_model : ndarray
        Model RDM
    n_permutations : int
        Number of permutations
    method : str
        Correlation method
    random_state : int, optional
        Random seed
    
    Returns
    -------
    observed_corr : float
        Observed correlation
    p_value : float
        Permutation-based p-value
    null_distribution : ndarray
        Null distribution of correlations
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Compute observed correlation
    observed_corr, _ = rsa_correlation(rdm_data, rdm_model, method=method)
    
    # Get upper triangular indices
    n = rdm_data.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    data_vec = rdm_data[triu_idx]
    model_vec = rdm_model[triu_idx]
    
    # Generate null distribution
    null_distribution = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        # Permute condition labels
        perm_idx = np.random.permutation(n)
        rdm_perm = rdm_data[perm_idx][:, perm_idx]
        perm_vec = rdm_perm[triu_idx]
        
        # Compute correlation for permuted data
        if method == 'spearman':
            null_distribution[i], _ = spearmanr(perm_vec, model_vec)
        else:
            null_distribution[i], _ = pearsonr(perm_vec, model_vec)
    
    # Compute p-value (two-tailed)
    p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_corr))
    
    print(f"Permutation test: r = {observed_corr:.3f}, p = {p_value:.4f}")
    
    return observed_corr, p_value, null_distribution


def bootstrap_confidence_interval(rdm_data: np.ndarray,
                                 rdm_model: np.ndarray,
                                 n_bootstrap: int = 10000,
                                 confidence_level: float = 0.95,
                                 method: str = 'spearman',
                                 random_state: Optional[int] = None) -> Tuple[float, Tuple[float, float]]:
    """Bootstrap confidence interval for RDM correlation.
    
    Parameters
    ----------
    rdm_data : ndarray
        Data RDM
    rdm_model : ndarray
        Model RDM
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    method : str
        Correlation method
    random_state : int, optional
        Random seed
    
    Returns
    -------
    correlation : float
        Observed correlation
    ci : tuple
        (lower_bound, upper_bound) of confidence interval
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Compute observed correlation
    correlation, _ = rsa_correlation(rdm_data, rdm_model, method=method)
    
    n = rdm_data.shape[0]
    bootstrap_corrs = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Bootstrap sample conditions
        bootstrap_idx = np.random.choice(n, size=n, replace=True)
        rdm_boot = rdm_data[bootstrap_idx][:, bootstrap_idx]
        model_boot = rdm_model[bootstrap_idx][:, bootstrap_idx]
        
        bootstrap_corrs[i], _ = rsa_correlation(rdm_boot, model_boot, method=method)
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_corrs, 100 * alpha / 2)
    upper = np.percentile(bootstrap_corrs, 100 * (1 - alpha / 2))
    
    print(f"Bootstrap CI ({confidence_level*100:.0f}%): [{lower:.3f}, {upper:.3f}]")
    
    return correlation, (lower, upper)


def compare_models(rdm_data: np.ndarray,
                  model_rdms: Dict[str, np.ndarray],
                  method: str = 'spearman') -> Dict[str, Tuple[float, float]]:
    """Compare multiple model RDMs against data RDM.
    
    Parameters
    ----------
    rdm_data : ndarray
        Data RDM
    model_rdms : dict
        Dictionary of model RDMs
    method : str
        Correlation method
    
    Returns
    -------
    results : dict
        Dictionary with (correlation, p_value) for each model
    """
    results = {}
    
    for model_name, rdm_model in model_rdms.items():
        corr, p_val = rsa_correlation(rdm_data, rdm_model, method=method)
        results[model_name] = (corr, p_val)
        print(f"{model_name}: r = {corr:.3f}, p = {p_val:.4f}")
    
    return results


def compare_two_conditions(rdm1: np.ndarray,
                          rdm2: np.ndarray,
                          rdm_model: np.ndarray,
                          method: str = 'spearman') -> Tuple[float, float, float]:
    """Compare model fit for two conditions (e.g., imagery vs perception).
    
    Parameters
    ----------
    rdm1, rdm2 : ndarray
        RDMs for two conditions
    rdm_model : ndarray
        Model RDM
    method : str
        Correlation method
    
    Returns
    -------
    corr1 : float
        Correlation for condition 1
    corr2 : float
        Correlation for condition 2
    p_value : float
        P-value for difference (paired t-test)
    """
    # Compute correlations
    corr1, _ = rsa_correlation(rdm1, rdm_model, method=method)
    corr2, _ = rsa_correlation(rdm2, rdm_model, method=method)
    
    # Convert to Fisher z for statistical comparison
    z1 = np.arctanh(corr1)
    z2 = np.arctanh(corr2)
    
    # Test difference
    from scipy.stats import norm
    n1 = rdm1.shape[0]
    se_diff = np.sqrt(1/(n1-3) + 1/(n1-3))
    z_stat = (z1 - z2) / se_diff
    p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))
    
    print(f"Condition 1: r = {corr1:.3f}")
    print(f"Condition 2: r = {corr2:.3f}")
    print(f"Difference: p = {p_value:.4f}")
    
    return corr1, corr2, p_value


def noise_ceiling(rdm_subjects: List[np.ndarray],
                 method: str = 'spearman') -> Tuple[float, float]:
    """Estimate noise ceiling for RSA analysis.
    
    Parameters
    ----------
    rdm_subjects : list of ndarray
        List of RDMs, one per subject
    method : str
        Correlation method
    
    Returns
    -------
    lower_bound : float
        Lower bound of noise ceiling
    upper_bound : float
        Upper bound of noise ceiling
    """
    n_subjects = len(rdm_subjects)
    
    # Compute mean RDM across subjects
    rdm_mean = np.mean(rdm_subjects, axis=0)
    
    # Upper bound: correlation with mean of all subjects
    upper_correlations = []
    for rdm in rdm_subjects:
        corr, _ = rsa_correlation(rdm, rdm_mean, method=method)
        upper_correlations.append(corr)
    upper_bound = np.mean(upper_correlations)
    
    # Lower bound: average leave-one-out correlation
    lower_correlations = []
    for i in range(n_subjects):
        # Mean of all subjects except i
        rdm_loo = np.mean([rdm_subjects[j] for j in range(n_subjects) if j != i], axis=0)
        corr, _ = rsa_correlation(rdm_subjects[i], rdm_loo, method=method)
        lower_correlations.append(corr)
    lower_bound = np.mean(lower_correlations)
    
    print(f"Noise ceiling: [{lower_bound:.3f}, {upper_bound:.3f}]")
    
    return lower_bound, upper_bound


def cross_validated_rsa(features: np.ndarray,
                       rdm_model: np.ndarray,
                       labels: np.ndarray,
                       n_folds: int = 5,
                       method: str = 'spearman') -> Tuple[float, float]:
    """Cross-validated RSA analysis.
    
    Parameters
    ----------
    features : ndarray
        Feature matrix (n_trials, n_features)
    rdm_model : ndarray
        Model RDM (n_conditions, n_conditions)
    labels : ndarray
        Condition labels (n_trials,)
    n_folds : int
        Number of cross-validation folds
    method : str
        Correlation method
    
    Returns
    -------
    mean_correlation : float
        Mean correlation across folds
    std_correlation : float
        Standard deviation of correlations
    """
    from sklearn.model_selection import KFold
    from rsa.compute_rdm import compute_rdm
    
    correlations = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for train_idx, test_idx in kf.split(features):
        # Use test set to compute RDM
        test_features = features[test_idx]
        test_labels = labels[test_idx]
        
        # Compute condition averages
        unique_labels = np.unique(test_labels)
        condition_features = []
        for label in unique_labels:
            mask = test_labels == label
            if np.any(mask):
                condition_features.append(np.mean(test_features[mask], axis=0))
        
        condition_features = np.array(condition_features)
        
        # Compute data RDM
        rdm_data = compute_rdm(condition_features, metric='correlation')
        
        # Correlate with model
        corr, _ = rsa_correlation(rdm_data, rdm_model, method=method)
        correlations.append(corr)
    
    mean_corr = np.mean(correlations)
    std_corr = np.std(correlations)
    
    print(f"Cross-validated RSA: r = {mean_corr:.3f} ± {std_corr:.3f}")
    
    return mean_corr, std_corr


if __name__ == "__main__":
    # Example usage
    pass
