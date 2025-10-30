"""Generate model RDMs for hypothesis testing in RSA.

Create theoretical RDMs based on different hypotheses about
representational structure.
"""

import numpy as np
from typing import List, Dict, Optional
import itertools


def create_categorical_rdm(categories: List[str],
                          within_distance: float = 0.0,
                          between_distance: float = 1.0) -> np.ndarray:
    """Create RDM based on categorical structure.
    
    Parameters
    ----------
    categories : list
        Category label for each condition
    within_distance : float
        Dissimilarity within same category
    between_distance : float
        Dissimilarity between different categories
    
    Returns
    -------
    rdm : ndarray
        Categorical model RDM
    """
    n = len(categories)
    rdm = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                rdm[i, j] = 0.0  # Diagonal is always 0
            elif categories[i] == categories[j]:
                rdm[i, j] = within_distance
            else:
                rdm[i, j] = between_distance
    
    return rdm


def create_imagery_perception_model(condition_types: List[str],
                                   stimulus_types: List[str]) -> Dict[str, np.ndarray]:
    """Create model RDMs for imagery-perception comparison (Wilson et al.).
    
    Parameters
    ----------
    condition_types : list
        List indicating 'imagery' or 'perception' for each condition
    stimulus_types : list
        List indicating stimulus category (e.g., 'face', 'house') for each condition
    
    Returns
    -------
    model_rdms : dict
        Dictionary with different model RDMs:
        - 'stimulus_specific': Same stimulus, different conditions are similar
        - 'modality_specific': Same modality (imagery/perception) are similar
        - 'stimulus_only': Only stimulus category matters
        - 'independent': Imagery and perception are completely independent
    """
    n = len(condition_types)
    model_rdms = {}
    
    # Model 1: Stimulus-specific (shared representations)
    # Same stimulus type is similar regardless of imagery/perception
    rdm_stimulus = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                if stimulus_types[i] == stimulus_types[j]:
                    rdm_stimulus[i, j] = 0.3  # Similar
                else:
                    rdm_stimulus[i, j] = 1.0  # Dissimilar
    model_rdms['stimulus_specific'] = rdm_stimulus
    
    # Model 2: Modality-specific (separate representations)
    # Same modality (imagery or perception) is similar
    rdm_modality = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                if condition_types[i] == condition_types[j]:
                    if stimulus_types[i] == stimulus_types[j]:
                        rdm_modality[i, j] = 0.2  # Very similar (same modality and stimulus)
                    else:
                        rdm_modality[i, j] = 0.5  # Somewhat similar (same modality)
                else:
                    rdm_modality[i, j] = 1.0  # Dissimilar (different modality)
    model_rdms['modality_specific'] = rdm_modality
    
    # Model 3: Stimulus-only
    # Only stimulus category matters, modality is irrelevant
    rdm_stimulus_only = create_categorical_rdm(stimulus_types, 
                                               within_distance=0.0,
                                               between_distance=1.0)
    model_rdms['stimulus_only'] = rdm_stimulus_only
    
    # Model 4: Independent representations
    # Imagery and perception are completely separate
    rdm_independent = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                if condition_types[i] != condition_types[j]:
                    rdm_independent[i, j] = 1.0  # Maximally dissimilar
                else:
                    if stimulus_types[i] == stimulus_types[j]:
                        rdm_independent[i, j] = 0.3
                    else:
                        rdm_independent[i, j] = 0.7
    model_rdms['independent'] = rdm_independent
    
    return model_rdms


def create_hierarchical_rdm(hierarchy: List[List[str]]) -> np.ndarray:
    """Create RDM based on hierarchical category structure.
    
    Parameters
    ----------
    hierarchy : list of lists
        Nested list representing hierarchical structure
        e.g., [['face_1', 'face_2'], ['house_1', 'house_2']]
    
    Returns
    -------
    rdm : ndarray
        Hierarchical model RDM
    """
    # Flatten hierarchy to get all items
    all_items = [item for group in hierarchy for item in group]
    n = len(all_items)
    rdm = np.zeros((n, n))
    
    # Create mapping from item to group
    item_to_group = {}
    for group_idx, group in enumerate(hierarchy):
        for item in group:
            item_to_group[item] = group_idx
    
    # Fill RDM
    for i, item_i in enumerate(all_items):
        for j, item_j in enumerate(all_items):
            if i != j:
                if item_to_group[item_i] == item_to_group[item_j]:
                    rdm[i, j] = 0.3  # Same category
                else:
                    rdm[i, j] = 1.0  # Different category
    
    return rdm


def create_continuous_rdm(values: np.ndarray) -> np.ndarray:
    """Create RDM based on continuous parameter values.
    
    Parameters
    ----------
    values : ndarray
        Continuous values for each condition (e.g., stimulus size, contrast)
    
    Returns
    -------
    rdm : ndarray
        RDM where dissimilarity is proportional to difference in values
    """
    n = len(values)
    rdm = np.zeros((n, n))
    
    # Normalize values to [0, 1]
    values_norm = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-10)
    
    for i in range(n):
        for j in range(n):
            rdm[i, j] = np.abs(values_norm[i] - values_norm[j])
    
    return rdm


def create_null_rdm(n_conditions: int) -> np.ndarray:
    """Create null model RDM (all conditions equally dissimilar).
    
    Parameters
    ----------
    n_conditions : int
        Number of conditions
    
    Returns
    -------
    rdm : ndarray
        Null model RDM
    """
    rdm = np.ones((n_conditions, n_conditions))
    np.fill_diagonal(rdm, 0)
    return rdm


def create_semantic_similarity_rdm(stimuli: List[str],
                                  similarity_matrix: Optional[np.ndarray] = None) -> np.ndarray:
    """Create RDM based on semantic similarity.
    
    Parameters
    ----------
    stimuli : list
        List of stimulus names/descriptions
    similarity_matrix : ndarray, optional
        Pre-computed similarity matrix. If None, uses simple word overlap
    
    Returns
    -------
    rdm : ndarray
        Semantic similarity RDM
    """
    n = len(stimuli)
    
    if similarity_matrix is not None:
        # Convert similarity to dissimilarity
        rdm = 1 - similarity_matrix
    else:
        # Simple word overlap metric
        rdm = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    words_i = set(stimuli[i].lower().split())
                    words_j = set(stimuli[j].lower().split())
                    overlap = len(words_i & words_j) / max(len(words_i | words_j), 1)
                    rdm[i, j] = 1 - overlap
    
    return rdm


def combine_model_rdms(rdm_dict: Dict[str, np.ndarray],
                      weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Combine multiple model RDMs with weights.
    
    Parameters
    ----------
    rdm_dict : dict
        Dictionary of model RDMs
    weights : dict, optional
        Weights for each model. If None, uses equal weights
    
    Returns
    -------
    combined_rdm : ndarray
        Weighted combination of model RDMs
    """
    if weights is None:
        weights = {name: 1.0 / len(rdm_dict) for name in rdm_dict.keys()}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    # Combine RDMs
    combined_rdm = np.zeros_like(next(iter(rdm_dict.values())))
    for name, rdm in rdm_dict.items():
        combined_rdm += weights[name] * rdm
    
    return combined_rdm


if __name__ == "__main__":
    # Example: Create model RDMs for imagery-perception study
    condition_types = ['imagery', 'imagery', 'perception', 'perception']
    stimulus_types = ['face', 'house', 'face', 'house']
    
    models = create_imagery_perception_model(condition_types, stimulus_types)
    
    for model_name, rdm in models.items():
        print(f"\n{model_name} model:")
        print(rdm)
