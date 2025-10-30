"""RSA (Representational Similarity Analysis) module.

Provides functions for computing RDMs, creating model RDMs, and statistical testing.
"""

from .compute_rdm import (
    compute_rdm,
    compute_rdm_across_time,
    compute_rdm_crossvalidated,
    compute_condition_rdm,
    visualize_rdm,
    compare_rdms
)

from .model_rdms import (
    create_categorical_rdm,
    create_imagery_perception_model,
    create_hierarchical_rdm,
    create_continuous_rdm,
    create_null_rdm,
    create_semantic_similarity_rdm,
    combine_model_rdms
)

from .statistics import (
    rsa_correlation,
    permutation_test,
    bootstrap_confidence_interval,
    compare_models,
    compare_two_conditions,
    noise_ceiling,
    cross_validated_rsa
)

__all__ = [
    'compute_rdm', 'compute_rdm_across_time', 'compute_rdm_crossvalidated',
    'compute_condition_rdm', 'visualize_rdm', 'compare_rdms',
    'create_categorical_rdm', 'create_imagery_perception_model',
    'create_hierarchical_rdm', 'create_continuous_rdm', 'create_null_rdm',
    'create_semantic_similarity_rdm', 'combine_model_rdms',
    'rsa_correlation', 'permutation_test', 'bootstrap_confidence_interval',
    'compare_models', 'compare_two_conditions', 'noise_ceiling', 'cross_validated_rsa'
]
