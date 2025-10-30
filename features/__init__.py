"""Feature extraction module for EEG-RSA analysis.

Provides functions for extracting various features from epoched EEG data.
"""

from .extract_features import (
    extract_erp_features,
    extract_spectral_features,
    extract_spatial_features,
    extract_time_frequency_features,
    extract_all_features
)

__all__ = [
    'extract_erp_features',
    'extract_spectral_features',
    'extract_spatial_features',
    'extract_time_frequency_features',
    'extract_all_features'
]
