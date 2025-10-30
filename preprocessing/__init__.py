"""Preprocessing module for EEG data.

Provides functions for loading, filtering, artifact removal, and epoching.
"""

from .load_data import load_raw_eeg, load_events, load_subject_data
from .filter_eeg import apply_bandpass_filter, apply_notch_filter, apply_reference, preprocess_raw
from .artifact_removal import (
    detect_bad_channels, interpolate_bad_channels,
    run_ica, detect_artifact_components, apply_ica_cleaning,
    remove_artifacts_pipeline
)
from .epoching import (
    create_epochs, reject_bad_epochs, equalize_epoch_counts,
    create_condition_epochs, extract_imagery_perception_epochs
)

__all__ = [
    'load_raw_eeg', 'load_events', 'load_subject_data',
    'apply_bandpass_filter', 'apply_notch_filter', 'apply_reference', 'preprocess_raw',
    'detect_bad_channels', 'interpolate_bad_channels',
    'run_ica', 'detect_artifact_components', 'apply_ica_cleaning', 'remove_artifacts_pipeline',
    'create_epochs', 'reject_bad_epochs', 'equalize_epoch_counts',
    'create_condition_epochs', 'extract_imagery_perception_epochs'
]
