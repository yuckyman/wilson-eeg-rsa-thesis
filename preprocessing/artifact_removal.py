"""Artifact detection and removal for EEG data.

Implements ICA-based artifact removal and automated bad channel detection.
"""

import mne
import numpy as np
from typing import Optional, List
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs


def detect_bad_channels(raw: mne.io.Raw,
                       threshold: float = 3.0) -> List[str]:
    """Detect bad channels based on signal statistics.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data object
    threshold : float
        Z-score threshold for marking channels as bad
    
    Returns
    -------
    bad_channels : list
        List of bad channel names
    """
    picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude=[])
    data = raw.get_data(picks=picks)
    
    # Compute statistics
    variance = np.var(data, axis=1)
    kurtosis = np.apply_along_axis(lambda x: np.mean((x - np.mean(x))**4) / np.std(x)**4, 
                                   axis=1, arr=data)
    
    # Z-score normalization
    variance_z = np.abs((variance - np.median(variance)) / np.std(variance))
    kurtosis_z = np.abs((kurtosis - np.median(kurtosis)) / np.std(kurtosis))
    
    # Find bad channels
    bad_idx = np.where((variance_z > threshold) | (kurtosis_z > threshold))[0]
    bad_channels = [raw.ch_names[picks[i]] for i in bad_idx]
    
    print(f"Detected {len(bad_channels)} bad channels: {bad_channels}")
    return bad_channels


def interpolate_bad_channels(raw: mne.io.Raw,
                            bad_channels: Optional[List[str]] = None) -> mne.io.Raw:
    """Interpolate bad channels.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data object
    bad_channels : list, optional
        List of bad channel names. If None, uses raw.info['bads']
    
    Returns
    -------
    raw_interp : mne.io.Raw
        EEG data with interpolated channels
    """
    raw_interp = raw.copy()
    
    if bad_channels is not None:
        raw_interp.info['bads'] = bad_channels
    
    if raw_interp.info['bads']:
        raw_interp.interpolate_bads(reset_bads=True)
        print(f"Interpolated {len(raw_interp.info['bads'])} bad channels")
    else:
        print("No bad channels to interpolate")
    
    return raw_interp


def run_ica(raw: mne.io.Raw,
           n_components: Optional[int] = None,
           method: str = 'fastica',
           random_state: int = 42) -> ICA:
    """Run Independent Component Analysis on EEG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data object
    n_components : int, optional
        Number of ICA components. If None, uses min(n_channels, n_timepoints)
    method : str
        ICA method ('fastica', 'infomax', 'picard')
    random_state : int
        Random seed for reproducibility
    
    Returns
    -------
    ica : ICA
        Fitted ICA object
    """
    if n_components is None:
        n_components = min(len(raw.ch_names), 25)  # Typical default
    
    ica = ICA(n_components=n_components, method=method, 
              random_state=random_state, max_iter=500)
    
    # Fit ICA
    picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude='bads')
    ica.fit(raw, picks=picks, verbose=True)
    
    print(f"ICA fitted with {n_components} components")
    return ica


def detect_artifact_components(ica: ICA,
                              raw: mne.io.Raw,
                              eog_channels: Optional[List[str]] = None,
                              ecg_channels: Optional[List[str]] = None,
                              threshold: float = 0.3) -> dict:
    """Automatically detect artifact components (EOG, ECG).
    
    Parameters
    ----------
    ica : ICA
        Fitted ICA object
    raw : mne.io.Raw
        Raw EEG data object
    eog_channels : list, optional
        EOG channel names
    ecg_channels : list, optional
        ECG channel names
    threshold : float
        Correlation threshold for artifact detection
    
    Returns
    -------
    artifact_components : dict
        Dictionary with 'eog' and 'ecg' component indices
    """
    artifact_components = {'eog': [], 'ecg': []}
    
    # Detect EOG artifacts
    if eog_channels or 'eog' in raw:
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_channels,
                                                     threshold=threshold)
        artifact_components['eog'] = eog_indices
        print(f"Found {len(eog_indices)} EOG components: {eog_indices}")
    
    # Detect ECG artifacts
    if ecg_channels or 'ecg' in raw:
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name=ecg_channels,
                                                     threshold=threshold)
        artifact_components['ecg'] = ecg_indices
        print(f"Found {len(ecg_indices)} ECG components: {ecg_indices}")
    
    return artifact_components


def apply_ica_cleaning(raw: mne.io.Raw,
                      ica: ICA,
                      exclude_components: List[int]) -> mne.io.Raw:
    """Apply ICA to remove artifact components from data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data object
    ica : ICA
        Fitted ICA object
    exclude_components : list
        List of component indices to remove
    
    Returns
    -------
    raw_clean : mne.io.Raw
        Cleaned EEG data
    """
    raw_clean = raw.copy()
    ica.exclude = exclude_components
    ica.apply(raw_clean)
    
    print(f"Removed {len(exclude_components)} ICA components: {exclude_components}")
    return raw_clean


def remove_artifacts_pipeline(raw: mne.io.Raw,
                             detect_bad_chans: bool = True,
                             run_ica_cleaning: bool = True,
                             n_components: Optional[int] = None) -> tuple:
    """Complete artifact removal pipeline.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data object
    detect_bad_chans : bool
        Whether to detect and interpolate bad channels
    run_ica_cleaning : bool
        Whether to run ICA-based artifact removal
    n_components : int, optional
        Number of ICA components
    
    Returns
    -------
    raw_clean : mne.io.Raw
        Cleaned EEG data
    ica : ICA or None
        Fitted ICA object if run_ica_cleaning=True
    """
    print("Starting artifact removal pipeline...")
    raw_clean = raw.copy()
    ica = None
    
    # Detect and interpolate bad channels
    if detect_bad_chans:
        bad_channels = detect_bad_channels(raw_clean)
        raw_clean = interpolate_bad_channels(raw_clean, bad_channels)
    
    # ICA-based artifact removal
    if run_ica_cleaning:
        ica = run_ica(raw_clean, n_components=n_components)
        artifacts = detect_artifact_components(ica, raw_clean)
        
        # Combine all artifact components
        exclude = list(set(artifacts['eog'] + artifacts['ecg']))
        raw_clean = apply_ica_cleaning(raw_clean, ica, exclude)
    
    print("Artifact removal complete")
    return raw_clean, ica


if __name__ == "__main__":
    # Example usage
    pass
