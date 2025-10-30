"""Feature extraction for EEG-RSA analysis.

Extract ERP (event-related potential), spectral, and spatial features
from epoched EEG data for representational similarity analysis.
"""

import mne
import numpy as np
from typing import Optional, Dict, List, Tuple
from scipy import signal
from scipy.stats import zscore


def extract_erp_features(epochs: mne.Epochs,
                        time_windows: Optional[List[Tuple[float, float]]] = None,
                        channels: Optional[List[str]] = None) -> np.ndarray:
    """Extract ERP amplitude features from epochs.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    time_windows : list of tuples, optional
        List of (tmin, tmax) time windows to extract (seconds)
        If None, uses common ERP windows: [(0.08, 0.12), (0.17, 0.20), (0.30, 0.50)]
    channels : list, optional
        Channel names to include. If None, uses all EEG channels
    
    Returns
    -------
    erp_features : ndarray
        Shape (n_epochs, n_features)
        Features are concatenated: [window1_ch1, window1_ch2, ..., window2_ch1, ...]
    """
    if time_windows is None:
        # Default windows: P1, N170, P300
        time_windows = [(0.08, 0.12), (0.15, 0.20), (0.30, 0.50)]
    
    if channels is None:
        picks = mne.pick_types(epochs.info, meg=False, eeg=True)
        channels = [epochs.ch_names[i] for i in picks]
    
    features_list = []
    
    for tmin, tmax in time_windows:
        # Get data in time window
        epoch_data = epochs.copy().crop(tmin=tmin, tmax=tmax)
        data = epoch_data.get_data(picks=channels)  # (n_epochs, n_channels, n_times)
        
        # Average over time
        mean_amplitude = np.mean(data, axis=2)  # (n_epochs, n_channels)
        features_list.append(mean_amplitude)
    
    # Concatenate all features
    erp_features = np.concatenate(features_list, axis=1)
    
    print(f"Extracted ERP features: {erp_features.shape[1]} features from "
          f"{len(time_windows)} time windows and {len(channels)} channels")
    
    return erp_features


def extract_spectral_features(epochs: mne.Epochs,
                             freq_bands: Optional[Dict[str, Tuple[float, float]]] = None,
                             channels: Optional[List[str]] = None,
                             method: str = 'multitaper') -> np.ndarray:
    """Extract spectral power features from epochs.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    freq_bands : dict, optional
        Dictionary mapping band names to (fmin, fmax) tuples
        If None, uses standard bands: delta, theta, alpha, beta, gamma
    channels : list, optional
        Channel names to include. If None, uses all EEG channels
    method : str
        Method for spectral estimation ('multitaper', 'welch')
    
    Returns
    -------
    spectral_features : ndarray
        Shape (n_epochs, n_features)
        Features: [delta_ch1, delta_ch2, ..., theta_ch1, ...]
    """
    if freq_bands is None:
        freq_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
    
    if channels is None:
        picks = mne.pick_types(epochs.info, meg=False, eeg=True)
        channels = [epochs.ch_names[i] for i in picks]
    
    # Compute power spectral density
    if method == 'multitaper':
        psds, freqs = mne.time_frequency.psd_multitaper(epochs, picks=channels,
                                                        fmin=1, fmax=50,
                                                        verbose=False)
    else:  # welch
        psds, freqs = mne.time_frequency.psd_welch(epochs, picks=channels,
                                                   fmin=1, fmax=50,
                                                   verbose=False)
    
    # psds shape: (n_epochs, n_channels, n_freqs)
    features_list = []
    
    for band_name, (fmin, fmax) in freq_bands.items():
        # Find frequency indices
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        
        # Average power in frequency band
        band_power = np.mean(psds[:, :, freq_mask], axis=2)  # (n_epochs, n_channels)
        
        # Log transform
        band_power_log = np.log10(band_power + 1e-10)
        
        features_list.append(band_power_log)
    
    # Concatenate all features
    spectral_features = np.concatenate(features_list, axis=1)
    
    print(f"Extracted spectral features: {spectral_features.shape[1]} features from "
          f"{len(freq_bands)} frequency bands and {len(channels)} channels")
    
    return spectral_features


def extract_spatial_features(epochs: mne.Epochs,
                            n_components: int = 10,
                            method: str = 'pca') -> np.ndarray:
    """Extract spatial features using dimensionality reduction.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    n_components : int
        Number of components to extract
    method : str
        Method for dimensionality reduction ('pca', 'ica')
    
    Returns
    -------
    spatial_features : ndarray
        Shape (n_epochs, n_components)
    """
    from sklearn.decomposition import PCA, FastICA
    
    # Get epoch data
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape
    
    # Reshape to (n_epochs, n_channels * n_times)
    data_flat = data.reshape(n_epochs, -1)
    
    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    else:  # ica
        reducer = FastICA(n_components=n_components, random_state=42)
    
    spatial_features = reducer.fit_transform(data_flat)
    
    print(f"Extracted {n_components} spatial components using {method.upper()}")
    
    return spatial_features


def extract_time_frequency_features(epochs: mne.Epochs,
                                   freqs: np.ndarray,
                                   n_cycles: Optional[np.ndarray] = None,
                                   channels: Optional[List[str]] = None,
                                   baseline_mode: str = 'ratio') -> np.ndarray:
    """Extract time-frequency features using wavelet transform.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    freqs : ndarray
        Frequencies of interest (Hz)
    n_cycles : ndarray, optional
        Number of cycles for each frequency. If None, uses freqs / 2
    channels : list, optional
        Channel names to include
    baseline_mode : str
        Baseline correction mode ('ratio', 'percent', 'zscore', 'mean')
    
    Returns
    -------
    tf_features : ndarray
        Shape (n_epochs, n_features)
    """
    if n_cycles is None:
        n_cycles = freqs / 2.0
    
    if channels is None:
        picks = mne.pick_types(epochs.info, meg=False, eeg=True)
        channels = [epochs.ch_names[i] for i in picks]
    
    # Compute time-frequency representation
    tfr = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                                       picks=channels, return_itc=False,
                                       average=False, verbose=False)
    
    # Apply baseline correction
    if baseline_mode:
        tfr.apply_baseline(baseline=(-0.2, 0), mode=baseline_mode)
    
    # Get power data: (n_epochs, n_channels, n_freqs, n_times)
    power = tfr.data
    
    # Average over time and channels for each frequency
    tf_features = np.mean(power, axis=(1, 3))  # (n_epochs, n_freqs)
    
    print(f"Extracted time-frequency features: {tf_features.shape[1]} frequencies")
    
    return tf_features


def extract_all_features(epochs: mne.Epochs,
                        include_erp: bool = True,
                        include_spectral: bool = True,
                        include_spatial: bool = False,
                        normalize: bool = True) -> Dict[str, np.ndarray]:
    """Extract all feature types from epochs.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    include_erp : bool
        Extract ERP features
    include_spectral : bool
        Extract spectral features
    include_spatial : bool
        Extract spatial features (can be slow)
    normalize : bool
        Z-score normalize features
    
    Returns
    -------
    features_dict : dict
        Dictionary with feature arrays and concatenated 'all' features
    """
    features_dict = {}
    all_features = []
    
    if include_erp:
        erp_features = extract_erp_features(epochs)
        if normalize:
            erp_features = zscore(erp_features, axis=0)
        features_dict['erp'] = erp_features
        all_features.append(erp_features)
    
    if include_spectral:
        spectral_features = extract_spectral_features(epochs)
        if normalize:
            spectral_features = zscore(spectral_features, axis=0)
        features_dict['spectral'] = spectral_features
        all_features.append(spectral_features)
    
    if include_spatial:
        spatial_features = extract_spatial_features(epochs)
        if normalize:
            spatial_features = zscore(spatial_features, axis=0)
        features_dict['spatial'] = spatial_features
        all_features.append(spatial_features)
    
    # Concatenate all features
    if all_features:
        features_dict['all'] = np.concatenate(all_features, axis=1)
        print(f"Total features: {features_dict['all'].shape[1]}")
    
    return features_dict


if __name__ == "__main__":
    # Example usage
    pass
