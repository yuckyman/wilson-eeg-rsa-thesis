"""Apply frequency filters to EEG data.

This module implements bandpass, highpass, and lowpass filtering
for EEG signal preprocessing.
"""

import mne
import numpy as np
from typing import Optional, Union


def apply_bandpass_filter(raw: mne.io.Raw,
                         l_freq: float = 0.1,
                         h_freq: float = 40.0,
                         method: str = 'fir',
                         filter_length: str = 'auto',
                         phase: str = 'zero') -> mne.io.Raw:
    """Apply bandpass filter to raw EEG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data object
    l_freq : float
        Lower frequency bound (Hz)
    h_freq : float
        Upper frequency bound (Hz)
    method : str
        Filter method ('fir' or 'iir')
    filter_length : str
        Length of the filter ('auto' or int)
    phase : str
        Phase of the filter ('zero', 'zero-double', 'minimum')
    
    Returns
    -------
    raw_filtered : mne.io.Raw
        Filtered EEG data object
    """
    raw_filtered = raw.copy()
    raw_filtered.filter(l_freq=l_freq, h_freq=h_freq,
                       method=method, filter_length=filter_length,
                       phase=phase, verbose=True)
    
    print(f"Applied bandpass filter: {l_freq}-{h_freq} Hz")
    return raw_filtered


def apply_notch_filter(raw: mne.io.Raw,
                      freqs: Union[float, list] = 50.0,
                      notch_widths: Optional[float] = None) -> mne.io.Raw:
    """Apply notch filter to remove line noise.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data object
    freqs : float or list
        Frequency or list of frequencies to notch out (Hz)
        Common values: 50 Hz (Europe) or 60 Hz (US)
    notch_widths : float, optional
        Width of the notch filter
    
    Returns
    -------
    raw_filtered : mne.io.Raw
        Filtered EEG data object
    """
    raw_filtered = raw.copy()
    raw_filtered.notch_filter(freqs=freqs, notch_widths=notch_widths,
                             verbose=True)
    
    print(f"Applied notch filter at {freqs} Hz")
    return raw_filtered


def apply_reference(raw: mne.io.Raw,
                   ref_channels: Union[str, list] = 'average') -> mne.io.Raw:
    """Apply reference to EEG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data object
    ref_channels : str or list
        Reference channels ('average', 'REST', or list of channel names)
    
    Returns
    -------
    raw_reref : mne.io.Raw
        Re-referenced EEG data object
    """
    raw_reref = raw.copy()
    
    if ref_channels == 'average':
        raw_reref.set_eeg_reference('average', projection=False)
        print("Applied average reference")
    elif ref_channels == 'REST':
        # Requires sphere parameter and forward model
        sphere = mne.make_sphere_model('auto', 'auto', raw_reref.info)
        src = mne.setup_volume_source_space(sphere=sphere, pos=10.0)
        raw_reref.set_eeg_reference('REST', forward=src)
        print("Applied REST reference")
    else:
        raw_reref.set_eeg_reference(ref_channels)
        print(f"Applied reference to {ref_channels}")
    
    return raw_reref


def preprocess_raw(raw: mne.io.Raw,
                  l_freq: float = 0.1,
                  h_freq: float = 40.0,
                  notch_freq: Union[float, list] = 50.0,
                  reference: str = 'average') -> mne.io.Raw:
    """Complete preprocessing pipeline for raw EEG.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data object
    l_freq : float
        Lower frequency bound for bandpass filter (Hz)
    h_freq : float
        Upper frequency bound for bandpass filter (Hz)
    notch_freq : float or list
        Frequency for notch filter (Hz)
    reference : str or list
        Reference type ('average', 'REST', or channel names)
    
    Returns
    -------
    raw_processed : mne.io.Raw
        Preprocessed EEG data object
    """
    print("Starting preprocessing pipeline...")
    
    # Apply filters
    raw_processed = apply_bandpass_filter(raw, l_freq=l_freq, h_freq=h_freq)
    raw_processed = apply_notch_filter(raw_processed, freqs=notch_freq)
    
    # Apply reference
    raw_processed = apply_reference(raw_processed, ref_channels=reference)
    
    print("Preprocessing complete")
    return raw_processed


if __name__ == "__main__":
    # Example usage
    pass
