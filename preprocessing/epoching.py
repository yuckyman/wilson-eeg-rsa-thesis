"""Epoch extraction and processing for EEG data.

Extract epochs around events, apply baseline correction, and reject bad epochs.
"""

import mne
import numpy as np
from typing import Optional, Union, Tuple


def create_epochs(raw: mne.io.Raw,
                 events: np.ndarray,
                 event_id: dict,
                 tmin: float = -0.2,
                 tmax: float = 1.0,
                 baseline: Optional[Tuple[float, float]] = (-0.2, 0.0),
                 detrend: Optional[int] = None,
                 preload: bool = True) -> mne.Epochs:
    """Create epochs from continuous EEG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data object
    events : ndarray
        Events array (n_events, 3)
    event_id : dict
        Dictionary mapping event names to integer codes
    tmin : float
        Start time before event (seconds)
    tmax : float
        End time after event (seconds)
    baseline : tuple or None
        Baseline period (start, end) in seconds
    detrend : int or None
        Detrend type (0=constant, 1=linear)
    preload : bool
        Load all epochs into memory
    
    Returns
    -------
    epochs : mne.Epochs
        Epochs object
    """
    epochs = mne.Epochs(raw, events, event_id,
                       tmin=tmin, tmax=tmax,
                       baseline=baseline, detrend=detrend,
                       preload=preload, verbose=True)
    
    print(f"Created {len(epochs)} epochs for {list(event_id.keys())}")
    return epochs


def reject_bad_epochs(epochs: mne.Epochs,
                     reject_criteria: Optional[dict] = None,
                     use_autoreject: bool = False) -> mne.Epochs:
    """Reject epochs with artifacts based on amplitude thresholds.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    reject_criteria : dict, optional
        Peak-to-peak amplitude rejection thresholds for each channel type
        e.g., {'eeg': 100e-6} for 100 µV
        If None, uses default: {'eeg': 150e-6}
    use_autoreject : bool
        If True, use autoreject algorithm (requires autoreject package)
    
    Returns
    -------
    epochs_clean : mne.Epochs
        Epochs with bad epochs rejected
    """
    if reject_criteria is None:
        reject_criteria = {'eeg': 150e-6}  # 150 µV
    
    if use_autoreject:
        try:
            from autoreject import AutoReject
            ar = AutoReject(n_interpolate=[1, 2, 3, 4],
                          n_jobs=-1, random_state=42, verbose=True)
            epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
            
            print(f"AutoReject removed {reject_log.bad_epochs.sum()} epochs")
        except ImportError:
            print("AutoReject not available, using threshold-based rejection")
            epochs_clean = epochs.copy().drop_bad(reject=reject_criteria)
    else:
        epochs_clean = epochs.copy().drop_bad(reject=reject_criteria)
    
    n_rejected = len(epochs) - len(epochs_clean)
    print(f"Rejected {n_rejected} bad epochs ({n_rejected/len(epochs)*100:.1f}%)")
    
    return epochs_clean


def equalize_epoch_counts(epochs_list: list,
                         method: str = 'mintime') -> list:
    """Equalize epoch counts across conditions.
    
    Parameters
    ----------
    epochs_list : list of mne.Epochs
        List of Epochs objects for different conditions
    method : str
        Method for equalization ('mintime', 'truncate')
    
    Returns
    -------
    epochs_equalized : list of mne.Epochs
        List of Epochs with equalized counts
    """
    mne.epochs.equalize_epoch_counts(epochs_list, method=method)
    
    counts = [len(ep) for ep in epochs_list]
    print(f"Equalized epoch counts: {counts}")
    
    return epochs_list


def create_condition_epochs(raw: mne.io.Raw,
                           events: np.ndarray,
                           event_id: dict,
                           conditions: dict,
                           tmin: float = -0.2,
                           tmax: float = 1.0,
                           baseline: Optional[Tuple[float, float]] = (-0.2, 0.0),
                           reject: Optional[dict] = None) -> dict:
    """Create separate epoch objects for different experimental conditions.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data object
    events : ndarray
        Events array
    event_id : dict
        All event codes
    conditions : dict
        Dictionary mapping condition names to event code lists
        e.g., {'imagery': ['img_face', 'img_house'], 
               'perception': ['per_face', 'per_house']}
    tmin, tmax : float
        Epoch time window
    baseline : tuple or None
        Baseline period
    reject : dict, optional
        Rejection criteria
    
    Returns
    -------
    condition_epochs : dict
        Dictionary of Epochs objects for each condition
    """
    condition_epochs = {}
    
    for condition_name, condition_events in conditions.items():
        # Filter event_id for this condition
        condition_event_id = {k: v for k, v in event_id.items() 
                             if k in condition_events}
        
        # Create epochs
        epochs = create_epochs(raw, events, condition_event_id,
                             tmin=tmin, tmax=tmax, baseline=baseline)
        
        # Reject bad epochs
        if reject is not None:
            epochs = reject_bad_epochs(epochs, reject_criteria=reject)
        
        condition_epochs[condition_name] = epochs
    
    return condition_epochs


def extract_imagery_perception_epochs(raw: mne.io.Raw,
                                     events: np.ndarray,
                                     event_id: dict,
                                     tmin: float = -0.2,
                                     tmax: float = 1.0,
                                     baseline: Optional[Tuple[float, float]] = (-0.2, 0.0)) -> dict:
    """Extract epochs for imagery and perception conditions (Wilson et al. paradigm).
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data object
    events : ndarray
        Events array
    event_id : dict
        Event codes dictionary
    tmin, tmax : float
        Epoch time window
    baseline : tuple or None
        Baseline period
    
    Returns
    -------
    epochs_dict : dict
        Dictionary with 'imagery' and 'perception' Epochs objects
    """
    # Automatically detect imagery and perception events
    imagery_events = {k: v for k, v in event_id.items() 
                     if 'imag' in k.lower() or 'mental' in k.lower()}
    perception_events = {k: v for k, v in event_id.items() 
                        if 'percep' in k.lower() or 'visual' in k.lower() or 'stim' in k.lower()}
    
    print(f"Imagery events: {list(imagery_events.keys())}")
    print(f"Perception events: {list(perception_events.keys())}")
    
    # Create epochs for each condition
    imagery_epochs = create_epochs(raw, events, imagery_events,
                                  tmin=tmin, tmax=tmax, baseline=baseline)
    perception_epochs = create_epochs(raw, events, perception_events,
                                     tmin=tmin, tmax=tmax, baseline=baseline)
    
    epochs_dict = {
        'imagery': imagery_epochs,
        'perception': perception_epochs
    }
    
    return epochs_dict


if __name__ == "__main__":
    # Example usage
    pass
