"""Load EEG data from various file formats.

This module handles loading raw EEG data and associated metadata
for subsequent preprocessing steps.
"""

import mne
import numpy as np
from pathlib import Path
from typing import Union, List, Optional
import yaml


def load_raw_eeg(filepath: Union[str, Path], 
                 file_format: str = 'auto') -> mne.io.Raw:
    """Load raw EEG data from file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the EEG data file
    file_format : str
        File format ('auto', 'bdf', 'edf', 'fif', 'set', etc.)
        If 'auto', will attempt to infer from file extension
    
    Returns
    -------
    raw : mne.io.Raw
        Raw EEG data object
    """
    filepath = Path(filepath)
    
    if file_format == 'auto':
        ext = filepath.suffix.lower()
        format_map = {
            '.bdf': 'biosemi',
            '.edf': 'edf',
            '.fif': 'fif',
            '.set': 'eeglab',
            '.vhdr': 'brainvision'
        }
        file_format = format_map.get(ext, 'auto')
    
    # Load based on format
    if file_format == 'biosemi':
        raw = mne.io.read_raw_bdf(filepath, preload=True)
    elif file_format == 'edf':
        raw = mne.io.read_raw_edf(filepath, preload=True)
    elif file_format == 'fif':
        raw = mne.io.read_raw_fif(filepath, preload=True)
    elif file_format == 'eeglab':
        raw = mne.io.read_raw_eeglab(filepath, preload=True)
    elif file_format == 'brainvision':
        raw = mne.io.read_raw_brainvision(filepath, preload=True)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    print(f"Loaded EEG data: {raw.n_times} timepoints, "
          f"{len(raw.ch_names)} channels, {raw.info['sfreq']} Hz")
    
    return raw


def load_events(raw: mne.io.Raw, 
                event_id: Optional[dict] = None) -> tuple:
    """Extract events from raw EEG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data object
    event_id : dict, optional
        Dictionary mapping event names to integer codes
    
    Returns
    -------
    events : ndarray
        Events array (n_events, 3)
    event_id : dict
        Event ID dictionary
    """
    # Find events from stimulus channel
    events = mne.find_events(raw, stim_channel='auto')
    
    if event_id is None:
        # Create default event_id from unique event codes
        unique_events = np.unique(events[:, 2])
        event_id = {f'event_{code}': code for code in unique_events}
    
    print(f"Found {len(events)} events with codes: {list(event_id.keys())}")
    
    return events, event_id


def load_subject_data(subject_id: str, 
                     data_dir: Union[str, Path],
                     config: Optional[dict] = None) -> dict:
    """Load all data for a single subject.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier
    data_dir : str or Path
        Root directory containing subject data
    config : dict, optional
        Configuration dictionary with data loading parameters
    
    Returns
    -------
    subject_data : dict
        Dictionary containing raw data, events, and metadata
    """
    data_dir = Path(data_dir)
    subject_dir = data_dir / f"sub-{subject_id}"
    
    # Load configuration
    if config is None:
        config = {}
    
    # Find EEG file
    eeg_files = list(subject_dir.glob("*.bdf")) + \
                list(subject_dir.glob("*.edf")) + \
                list(subject_dir.glob("*.fif"))
    
    if not eeg_files:
        raise FileNotFoundError(f"No EEG files found for subject {subject_id}")
    
    eeg_file = eeg_files[0]
    
    # Load raw data
    raw = load_raw_eeg(eeg_file)
    
    # Load events
    events, event_id = load_events(raw)
    
    subject_data = {
        'subject_id': subject_id,
        'raw': raw,
        'events': events,
        'event_id': event_id,
        'filepath': eeg_file
    }
    
    return subject_data


if __name__ == "__main__":
    # Example usage
    pass
