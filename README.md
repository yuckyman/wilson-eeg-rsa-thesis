# EEG-RSA Analysis Pipeline: Imagery-Perception Comparison

A comprehensive Python pipeline for analyzing EEG data using Representational Similarity Analysis (RSA) to compare mental imagery and visual perception, following the methodology of Wilson et al.

## Overview

This repository contains a complete analysis pipeline for investigating shared neural representations between mental imagery and visual perception using EEG and RSA techniques.

### Key Features

- **Complete preprocessing pipeline**: filtering, artifact removal, epoching
- **Multiple feature extraction methods**: ERP, spectral, spatial features
- **RSA implementation**: RDM computation with various distance metrics
- **Statistical testing**: permutation tests, bootstrap CIs, model comparison
- **Visualization tools**: publication-quality figures and plots
- **Flexible configuration**: YAML-based parameter management

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yuckyman/wilson-eeg-rsa-thesis.git
cd wilson-eeg-rsa-thesis
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
wilson-eeg-rsa-thesis/
├── data/
│   ├── raw/              # Raw EEG files (.bdf, .edf, .fif, etc.)
│   ├── preprocessed/    # Preprocessed and epoched data
│   └── outputs/         # Analysis results, RDMs, figures
├── preprocessing/
│   ├── load_data.py          # Load raw EEG data
│   ├── filter_eeg.py         # Filtering and referencing
│   ├── artifact_removal.py   # ICA-based artifact removal
│   └── epoching.py           # Epoch extraction
├── features/
│   └── extract_features.py   # ERP, spectral, spatial features
├── rsa/
│   ├── compute_rdm.py        # RDM computation
│   ├── model_rdms.py         # Theoretical model RDMs
│   └── statistics.py         # Statistical testing
├── analysis/
│   ├── imagery_perception_comparison.py  # Main analysis script
│   └── visualization.py                 # Plotting functions
├── notebooks/
│   └── exploratory_analysis.ipynb  # Jupyter notebook template
├── config/
│   ├── preprocessing_config.yaml  # Preprocessing parameters
│   ├── analysis_config.yaml       # Analysis parameters
│   └── paths.yaml                 # Path configuration
├── requirements.txt
├── README.md
└── .gitignore
```

## Usage

### Quick Start

1. **Place your EEG data** in the `data/raw/` directory

2. **Update configuration files** in `config/` with your experiment parameters

3. **Run the analysis** using the Jupyter notebook:

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

### Step-by-Step Pipeline

#### 1. Load and Preprocess Data

```python
from preprocessing import load_raw_eeg, preprocess_raw

# Load raw EEG
raw = load_raw_eeg('data/raw/subject_01.bdf')

# Apply preprocessing
raw_clean = preprocess_raw(raw, l_freq=0.1, h_freq=40.0, notch_freq=50.0)
```

#### 2. Artifact Removal

```python
from preprocessing import remove_artifacts_pipeline

# Remove artifacts using ICA
raw_clean, ica = remove_artifacts_pipeline(raw_clean, 
                                          detect_bad_chans=True,
                                          run_ica_cleaning=True)
```

#### 3. Extract Epochs

```python
from preprocessing import create_epochs
import mne

# Define events
events = mne.find_events(raw_clean)
event_id = {
    'imagery_face': 1,
    'imagery_house': 2,
    'perception_face': 11,
    'perception_house': 12
}

# Create epochs
epochs = create_epochs(raw_clean, events, event_id, 
                      tmin=-0.2, tmax=1.0)
```

#### 4. Extract Features

```python
from features import extract_all_features

# Extract ERP and spectral features
features_dict = extract_all_features(epochs, 
                                    include_erp=True,
                                    include_spectral=True,
                                    normalize=True)
```

#### 5. Compute RDMs

```python
from rsa import compute_rdm, visualize_rdm
import numpy as np

# Average features by condition
imagery_face_features = features_dict['all'][epochs['imagery_face'].selection]
imagery_house_features = features_dict['all'][epochs['imagery_house'].selection]

imagery_cond_features = np.vstack([
    imagery_face_features.mean(axis=0),
    imagery_house_features.mean(axis=0)
])

# Compute RDM
rdm_imagery = compute_rdm(imagery_cond_features, metric='correlation')

# Visualize
visualize_rdm(rdm_imagery, labels=['Face', 'House'], title='Imagery RDM')
```

#### 6. Statistical Testing

```python
from rsa import permutation_test, compare_models
from rsa import create_imagery_perception_model

# Test correlation between imagery and perception
corr, p_value, null_dist = permutation_test(rdm_imagery, rdm_perception, 
                                           n_permutations=10000)

# Compare with theoretical models
models = create_imagery_perception_model(condition_types, stimulus_types)
results = compare_models(rdm_data, models)
```

### Complete Analysis

Run the full analysis pipeline:

```python
from analysis import run_full_analysis

results = run_full_analysis(
    imagery_epochs=imagery_epochs,
    perception_epochs=perception_epochs,
    stimulus_labels=['Face', 'House'],
    feature_type='all',
    save_dir='data/outputs'
)
```

## Methodology

### RSA (Representational Similarity Analysis)

RSA is a powerful technique for comparing neural representations across conditions, modalities, or species. The pipeline implements:

1. **Feature Extraction**: Multiple feature types from EEG epochs
2. **RDM Computation**: Pairwise dissimilarity between conditions
3. **Model Comparison**: Test theoretical predictions
4. **Statistical Testing**: Permutation tests and bootstrap confidence intervals

### Wilson et al. Paradigm

This pipeline is designed for experiments comparing:
- **Mental Imagery**: Internal generation of visual representations
- **Visual Perception**: External presentation of stimuli

Key question: Do imagery and perception share common neural representations?

### Features Extracted

- **ERP Features**: Amplitude in specific time windows (P1, N170, P300)
- **Spectral Features**: Power in frequency bands (delta, theta, alpha, beta, gamma)
- **Spatial Features**: PCA/ICA components (optional)

### Distance Metrics

- Correlation distance (1 - Pearson correlation)
- Euclidean distance
- Cosine distance

## Configuration

Edit YAML files in `config/` to customize:

### preprocessing_config.yaml
- Filtering parameters
- ICA settings
- Epoch extraction parameters

### analysis_config.yaml
- Feature extraction options
- RDM computation settings
- Statistical test parameters

### paths.yaml
- Data directories
- Subject information
- File naming conventions

## Outputs

The pipeline generates:

- **Preprocessed data**: Cleaned and epoched EEG files (.fif format)
- **RDMs**: NumPy arrays (.npy) and visualizations (.png)
- **Statistical results**: Test results and model comparisons
- **Figures**: Publication-ready plots (300 dpi)

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{wilson2023imagery,
  title={Shared representations between imagery and perception in EEG},
  author={Wilson, A. et al.},
  journal={Journal Name},
  year={2023}
}
```

## References

- **RSA**: Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). Representational similarity analysis - connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*, 2, 4.

- **MNE-Python**: Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.

- **Wilson et al.**: [Update with actual reference when available]

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, please open an issue on GitHub or contact:
- Ian (repository owner)
- Email: [your-email@domain.com]

## Acknowledgments

- MNE-Python development team
- Wilson et al. for the original methodology
- Contributors and collaborators

---

**Note**: This is a research pipeline for EEG-RSA analysis. Please ensure you understand the methods and validate results for your specific use case.
