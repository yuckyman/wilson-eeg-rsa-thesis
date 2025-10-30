# Setup Instructions for wilson-eeg-rsa-thesis

## Quick Start Guide

This document provides step-by-step instructions to get started with the EEG-RSA analysis pipeline.

## 1. Repository Setup

The repository is already created and contains:

✅ Complete directory structure  
✅ All preprocessing modules (load_data, filter_eeg, artifact_removal, epoching)  
✅ Feature extraction module (ERP, spectral, spatial features)  
✅ RSA modules (compute_rdm, model_rdms, statistics)  
✅ Analysis scripts (imagery_perception_comparison, visualization)  
✅ Jupyter notebook template  
✅ Configuration files (YAML)  
✅ Requirements.txt with all dependencies  
✅ Comprehensive README.md  
✅ .gitignore configured for data files  

## 2. Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/yuckyman/wilson-eeg-rsa-thesis.git
cd wilson-eeg-rsa-thesis
```

### Step 2: Create Virtual Environment

**Using venv (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Or using conda:**
```bash
conda create -n eeg-rsa python=3.9
conda activate eeg-rsa
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import mne; import numpy; import scipy; print('Installation successful!')"
```

## 3. Data Organization

Place your EEG data files in the appropriate directories:

```
data/
├── raw/               # Place your raw EEG files here
│   ├── sub-01.bdf
│   ├── sub-02.bdf
│   └── ...
├── preprocessed/      # Processed data will be saved here
└── outputs/           # Analysis results will be saved here
    └── figures/       # Generated figures
```

### Supported File Formats

- BioSemi (.bdf)
- European Data Format (.edf)
- Neuroscan (.cnt)
- BrainVision (.vhdr, .vmrk, .eeg)
- EEGLAB (.set)
- Fieldtrip (.mat)
- MNE-Python (.fif)

## 4. Configuration

### Update Configuration Files

Edit the YAML files in `config/` directory:

**config/preprocessing_config.yaml:**
- Filtering parameters (lowpass, highpass, notch frequency)
- ICA settings
- Epoching parameters
- Event codes for your experiment

**config/analysis_config.yaml:**
- Feature extraction settings
- RDM computation parameters
- Statistical test settings

**config/paths.yaml:**
- Subject IDs
- File naming conventions

## 5. Running the Analysis

### Option A: Using Jupyter Notebook (Recommended for Beginners)

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

Follow the notebook cells step-by-step to:
1. Load and preprocess data
2. Extract epochs
3. Compute features
4. Generate RDMs
5. Run statistical tests
6. Visualize results

### Option B: Using Python Scripts

**Preprocess a single subject:**
```python
from preprocessing import load_raw_eeg, preprocess_raw, remove_artifacts_pipeline
from preprocessing import create_epochs

# Load and preprocess
raw = load_raw_eeg('data/raw/sub-01.bdf')
raw_clean = preprocess_raw(raw, l_freq=0.1, h_freq=40.0)
raw_clean, ica = remove_artifacts_pipeline(raw_clean)

# Create epochs
import mne
events = mne.find_events(raw_clean)
event_id = {'imagery_face': 1, 'perception_face': 11}
epochs = create_epochs(raw_clean, events, event_id)
```

**Run complete RSA analysis:**
```python
from analysis import run_full_analysis

results = run_full_analysis(
    imagery_epochs=imagery_epochs,
    perception_epochs=perception_epochs,
    stimulus_labels=['Face', 'House', 'Object'],
    feature_type='all',
    save_dir='data/outputs'
)
```

## 6. Pipeline Workflow

### Complete Processing Pipeline

```
Raw EEG Data
    ↓
[1] Load Data (load_data.py)
    ↓
[2] Filter & Reference (filter_eeg.py)
    ↓
[3] Artifact Removal (artifact_removal.py)
    ↓
[4] Epoch Extraction (epoching.py)
    ↓
[5] Feature Extraction (extract_features.py)
    ↓
[6] RDM Computation (compute_rdm.py)
    ↓
[7] Statistical Testing (statistics.py)
    ↓
[8] Visualization (visualization.py)
    ↓
Results & Figures
```

## 7. Expected Outputs

### Preprocessed Data
- `data/preprocessed/imagery_epochs-epo.fif`
- `data/preprocessed/perception_epochs-epo.fif`

### RDMs
- `data/outputs/rdm_imagery.npy`
- `data/outputs/rdm_perception.npy`

### Figures
- `data/outputs/figures/rdm_comparison.png`
- `data/outputs/figures/model_comparison.png`
- `data/outputs/figures/erp_plots.png`

### Statistical Results
- Correlation values
- P-values from permutation tests
- Bootstrap confidence intervals
- Model comparison results

## 8. Customization

### Adding New Features

Create new feature extraction functions in `features/extract_features.py`:

```python
def extract_custom_features(epochs):
    # Your custom feature extraction
    return features
```

### Adding New Models

Create new theoretical models in `rsa/model_rdms.py`:

```python
def create_custom_model(conditions):
    # Your custom model RDM
    return rdm
```

### Adding New Analyses

Create new analysis scripts in `analysis/` directory following the template structure.

## 9. Troubleshooting

### Common Issues

**Issue: Import errors**
```bash
# Solution: Make sure virtual environment is activated
source venv/bin/activate  # or conda activate eeg-rsa
pip install -r requirements.txt
```

**Issue: "No module named 'preprocessing'"**
```bash
# Solution: Add parent directory to Python path
import sys
sys.path.append('..')
```

**Issue: MNE-Python errors with channel locations**
```python
# Solution: Set montage for channel positions
montage = mne.channels.make_standard_montage('biosemi64')
raw.set_montage(montage)
```

**Issue: Memory errors with large datasets**
```python
# Solution: Use memory-efficient loading
raw = mne.io.read_raw_fif('file.fif', preload=False)
```

## 10. Best Practices

### Data Management
- Keep raw data separate from processed data
- Use version control for code, not data
- Document any manual preprocessing steps
- Save intermediate results

### Analysis Workflow
- Start with a single subject/pilot data
- Verify each preprocessing step visually
- Use cross-validation for robust results
- Document parameter choices

### Code Organization
- Use configuration files for parameters
- Write modular, reusable functions
- Add comments and docstrings
- Test with subset of data first

## 11. Next Steps

1. **Validate preprocessing**: Check filtered data and artifact removal
2. **Explore features**: Examine ERP waveforms and spectral profiles
3. **Compute RDMs**: Generate RDMs for imagery and perception
4. **Statistical testing**: Run permutation tests and model comparisons
5. **Visualization**: Create publication-quality figures
6. **Iterate**: Refine parameters based on results

## 12. Getting Help

### Documentation
- MNE-Python: https://mne.tools/stable/index.html
- NumPy: https://numpy.org/doc/
- SciPy: https://docs.scipy.org/
- Matplotlib: https://matplotlib.org/

### Resources
- RSA Toolbox: https://github.com/rsagroup/rsatoolbox
- EEG preprocessing guide: MNE tutorials
- Wilson et al. paper: [Add reference]

### Support
- Open an issue on GitHub
- Check existing issues for solutions
- Contact repository maintainer

## 13. Citation

If you use this pipeline, please cite:

```bibtex
@software{wilson_eeg_rsa_2024,
  title = {EEG-RSA Analysis Pipeline for Imagery-Perception Comparison},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yuckyman/wilson-eeg-rsa-thesis}
}
```

---

**Happy Analyzing! 🧠📊**

For questions or suggestions, please open an issue on GitHub.
