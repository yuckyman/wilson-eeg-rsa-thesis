"""Visualization utilities for EEG-RSA analysis.

Create publication-quality figures for RDM analysis, time-resolved RSA,
and topographic maps.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict
import mne


def plot_rdm_matrix(rdm: np.ndarray,
                    labels: Optional[List[str]] = None,
                    title: str = "RDM",
                    cmap: str = 'viridis',
                    annot: bool = True,
                    figsize: Tuple[int, int] = (8, 6),
                    save_path: Optional[str] = None) -> None:
    """Plot a single RDM as a heatmap.
    
    Parameters
    ----------
    rdm : ndarray
        Representational Dissimilarity Matrix
    labels : list, optional
        Labels for conditions
    title : str
        Plot title
    cmap : str
        Colormap name
    annot : bool
        Whether to annotate cells with values
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(rdm, annot=annot, fmt='.2f' if annot else '',
               cmap=cmap, square=True, cbar_kws={'label': 'Dissimilarity'},
               xticklabels=labels, yticklabels=labels, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_rdm_comparison(rdm1: np.ndarray,
                       rdm2: np.ndarray,
                       labels1: Optional[List[str]] = None,
                       labels2: Optional[List[str]] = None,
                       title1: str = "RDM 1",
                       title2: str = "RDM 2",
                       cmap: str = 'viridis',
                       save_path: Optional[str] = None) -> None:
    """Plot two RDMs side by side for comparison.
    
    Parameters
    ----------
    rdm1, rdm2 : ndarray
        Two RDMs to compare
    labels1, labels2 : list, optional
        Labels for each RDM
    title1, title2 : str
        Titles for each RDM
    cmap : str
        Colormap name
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Find common colorbar limits
    vmin = min(rdm1.min(), rdm2.min())
    vmax = max(rdm1.max(), rdm2.max())
    
    # Plot first RDM
    sns.heatmap(rdm1, annot=False, cmap=cmap, square=True,
               vmin=vmin, vmax=vmax,
               xticklabels=labels1, yticklabels=labels1,
               cbar_kws={'label': 'Dissimilarity'}, ax=axes[0])
    axes[0].set_title(title1, fontsize=14, fontweight='bold')
    
    # Plot second RDM
    sns.heatmap(rdm2, annot=False, cmap=cmap, square=True,
               vmin=vmin, vmax=vmax,
               xticklabels=labels2, yticklabels=labels2,
               cbar_kws={'label': 'Dissimilarity'}, ax=axes[1])
    axes[1].set_title(title2, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_mds(rdm: np.ndarray,
            labels: Optional[List[str]] = None,
            colors: Optional[List] = None,
            title: str = "MDS Representation",
            n_components: int = 2,
            save_path: Optional[str] = None) -> None:
    """Plot multidimensional scaling (MDS) of RDM.
    
    Parameters
    ----------
    rdm : ndarray
        Representational Dissimilarity Matrix
    labels : list, optional
        Labels for each condition
    colors : list, optional
        Colors for each condition
    title : str
        Plot title
    n_components : int
        Number of MDS dimensions (2 or 3)
    save_path : str, optional
        Path to save figure
    """
    from sklearn.manifold import MDS
    
    # Perform MDS
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(rdm)
    
    if n_components == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=100, alpha=0.7)
        
        # Add labels
        if labels:
            for i, label in enumerate(labels):
                ax.annotate(label, (coords[i, 0], coords[i, 1]),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=10, fontweight='bold')
        
        ax.set_xlabel('MDS Dimension 1', fontsize=12)
        ax.set_ylabel('MDS Dimension 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
    else:  # 3D
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                           c=colors, s=100, alpha=0.7)
        
        # Add labels
        if labels:
            for i, label in enumerate(labels):
                ax.text(coords[i, 0], coords[i, 1], coords[i, 2], label,
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('MDS Dimension 1', fontsize=12)
        ax.set_ylabel('MDS Dimension 2', fontsize=12)
        ax.set_zlabel('MDS Dimension 3', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_time_resolved_rsa(rdms_time: np.ndarray,
                          times: np.ndarray,
                          model_rdm: np.ndarray,
                          title: str = "Time-Resolved RSA",
                          significance: Optional[np.ndarray] = None,
                          save_path: Optional[str] = None) -> None:
    """Plot time course of RSA correlations.
    
    Parameters
    ----------
    rdms_time : ndarray
        Time-resolved RDMs (n_times, n_conditions, n_conditions)
    times : ndarray
        Time points in seconds
    model_rdm : ndarray
        Model RDM to compare against
    title : str
        Plot title
    significance : ndarray, optional
        Boolean array indicating significant time points
    save_path : str, optional
        Path to save figure
    """
    from scipy.stats import spearmanr
    
    # Compute correlations at each time point
    n_times = rdms_time.shape[0]
    correlations = np.zeros(n_times)
    
    n = model_rdm.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    model_vec = model_rdm[triu_idx]
    
    for t in range(n_times):
        data_vec = rdms_time[t][triu_idx]
        correlations[t], _ = spearmanr(data_vec, model_vec)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times, correlations, linewidth=2, color='steelblue')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.axvline(0, color='red', linestyle='--', alpha=0.3, label='Stimulus onset')
    
    # Shade significant periods
    if significance is not None:
        sig_periods = np.where(significance)[0]
        if len(sig_periods) > 0:
            ax.fill_between(times, correlations.min(), correlations.max(),
                          where=significance, alpha=0.2, color='green',
                          label='Significant')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Model Correlation (Spearman r)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_erp_comparison(epochs_dict: Dict[str, 'mne.Epochs'],
                       channels: Optional[List[str]] = None,
                       title: str = "ERP Comparison",
                       save_path: Optional[str] = None) -> None:
    """Plot ERP waveforms for different conditions.
    
    Parameters
    ----------
    epochs_dict : dict
        Dictionary of Epochs objects for different conditions
    channels : list, optional
        Channels to plot. If None, uses all channels
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, len(epochs_dict), figsize=(6*len(epochs_dict), 5),
                            sharex=True, sharey=True)
    
    if len(epochs_dict) == 1:
        axes = [axes]
    
    for i, (condition, epochs) in enumerate(epochs_dict.items()):
        evoked = epochs.average(picks=channels)
        evoked.plot(axes=axes[i], show=False, spatial_colors=True, time_unit='s')
        axes[i].set_title(f"{condition}", fontsize=12, fontweight='bold')
        axes[i].axvline(0, color='red', linestyle='--', alpha=0.5)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_topomaps(epochs: 'mne.Epochs',
                 times: List[float],
                 title: str = "Topographic Maps",
                 save_path: Optional[str] = None) -> None:
    """Plot topographic maps at specific time points.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    times : list
        Time points to plot (in seconds)
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    evoked = epochs.average()
    
    fig = evoked.plot_topomap(times=times, ch_type='eeg', 
                             time_unit='s', show=False,
                             colorbar=True, size=3)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def create_summary_figure(results: Dict,
                         save_path: Optional[str] = None) -> None:
    """Create comprehensive summary figure with multiple panels.
    
    Parameters
    ----------
    results : dict
        Results dictionary from analysis
    save_path : str, optional
        Path to save figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # RDM panels would go here
    # Model comparison bars
    # Correlation scatter
    # Time-resolved plot if available
    
    fig.suptitle('EEG-RSA Analysis Summary', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary figure to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    pass
