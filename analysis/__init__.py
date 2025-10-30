"""Analysis module for EEG-RSA experiments.

Provides high-level analysis functions and visualization tools.
"""

from .imagery_perception_comparison import (
    load_epoched_data,
    extract_condition_features,
    compute_condition_rdms,
    test_shared_representations,
    compare_with_models,
    visualize_results,
    run_full_analysis
)

from .visualization import (
    plot_rdm_matrix,
    plot_rdm_comparison,
    plot_mds,
    plot_time_resolved_rsa,
    plot_erp_comparison,
    plot_topomaps,
    create_summary_figure
)

__all__ = [
    'load_epoched_data', 'extract_condition_features', 'compute_condition_rdms',
    'test_shared_representations', 'compare_with_models', 'visualize_results',
    'run_full_analysis',
    'plot_rdm_matrix', 'plot_rdm_comparison', 'plot_mds',
    'plot_time_resolved_rsa', 'plot_erp_comparison', 'plot_topomaps',
    'create_summary_figure'
]
