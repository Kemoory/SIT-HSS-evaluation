"""
Module d'évaluation et de métriques
"""
from .metrics import (
    boundary_recall,
    under_segmentation_error,
    achievable_segmentation_accuracy,
    compactness,
    regularity,
    compute_all_metrics,
    compute_metrics_multiple_gt,
    format_metrics
)

from .visualize import (
    visualize_segmentation,
    compare_segmentations,
    plot_metrics_comparison,
    plot_parameter_study,
    visualize_superpixel_sizes,
    create_comparison_grid,
    save_visualization
)

__all__ = [
    # Metrics
    'boundary_recall',
    'under_segmentation_error',
    'achievable_segmentation_accuracy',
    'compactness',
    'regularity',
    'compute_all_metrics',
    'compute_metrics_multiple_gt',
    'format_metrics',
    # Visualization
    'visualize_segmentation',
    'compare_segmentations',
    'plot_metrics_comparison',
    'plot_parameter_study',
    'visualize_superpixel_sizes',
    'create_comparison_grid',
    'save_visualization'
]