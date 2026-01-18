"""
Fonctions de visualisation pour l'évaluation des superpixels

Fournit des outils pour visualiser:
- Segmentations avec contours
- Comparaisons entre méthodes
- Distributions de métriques
- Analyses de paramètres
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import segmentation, color as skcolor
import seaborn as sns

def visualize_segmentation(image, labels, title="Segmentation",
                          show_boundaries=True, boundary_color=(1, 1, 0)):
    """
    Visualise une segmentation en superpixels.

    Args:
        image: Image originale RGB (H, W, 3) avec valeurs [0, 255] ou [0, 1]
        labels: Labels des superpixels (H, W)
        title: Titre de la figure
        show_boundaries: Si True, affiche les contours
        boundary_color: Couleur des contours RGB (default: jaune)

    Returns:
        fig: Figure matplotlib
    """
    if image.max() > 1.0:
        image = image / 255.0
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title('Image originale')
    axes[0].axis('off')
    
    colored = skcolor.label2rgb(labels, image, kind='avg', bg_label=-1)
    axes[1].imshow(colored)
    axes[1].set_title('Couleurs moyennes')
    axes[1].axis('off')
    
    if show_boundaries:
        marked = segmentation.mark_boundaries(image, labels, color=boundary_color, mode='thick', background_label=-1)
        axes[2].imshow(marked)
        axes[2].set_title('Avec contours')
    else:
        axes[2].imshow(image)
        axes[2].set_title('Image originale')
        axes[2].axis('off')
    
    fig.suptitle(title, fontsize=16, fontweight='bold')

    return fig

def visualize_superpixel_sizes(labels):
    """
    Visualise la distribution des tailles de superpixels.

    Args:
        labels: Labels des superpixels (H, W)

    Returns:
        fig: Figure matplotlib avec histogramme et box plot

    Example:
        >>> labels = slic.fit(image)
        >>> fig = visualize_superpixel_sizes(labels)
        >>> plt.show()
    """
    unique_labels = np.unique(labels)
    sizes = [np.sum(labels == label) for label in unique_labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(sizes, bins=30, color='steelblue', alpha=0.7,
                edgecolor='black', linewidth=1.2)
    axes[0].axvline(np.mean(sizes), color='red', linestyle='--',
                   linewidth=2, label=f'Moyenne: {np.mean(sizes):.1f}')
    axes[0].axvline(np.median(sizes), color='green', linestyle='--',
                   linewidth=2, label=f'Médiane: {np.median(sizes):.1f}')
    axes[0].set_xlabel('Taille (pixels)', fontsize=11)
    axes[0].set_ylabel('Fréquence', fontsize=11)
    axes[0].set_title('Distribution des tailles', fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    bp = axes[1].boxplot(sizes, vert=True, patch_artist=True,
                        widths=0.5)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)
    bp['medians'][0].set_color('red')
    bp['medians'][0].set_linewidth(2)

    axes[1].set_ylabel('Taille (pixels)', fontsize=11)
    axes[1].set_title('Box plot des tailles', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_xticks([])

    stats_text = f'Statistiques:\n'
    stats_text += f'N superpixels: {len(sizes)}\n'
    stats_text += f'Min: {np.min(sizes)}\n'
    stats_text += f'Max: {np.max(sizes)}\n'
    stats_text += f'Moyenne: {np.mean(sizes):.1f}\n'
    stats_text += f'Médiane: {np.median(sizes):.1f}\n'
    stats_text += f'Écart-type: {np.std(sizes):.1f}'

    axes[1].text(1.3, 0.5, stats_text, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    return fig

def create_comparison_grid(images, labels_list, method_names, n_cols=3):
    """
    Crée une grille de comparaison pour plusieurs images et méthodes.

    Args:
        images: Liste d'images RGB
        labels_list: Liste de listes de labels (une liste par image)
                    Exemple: [[labels_img1_method1, labels_img1_method2], ...]
        method_names: Liste des noms de méthodes
        n_cols: Nombre de colonnes (default: 3)

    Returns:
        fig: Figure matplotlib

    Example:
        >>> images = [image1, image2]
        >>> labels_list = [[labels1_m1, labels1_m2], [labels2_m1, labels2_m2]]
        >>> method_names = ['SLIC-100', 'SLIC-200']
        >>> fig = create_comparison_grid(images, labels_list, method_names)
    """
    n_images = len(images)
    n_methods = len(method_names)
    n_rows = n_images * n_methods

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for img_idx in range(n_images):
        image = images[img_idx]
        if image.max() > 1.0:
            image = image / 255.0

        for method_idx, method_name in enumerate(method_names):
            row = img_idx * n_methods + method_idx
            labels = labels_list[img_idx][method_idx]
            n_sp = len(np.unique(labels))

            axes[row, 0].imshow(image)
            axes[row, 0].set_title(f'Image {img_idx+1} - {method_name}',
                                  fontweight='bold')
            axes[row, 0].axis('off')

            colored = skcolor.label2rgb(labels, image, kind='avg')
            axes[row, 1].imshow(colored)
            axes[row, 1].set_title(f'{n_sp} superpixels')
            axes[row, 1].axis('off')

            marked = segmentation.mark_boundaries(image, labels,
                                                 color=(1, 1, 0))
            axes[row, 2].imshow(marked)
            axes[row, 2].axis('off')

    plt.tight_layout()

    return fig

def plot_metrics_radar(results_dict, metrics_to_plot=None):
    """
    Affiche un graphique radar pour comparer les métriques.

    Args:
        results_dict: Dictionnaire {méthode: metrics_dict}
        metrics_to_plot: Liste des métriques à afficher

    Returns:
        fig: Figure matplotlib

    Example:
        >>> results = {
        ...     'SLIC-100': {'br': 0.75, 'compactness': 0.82, 'asa': 0.88},
        ...     'SLIC-200': {'br': 0.82, 'compactness': 0.79, 'asa': 0.91}
        ... }
        >>> fig = plot_metrics_radar(results)
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['boundary_recall', 'asa', 'compactness', 'regularity']

    available_metrics = []
    for metric in metrics_to_plot:
        if any(metric in results for results in results_dict.values()):
            available_metrics.append(metric)

    if not available_metrics:
        print("Aucune métrique disponible")
        return None

    n_metrics = len(available_metrics)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    colors = plt.cm.Set2(np.linspace(0, 1, len(results_dict)))

    for (method_name, metrics), color in zip(results_dict.items(), colors):
        values = []
        for metric in available_metrics:
            val = metrics.get(metric, 0)

            if metric == 'under_segmentation_error':
                val = 1 - min(val, 1)
            values.append(val)

        values += values[:1]  

        ax.plot(angles, values, 'o-', linewidth=2, label=method_name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    readable_names = {
        'boundary_recall': 'BR',
        'under_segmentation_error': 'UE (inv)',
        'asa': 'ASA',
        'compactness': 'Compacité',
        'regularity': 'Régularité'
    }

    labels = [readable_names.get(m, m) for m in available_metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Comparaison des métriques', fontweight='bold', pad=20)
    plt.tight_layout()

    return fig

def save_visualization(fig, filepath, dpi=150):
    """
    Sauvegarde une figure matplotlib.

    Args:
        fig: Figure matplotlib
        filepath: Chemin du fichier de sortie
        dpi: Résolution (default: 150)

    Example:
        >>> fig = visualize_segmentation(image, labels)
        >>> save_visualization(fig, 'results/segmentation.png')
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Visualisation sauvegardée: {filepath}")

def plot_convergence(errors, title="Convergence SLIC"):
    """
    Affiche la courbe de convergence de l'algorithme.

    Args:
        errors: Liste des erreurs résiduelles par itération
        title: Titre du graphique

    Returns:
        fig: Figure matplotlib

    Example:
        >>> errors = [0.5, 0.3, 0.15, 0.08, 0.05, 0.03]
        >>> fig = plot_convergence(errors)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = range(1, len(errors) + 1)
    ax.plot(iterations, errors, marker='o', linewidth=2,
           markersize=8, color='steelblue',
           markeredgecolor='darkblue', markeredgewidth=1.5)

    ax.set_xlabel('Itération', fontsize=12)
    ax.set_ylabel('Erreur résiduelle', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')

    for i, err in zip(iterations, errors):
        ax.annotate(f'{err:.4f}', (i, err),
                   textcoords="offset points",
                   xytext=(0, 10), ha='center',
                   fontsize=9)

    plt.tight_layout()

    return fig

def visualize_superpixel_overlay(image, labels, alpha=0.5, cmap='tab20'):
    """
    Crée une superposition semi-transparente des superpixels sur l'image.

    Args:
        image: Image RGB (H, W, 3)
        labels: Labels des superpixels (H, W)
        alpha: Transparence de la superposition (0-1)
        cmap: Colormap pour les superpixels

    Returns:
        fig: Figure matplotlib

    Example:
        >>> fig = visualize_superpixel_overlay(image, labels, alpha=0.3)
    """
    if image.max() > 1.0:
        image = image / 255.0

    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)

    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)

    overlay = np.zeros_like(image)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = cmap(i / n_labels)[:3]
        overlay[mask] = color

    blended = (1 - alpha) * image + alpha * overlay

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title('Image originale')
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title('Superpixels')
    axes[1].axis('off')

    axes[2].imshow(blended)
    axes[2].set_title(f'Superposition (α={alpha})')
    axes[2].axis('off')

    plt.tight_layout()

    return fig

def create_summary_figure(image, labels, metrics, title="Résumé"):
    """
    Crée une figure récapitulative avec image, segmentation et métriques.

    Args:
        image: Image RGB (H, W, 3)
        labels: Labels des superpixels (H, W)
        metrics: Dictionnaire de métriques
        title: Titre général

    Returns:
        fig: Figure matplotlib

    Example:
        >>> metrics = compute_all_metrics(labels, gt)
        >>> fig = create_summary_figure(image, labels, metrics)
    """
    if image.max() > 1.0:
        image = image / 255.0

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title('Image originale', fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    colored = skcolor.label2rgb(labels, image, kind='avg')
    ax2.imshow(colored)
    n_sp = len(np.unique(labels))
    ax2.set_title(f'Couleurs moyennes\n({n_sp} superpixels)', fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    marked = segmentation.mark_boundaries(image, labels, color=(1, 1, 0))
    ax3.imshow(marked)
    ax3.set_title('Avec contours', fontweight='bold')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[1, 0])
    sizes = [np.sum(labels == l) for l in np.unique(labels)]
    ax4.hist(sizes, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Taille (pixels)')
    ax4.set_ylabel('Fréquence')
    ax4.set_title('Distribution des tailles', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis('off')

    metrics_text = "MÉTRIQUES\n" + "="*40 + "\n"
    readable_names = {
        'n_superpixels': 'Nombre de superpixels',
        'compactness': 'Compacité',
        'regularity': 'Régularité',
        'boundary_recall': 'Boundary Recall',
        'under_segmentation_error': 'Under-segmentation Error',
        'asa': 'ASA'
    }

    for key, value in metrics.items():
        name = readable_names.get(key, key)
        if isinstance(value, float):
            metrics_text += f"{name}: {value:.4f}\n"
        else:
            metrics_text += f"{name}: {value}\n"

    ax5.text(0.1, 0.5, metrics_text, transform=ax5.transAxes,
            fontsize=11, verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(title, fontsize=16, fontweight='bold')

    return fig
    """
    Example:
        >>> from src.methods.slic.slic_original import SLIC
        >>> slic = SLIC(n_segments=200)
        >>> labels = slic.fit(image)
        >>> fig = visualize_segmentation(image, labels)
        >>> plt.show()
    """

    if image.max() > 1.0:
        image = image / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title('Image originale')
    axes[0].axis('off')

    colored = skcolor.label2rgb(labels, image, kind='avg', bg_label=-1)
    axes[1].imshow(colored)
    n_sp = len(np.unique(labels))
    axes[1].set_title(f'Couleurs moyennes\n({n_sp} superpixels)')
    axes[1].axis('off')

    if show_boundaries:
        marked = segmentation.mark_boundaries(
            image, labels,
            color=boundary_color,
            mode='thick',
            background_label=-1
        )
        axes[2].imshow(marked)
        axes[2].set_title('Avec contours')
    else:
        axes[2].imshow(image)
        axes[2].set_title('Image originale')
    axes[2].axis('off')

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig

def compare_segmentations(image, labels_dict, titles=None):
    """
    Compare plusieurs segmentations côte à côte.

    Args:
        image: Image originale RGB (H, W, 3)
        labels_dict: Dictionnaire {nom_méthode: labels}
                    Exemple: {'SLIC-100': labels1, 'SLIC-200': labels2}
        titles: Liste de titres personnalisés (optionnel)

    Returns:
        fig: Figure matplotlib

    Example:
        >>> labels_dict = {
        ...     'SLIC-100': slic100.fit(image),
        ...     'SLIC-200': slic200.fit(image)
        ... }
        >>> fig = compare_segmentations(image, labels_dict)
        >>> plt.show()
    """

    if image.max() > 1.0:
        image = image / 255.0

    n_methods = len(labels_dict)

    fig, axes = plt.subplots(2, n_methods + 1, figsize=(5*(n_methods+1), 10))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Image originale', fontweight='bold')
    axes[0, 0].axis('off')

    axes[1, 0].imshow(image)
    axes[1, 0].set_title('Image originale', fontweight='bold')
    axes[1, 0].axis('off')

    for i, (method_name, labels) in enumerate(labels_dict.items(), start=1):
        n_sp = len(np.unique(labels))

        if titles and len(titles) >= i:
            display_title = titles[i-1]
        else:
            display_title = method_name

        colored = skcolor.label2rgb(labels, image, kind='avg', bg_label=-1)
        axes[0, i].imshow(colored)
        axes[0, i].set_title(f'{display_title}\n({n_sp} superpixels)', 
                            fontweight='bold')
        axes[0, i].axis('off')

        marked = segmentation.mark_boundaries(
            image, labels,
            color=(1, 1, 0),
            mode='thick',
            background_label=-1
        )
        axes[1, i].imshow(marked)
        axes[1, i].axis('off')

    plt.tight_layout()

    return fig

def plot_metrics_comparison(results_dict, metrics_to_plot=None):
    """
    Affiche un graphique comparatif des métriques entre plusieurs méthodes.

    Args:
        results_dict: Dictionnaire {méthode: metrics_dict}
                     Exemple: {'SLIC-100': {'br': 0.8, 'ue': 0.2, ...}, ...}
        metrics_to_plot: Liste des métriques à afficher (None = toutes)
                        Exemple: ['boundary_recall', 'compactness']

    Returns:
        fig: Figure matplotlib

    Example:
        >>> results = {
        ...     'SLIC-100': {'boundary_recall': 0.75, 'compactness': 0.82},
        ...     'SLIC-200': {'boundary_recall': 0.82, 'compactness': 0.79}
        ... }
        >>> fig = plot_metrics_comparison(results)
        >>> plt.show()
    """
    if metrics_to_plot is None:

        metrics_to_plot = ['boundary_recall', 'under_segmentation_error',
                          'asa', 'compactness', 'regularity']

    available_metrics = []
    for metric in metrics_to_plot:
        if any(metric in results for results in results_dict.values()):
            available_metrics.append(metric)

    if not available_metrics:
        print("Aucune métrique disponible à afficher")
        return None

    n_metrics = len(available_metrics)
    methods = list(results_dict.keys())

    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    readable_names = {
        'boundary_recall': 'Boundary Recall',
        'under_segmentation_error': 'Under-segmentation Error',
        'asa': 'ASA',
        'compactness': 'Compacité',
        'regularity': 'Régularité',
        'n_superpixels': 'Nombre de superpixels'
    }

    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

    for i, metric in enumerate(available_metrics):
        values = []
        for method in methods:
            if metric in results_dict[method]:
                values.append(results_dict[method][metric])
            else:
                values.append(0)

        bars = axes[i].bar(range(len(methods)), values, color=colors, 
                          alpha=0.7, edgecolor='black', linewidth=1.5)

        axes[i].set_xticks(range(len(methods)))
        axes[i].set_xticklabels(methods, rotation=45, ha='right')
        axes[i].set_ylabel('Score')
        title = readable_names.get(metric, metric.replace('_', ' ').title())
        axes[i].set_title(title, fontweight='bold')
        axes[i].grid(axis='y', alpha=0.3, linestyle='--')

        if metric in ['boundary_recall', 'asa', 'compactness', 'regularity']:
            axes[i].set_ylim([0, 1.05])
        elif metric == 'under_segmentation_error':
            axes[i].set_ylim([0, max(values) * 1.2 if values else 1])

        for j, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    return fig

def plot_parameter_study(param_name, param_values, metrics_list,
                        metric_names=None):
    """
    Affiche l'effet d'un paramètre sur plusieurs métriques.

    Args:
        param_name: Nom du paramètre étudié (ex: 'n_segments', 'compactness')
        param_values: Liste des valeurs du paramètre testées
        metrics_list: Liste de dictionnaires de métriques (un par valeur)
        metric_names: Liste des métriques à afficher (None = toutes)

    Returns:
        fig: Figure matplotlib
    """
    if metric_names is None:
        metric_names = list(metrics_list[0].keys())

    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.Set3(np.linspace(0, 1, n_metrics))

    for i, (metric_name, color) in enumerate(zip(metric_names, colors)):
        values = [metrics[metric_name] for metrics in metrics_list]
        axes[i].plot(param_values, values, color=color, marker='o', linewidth=2, markersize=8)
        axes[i].set_xlabel(param_name)
        axes[i].set_ylabel(metric_name)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    return fig
