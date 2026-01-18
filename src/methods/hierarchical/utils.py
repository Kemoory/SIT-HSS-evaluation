import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import segmentation, color as skcolor


def visualize_superpixels(image, labels, mark_boundaries_color=(1, 1, 0), 
                          mode='thick', alpha=0.3):
    """
    Visualise les superpixels avec leurs contours sur l'image.
    
    Args:
        image: Image originale RGB (H, W, 3) avec valeurs [0, 255] ou [0, 1]
        labels: Labels des superpixels (H, W)
        mark_boundaries_color: Couleur des contours RGB (default: jaune)
        mode: 'thick' ou 'thin' pour l'épaisseur des contours
        alpha: Transparence des contours (0=transparent, 1=opaque)
    
    Returns:
        marked_image: Image avec contours marqués (H, W, 3)
    """
    # Normaliser l'image si nécessaire
    if image.max() > 1.0:
        image = image / 255.0
    
    # Utiliser skimage pour colorer les régions
    colored = skcolor.label2rgb(labels, image, bg_label=-1)
    
    return colored


def get_average_colors(image, labels):
    """
    Calcule la couleur moyenne de chaque superpixel.
    
    Args:
        image: Image RGB (H, W, 3) avec valeurs [0, 255] ou [0, 1]
        labels: Labels des superpixels (H, W)
    
    Returns:
        colored_image: Image avec couleurs moyennes par superpixel (H, W, 3)
        color_dict: Dictionnaire {label: mean_color}
    """
    if image.max() > 1.0:
        image = image / 255.0
    
    colored = np.zeros_like(image)
    color_dict = {}
    
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        mask = (labels == label)
        mean_color = image[mask].mean(axis=0)
        colored[mask] = mean_color
        color_dict[label] = mean_color
    
    return colored, color_dict


def compute_superpixel_statistics(image, labels):
    """
    Calcule des statistiques détaillées sur les superpixels.
    
    Args:
        image: Image RGB (H, W, 3)
        labels: Labels des superpixels (H, W)
    
    Returns:
        stats: Dictionnaire avec statistiques globales
        per_superpixel: Liste de dictionnaires avec stats par superpixel
    """
    unique_labels = np.unique(labels)
    n_segments = len(unique_labels)
    
    # Statistiques par superpixel
    sizes = []
    mean_colors = []
    std_colors = []
    positions = []
    
    per_superpixel = []
    
    for label in unique_labels:
        mask = (labels == label)
        size = np.sum(mask)
        sizes.append(size)
        
        # Statistiques de couleur
        pixels = image[mask]
        mean_color = pixels.mean(axis=0)
        std_color = pixels.std(axis=0)
        mean_colors.append(mean_color)
        std_colors.append(std_color)
        
        # Position moyenne
        y_coords, x_coords = np.where(mask)
        mean_pos = [np.mean(x_coords), np.mean(y_coords)]
        positions.append(mean_pos)
        
        per_superpixel.append({
            'label': int(label),
            'size': int(size),
            'mean_color': mean_color,
            'std_color': std_color,
            'position': mean_pos
        })
    
    # Statistiques globales
    stats = {
        'n_superpixels': n_segments,
        'mean_size': np.mean(sizes),
        'std_size': np.std(sizes),
        'min_size': np.min(sizes),
        'max_size': np.max(sizes),
        'median_size': np.median(sizes),
        'total_pixels': image.shape[0] * image.shape[1]
    }
    
    return stats, per_superpixel


def analyze_compactness(labels):
    """
    Analyse la compacité des superpixels.
    Compacité = 4π × aire / périmètre²
    (1.0 = cercle parfait, < 1.0 = forme irrégulière)
    
    Args:
        labels: Labels des superpixels (H, W)
    
    Returns:
        compactness_scores: Array avec scores de compacité par superpixel
        mean_compactness: Compacité moyenne
    """
    from scipy import ndimage
    
    unique_labels = np.unique(labels)
    compactness_scores = []
    
    for label in unique_labels:
        mask = (labels == label).astype(np.uint8)
        
        # Calculer l'aire
        area = np.sum(mask)
        
        # Calculer le périmètre (approximation par érosion)
        struct = ndimage.generate_binary_structure(2, 1)
        eroded = ndimage.binary_erosion(mask, struct)
        perimeter = np.sum(mask) - np.sum(eroded)
        
        if perimeter > 0 and area > 0:
            # Formule de compacité: 4π × aire / périmètre²
            compactness = 4 * np.pi * area / (perimeter ** 2)
            # Limiter à 1.0 (les erreurs numériques peuvent donner > 1)
            compactness = min(compactness, 1.0)
            compactness_scores.append(compactness)
    
    compactness_scores = np.array(compactness_scores)
    mean_compactness = np.mean(compactness_scores) if len(compactness_scores) > 0 else 0.0
    
    return compactness_scores, mean_compactness


def analyze_regularity(labels):
    """
    Analyse la régularité de la taille des superpixels.
    Un coefficient de variation faible indique des tailles uniformes.
    
    Args:
        labels: Labels des superpixels (H, W)
    
    Returns:
        regularity_score: Score de régularité [0, 1]
        cv: Coefficient de variation
        sizes: Array des tailles de chaque superpixel
    """
    unique_labels = np.unique(labels)
    sizes = []
    
    for label in unique_labels:
        sizes.append(np.sum(labels == label))
    
    sizes = np.array(sizes)
    mean_size = np.mean(sizes)
    std_size = np.std(sizes)
    
    # Coefficient de variation
    cv = std_size / mean_size if mean_size > 0 else 0
    
    # Score de régularité (inversé, normalisé)
    regularity_score = 1.0 / (1.0 + cv)
    
    return regularity_score, cv, sizes


def compare_parameters(image, n_segments_list, compactness_list, max_iter=10):
    """
    Compare visuellement différentes combinaisons de paramètres SLIC.
    
    Args:
        image: Image RGB (H, W, 3)
        n_segments_list: Liste de valeurs pour n_segments
        compactness_list: Liste de valeurs pour compactness
        max_iter: Nombre d'itérations pour SLIC
    
    Returns:
        fig: Figure matplotlib avec comparaison
        results: Dictionnaire avec les labels pour chaque configuration
    """
    from src.methods.slic.slic_original import SLIC
    
    n_rows = len(n_segments_list)
    n_cols = len(compactness_list)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    # Gérer le cas d'une seule ligne ou colonne
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    results = {}
    
    for i, n_seg in enumerate(n_segments_list):
        for j, comp in enumerate(compactness_list):
            print(f"Traitement: n_segments={n_seg}, compactness={comp}")
            
            slic = SLIC(n_segments=n_seg, compactness=comp, max_iter=max_iter)
            labels = slic.fit(image)
            
            # Sauvegarder les résultats
            key = f"n{n_seg}_m{comp}"
            results[key] = labels
            
            # Visualiser
            marked = visualize_superpixels(image, labels)
            
            axes[i, j].imshow(marked)
            axes[i, j].set_title(f'n={n_seg}, m={comp}\n({len(np.unique(labels))} SP)')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    
    return fig, results


def plot_size_distribution(labels, bins=30):
    """
    Affiche la distribution des tailles de superpixels.
    
    Args:
        labels: Labels des superpixels (H, W)
        bins: Nombre de bins pour l'histogramme
    
    Returns:
        fig: Figure matplotlib
    """
    unique_labels = np.unique(labels)
    sizes = [np.sum(labels == label) for label in unique_labels]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogramme
    ax1.hist(sizes, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(sizes), color='red', linestyle='--', linewidth=2,
               label=f'Moyenne: {np.mean(sizes):.1f}')
    ax1.axvline(np.median(sizes), color='green', linestyle='--', linewidth=2,
               label=f'Médiane: {np.median(sizes):.1f}')
    ax1.set_xlabel('Taille (nombre de pixels)')
    ax1.set_ylabel('Fréquence')
    ax1.set_title('Distribution des tailles de superpixels')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Box plot
    bp = ax2.boxplot(sizes, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)
    ax2.set_ylabel('Taille (nombre de pixels)')
    ax2.set_title('Box plot des tailles')
    ax2.grid(axis='y', alpha=0.3)
    
    # Ajouter des statistiques
    stats_text = f'Statistiques:\n'
    stats_text += f'N superpixels: {len(sizes)}\n'
    stats_text += f'Min: {np.min(sizes)}\n'
    stats_text += f'Max: {np.max(sizes)}\n'
    stats_text += f'Moyenne: {np.mean(sizes):.1f}\n'
    stats_text += f'Médiane: {np.median(sizes):.1f}\n'
    stats_text += f'Écart-type: {np.std(sizes):.1f}'
    
    ax2.text(1.3, 0.5, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    return fig


def plot_compactness_distribution(labels, bins=30):
    """
    Affiche la distribution de la compacité des superpixels.
    
    Args:
        labels: Labels des superpixels (H, W)
        bins: Nombre de bins pour l'histogramme
    
    Returns:
        fig: Figure matplotlib
    """
    compactness_scores, mean_comp = analyze_compactness(labels)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(compactness_scores, bins=bins, color='forestgreen', alpha=0.7, 
           edgecolor='black', range=(0, 1))
    ax.axvline(mean_comp, color='red', linestyle='--', linewidth=2,
              label=f'Moyenne: {mean_comp:.3f}')
    ax.axvline(np.median(compactness_scores), color='orange', linestyle='--', 
              linewidth=2, label=f'Médiane: {np.median(compactness_scores):.3f}')
    
    ax.set_xlabel('Score de compacité')
    ax.set_ylabel('Fréquence')
    ax.set_title('Distribution de la compacité des superpixels\n(1.0 = cercle parfait)')
    ax.set_xlim([0, 1])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Statistiques
    stats_text = f'Statistiques:\n'
    stats_text += f'Moyenne: {mean_comp:.4f}\n'
    stats_text += f'Médiane: {np.median(compactness_scores):.4f}\n'
    stats_text += f'Écart-type: {np.std(compactness_scores):.4f}\n'
    stats_text += f'Min/Max: {np.min(compactness_scores):.4f} / {np.max(compactness_scores):.4f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    
    return fig


def create_superpixel_map(labels, cmap='tab20'):
    """
    Crée une carte colorée des superpixels.
    
    Args:
        labels: Labels des superpixels (H, W)
        cmap: Colormap à utiliser
    
    Returns:
        colored_map: Image colorée (H, W, 3)
    """
    from matplotlib import cm
    
    # Obtenir la colormap
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    
    # Normaliser les labels
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    
    # Créer une correspondance label -> couleur
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Créer l'image colorée
    colored_map = np.zeros((*labels.shape, 3))
    
    for label in unique_labels:
        mask = (labels == label)
        color = cmap(label_to_idx[label] / n_labels)[:3]
        colored_map[mask] = color
    
    return colored_map


def save_segmentation(labels, filename):
    """
    Sauvegarde la segmentation dans un fichier NumPy.
    
    Args:
        labels: Labels des superpixels (H, W)
        filename: Chemin du fichier de sortie (.npy)
    """
    np.save(filename, labels)
    print(f"Segmentation sauvegardée: {filename}")


def load_segmentation(filename):
    """
    Charge une segmentation depuis un fichier NumPy.
    
    Args:
        filename: Chemin du fichier (.npy)
    
    Returns:
        labels: Labels des superpixels (H, W)
    """
    labels = np.load(filename)
    print(f"Segmentation chargée: {filename}")
    return labels


def export_superpixels_to_dict(image, labels):
    """
    Exporte les superpixels dans un dictionnaire structuré.
    
    Args:
        image: Image RGB (H, W, 3)
        labels: Labels des superpixels (H, W)
    
    Returns:
        superpixels_dict: Dictionnaire avec infos complètes par superpixel
    """
    unique_labels = np.unique(labels)
    superpixels_dict = {}
    
    for label in unique_labels:
        mask = (labels == label)
        y_coords, x_coords = np.where(mask)
        
        # Pixels du superpixel
        pixels = image[mask]
        
        superpixels_dict[int(label)] = {
            'size': int(np.sum(mask)),
            'pixels': list(zip(x_coords.tolist(), y_coords.tolist())),
            'mean_color': pixels.mean(axis=0).tolist(),
            'std_color': pixels.std(axis=0).tolist(),
            'centroid': [float(np.mean(x_coords)), float(np.mean(y_coords))],
            'bbox': {
                'x_min': int(np.min(x_coords)),
                'x_max': int(np.max(x_coords)),
                'y_min': int(np.min(y_coords)),
                'y_max': int(np.max(y_coords))
            }
        }
    
    return superpixels_dict


def visualize_individual_superpixel(image, labels, target_label, highlight_color=(1, 0, 0)):
    """
    Visualise un superpixel individuel en le mettant en évidence.
    
    Args:
        image: Image RGB (H, W, 3)
        labels: Labels des superpixels (H, W)
        target_label: Label du superpixel à visualiser
        highlight_color: Couleur de mise en évidence RGB
    
    Returns:
        fig: Figure matplotlib
    """
    if image.max() > 1.0:
        image = image / 255.0
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Image originale
    axes[0].imshow(image)
    axes[0].set_title('Image originale')
    axes[0].axis('off')
    
    # Masque du superpixel
    mask = (labels == target_label)
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f'Superpixel {target_label}\n(taille: {np.sum(mask)} pixels)')
    axes[1].axis('off')
    
    # Superpixel mis en évidence
    highlighted = image.copy()
    highlighted[mask] = highlight_color
    axes[2].imshow(highlighted)
    axes[2].set_title('Superpixel mis en évidence')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    return fig


def compute_adjacency_matrix(labels):
    """
    Calcule la matrice d'adjacence des superpixels.
    
    Args:
        labels: Labels des superpixels (H, W)
    
    Returns:
        adjacency: Matrice d'adjacence (N, N) où N = nombre de superpixels
        label_list: Liste des labels correspondant aux indices
    """
    unique_labels = np.unique(labels)
    n = len(unique_labels)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    adjacency = np.zeros((n, n), dtype=bool)
    
    h, w = labels.shape
    
    # Parcourir tous les pixels
    for y in range(h):
        for x in range(w):
            current_label = labels[y, x]
            current_idx = label_to_idx[current_label]
            
            # Vérifier les voisins droite et bas
            for dy, dx in [(0, 1), (1, 0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    neighbor_label = labels[ny, nx]
                    if neighbor_label != current_label:
                        neighbor_idx = label_to_idx[neighbor_label]
                        adjacency[current_idx, neighbor_idx] = True
                        adjacency[neighbor_idx, current_idx] = True
    
    return adjacency, unique_labels.tolist()


def print_summary(image, labels):
    """
    Affiche un résumé de la segmentation.
    
    Args:
        image: Image RGB (H, W, 3)
        labels: Labels des superpixels (H, W)
    """
    unique_labels = np.unique(labels)
    sizes = [np.sum(labels == label) for label in unique_labels]
    _, mean_comp = analyze_compactness(labels)
    reg_score, cv, _ = analyze_regularity(labels)
    
    print("=" * 60)
    print("RÉSUMÉ DE LA SEGMENTATION")
    print("=" * 60)
    print(f"Nombre de superpixels      : {len(unique_labels)}")
    print(f"Taille moyenne             : {np.mean(sizes):.1f} pixels")
    print(f"Écart-type des tailles     : {np.std(sizes):.1f} pixels")
    print(f"Taille min/max             : {np.min(sizes)} / {np.max(sizes)} pixels")
    print(f"Compacité moyenne          : {mean_comp:.4f}")
    print(f"Score de régularité        : {reg_score:.4f}")
    print(f"Coefficient de variation   : {cv:.4f}")
    print("=" * 60)

    # Marquer les contours
    marked = segmentation.mark_boundaries(
        image, labels,
        color=(0, 1, 0),
        mode='outer',
        background_label=-1
    )
    
    return marked
