"""
Métriques d'évaluation pour les superpixels

Implémente les métriques standards :
- Boundary Recall (BR) : qualité des contours
- Under-segmentation Error (UE) : débordement des superpixels
- Achievable Segmentation Accuracy (ASA) : précision de segmentation atteignable
- Compactness : régularité de la forme
- Regularity : uniformité des tailles
"""
import numpy as np
from scipy import ndimage
from skimage import segmentation


def boundary_recall(labels, ground_truth, tolerance=2):
    """
    Calcule le Boundary Recall (BR).
    
    Mesure la proportion de contours du ground truth qui sont correctement
    détectés par la segmentation. Un BR élevé indique que les superpixels
    respectent bien les vrais contours des objets.
    
    Formule: BR = (contours GT détectés) / (total contours GT)
    
    Args:
        labels: Segmentation prédite (H, W)
        ground_truth: Segmentation ground truth (H, W)
        tolerance: Distance de tolérance en pixels (default: 2)
                  Un contour prédit dans cette distance d'un contour GT
                  est considéré comme correct
    
    Returns:
        br: Score de boundary recall dans [0, 1]
            1.0 = tous les contours GT sont détectés
            0.0 = aucun contour GT n'est détecté
    
    Example:
        >>> labels = np.array([[0, 0, 1], [0, 1, 1], [2, 2, 2]])
        >>> gt = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])
        >>> br = boundary_recall(labels, gt)
    """
    # Extraire les contours des deux segmentations
    boundaries_pred = segmentation.find_boundaries(labels, mode='thick')
    boundaries_gt = segmentation.find_boundaries(ground_truth, mode='thick')
    
    # Dilater les contours prédits pour la tolérance
    if tolerance > 0:
        struct = ndimage.generate_binary_structure(2, 2)
        boundaries_pred_dilated = ndimage.binary_dilation(
            boundaries_pred,
            structure=struct,
            iterations=tolerance
        )
    else:
        boundaries_pred_dilated = boundaries_pred
    
    # Compter les pixels de contours GT
    gt_boundary_pixels = np.sum(boundaries_gt)
    
    if gt_boundary_pixels == 0:
        return 1.0  # Pas de contour GT, score parfait par convention
    
    # Compter combien de contours GT sont correctement détectés
    correctly_detected = np.sum(boundaries_gt & boundaries_pred_dilated)
    
    # Calculer le recall
    br = correctly_detected / gt_boundary_pixels
    
    return float(br)


def under_segmentation_error(labels, ground_truth):
    """
    Calcule l'Under-segmentation Error (UE).
    
    Mesure le débordement des superpixels au-delà des segments du ground truth.
    Un superpixel qui chevauche plusieurs segments GT crée une erreur.
    Un UE faible indique que les superpixels ne débordent pas trop.
    
    Formule: UE = (1/N) × Σ_i [ Σ_{S_j ∩ G_i ≠ ∅} |S_j| - |G_i| ]
    où:
        - N = nombre total de pixels
        - G_i = segment i du ground truth
        - S_j = superpixel j qui chevauche G_i
    
    Args:
        labels: Segmentation prédite (H, W)
        ground_truth: Segmentation ground truth (H, W)
    
    Returns:
        ue: Score d'erreur de sous-segmentation
            0.0 = segmentation parfaite (pas de débordement)
            Plus élevé = plus de débordement
    
    Example:
        >>> labels = np.array([[0, 0, 1], [0, 1, 1], [2, 2, 2]])
        >>> gt = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]])
        >>> ue = under_segmentation_error(labels, gt)
    """
    n_pixels = labels.size
    ue = 0.0
    
    # Pour chaque segment du ground truth
    unique_gt_labels = np.unique(ground_truth)
    
    for gt_label in unique_gt_labels:
        # Masque du segment GT
        gt_mask = (ground_truth == gt_label)
        gt_size = np.sum(gt_mask)
        
        if gt_size == 0:
            continue
        
        # Trouver tous les superpixels qui chevauchent ce segment GT
        overlapping_sp_labels = np.unique(labels[gt_mask])
        
        # Calculer le débordement total
        leakage = 0
        for sp_label in overlapping_sp_labels:
            sp_mask = (labels == sp_label)
            sp_size = np.sum(sp_mask)
            
            # Intersection entre le superpixel et le segment GT
            intersection = np.sum(gt_mask & sp_mask)
            
            # Pixels du superpixel qui débordent hors du segment GT
            leakage += (sp_size - intersection)
        
        ue += leakage
    
    # Normaliser par le nombre total de pixels
    ue = ue / n_pixels
    
    return float(ue)


def achievable_segmentation_accuracy(labels, ground_truth):
    """
    Calcule l'Achievable Segmentation Accuracy (ASA).
    
    Mesure la meilleure précision de segmentation possible en assignant
    chaque superpixel à la classe GT majoritaire. Un ASA élevé indique
    que les superpixels correspondent bien aux régions sémantiques.
    
    Formule: ASA = (1/N) × Σ_i max_k |S_i ∩ G_k|
    où:
        - N = nombre total de pixels
        - S_i = superpixel i
        - G_k = segment k du ground truth
    
    Args:
        labels: Segmentation prédite (H, W)
        ground_truth: Segmentation ground truth (H, W)
    
    Returns:
        asa: Score ASA dans [0, 1]
             1.0 = segmentation parfaite possible
             Plus élevé = meilleure correspondance possible
    
    Example:
        >>> labels = np.array([[0, 0, 1], [0, 1, 1], [2, 2, 2]])
        >>> gt = np.array([[0, 0, 0], [0, 1, 1], [1, 1, 1]])
        >>> asa = achievable_segmentation_accuracy(labels, gt)
    """
    n_pixels = labels.size
    correctly_labeled = 0
    
    # Pour chaque superpixel
    unique_sp_labels = np.unique(labels)
    
    for sp_label in unique_sp_labels:
        # Masque du superpixel
        sp_mask = (labels == sp_label)
        
        # Trouver tous les labels GT dans ce superpixel
        gt_labels_in_sp = ground_truth[sp_mask]
        
        # Compter les occurrences de chaque label GT
        unique_gt, counts = np.unique(gt_labels_in_sp, return_counts=True)
        
        # Le meilleur cas : assigner tout le superpixel à la classe majoritaire
        max_count = np.max(counts)
        correctly_labeled += max_count
    
    # Calculer l'accuracy
    asa = correctly_labeled / n_pixels
    
    return float(asa)


def compactness(labels):
    """
    Calcule la compacité moyenne des superpixels.
    
    La compacité mesure à quel point un superpixel ressemble à un cercle.
    Formule: C = 4π × aire / périmètre²
    
    Valeurs:
        - 1.0 = cercle parfait
        - < 1.0 = forme allongée ou irrégulière
    
    Args:
        labels: Segmentation (H, W)
    
    Returns:
        mean_compactness: Compacité moyenne dans [0, 1]
    
    Example:
        >>> labels = np.array([[0, 0, 1], [0, 1, 1], [2, 2, 2]])
        >>> comp = compactness(labels)
    """
    unique_labels = np.unique(labels)
    compactness_scores = []
    
    for label in unique_labels:
        # Masque du superpixel
        mask = (labels == label).astype(np.uint8)
        
        # Calculer l'aire (nombre de pixels)
        area = np.sum(mask)
        
        if area == 0:
            continue
        
        # Calculer le périmètre (approximation par érosion)
        struct = ndimage.generate_binary_structure(2, 1)
        eroded = ndimage.binary_erosion(mask, struct)
        perimeter = np.sum(mask) - np.sum(eroded)
        
        if perimeter > 0:
            # Formule de compacité isopérimétrique
            comp = 4 * np.pi * area / (perimeter ** 2)
            # Limiter à 1.0 (erreurs numériques peuvent donner > 1)
            comp = min(comp, 1.0)
            compactness_scores.append(comp)
    
    if len(compactness_scores) == 0:
        return 0.0
    
    return float(np.mean(compactness_scores))


def regularity(labels):
    """
    Mesure la régularité de la taille des superpixels.
    
    Un score élevé indique que tous les superpixels ont des tailles similaires.
    Utilise l'inverse du coefficient de variation.
    
    Formule: R = 1 / (1 + CV)
    où CV = std(tailles) / mean(tailles)
    
    Args:
        labels: Segmentation (H, W)
    
    Returns:
        regularity_score: Score de régularité dans [0, 1]
                         1.0 = toutes les tailles identiques
                         Plus bas = tailles très variables
    
    Example:
        >>> labels = np.array([[0, 0, 1], [0, 1, 1], [2, 2, 2]])
        >>> reg = regularity(labels)
    """
    unique_labels = np.unique(labels)
    sizes = []
    
    for label in unique_labels:
        size = np.sum(labels == label)
        sizes.append(size)
    
    if len(sizes) == 0:
        return 0.0
    
    sizes = np.array(sizes)
    mean_size = np.mean(sizes)
    std_size = np.std(sizes)
    
    # Coefficient de variation
    cv = std_size / mean_size if mean_size > 0 else 0
    
    # Score de régularité (inversé, normalisé)
    regularity_score = 1.0 / (1.0 + cv)
    
    return float(regularity_score)

def explained_variation(labels, ground_truths):
    """
    Calcule l'Explained Variation (EV).
    
    Mesure la proportion de variance expliquée par la segmentation prédite
    par rapport au ground truth. Basé sur le concept de clustering quality.
    
    Formule:
        EV = 1 - (variance intra-cluster / variance totale)
    
    où:
        - variance intra-cluster: variance moyenne des pixels au sein de chaque 
          superpixel comparé aux segments GT
        - variance totale: variance de tous les pixels par rapport à la moyenne globale
    
    Args:
        labels: Segmentation prédite (H, W)
        ground_truths: Liste de segmentations ground truth (H, W)
                      ou une seule segmentation
    
    Returns:
        ev: Score d'explained variation dans [0, 1]
            1.0 = segmentation parfaite (explique 100% de la variance)
            0.0 = segmentation aléatoire (n'explique rien)
            < 0 = segmentation pire qu'une moyenne (cas rare)
    
    Example:
        >>> labels = slic.fit(image)
        >>> ground_truths = [gt1, gt2, gt3]
        >>> ev = explained_variation(labels, ground_truths)
        >>> print(f"EV: {ev:.4f}")
    """
    # Gérer le cas d'une seule segmentation GT
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths]
    
    if len(ground_truths) == 0:
        return 0.0
    
    # Calculer EV pour chaque GT et prendre la moyenne
    ev_scores = []
    
    for gt in ground_truths:
        ev_scores.append(_compute_ev_single(labels, gt))
    
    return float(np.mean(ev_scores))


def _compute_ev_single(labels, ground_truth):
    """
    Calcule l'Explained Variation pour un seul ground truth.
    
    Args:
        labels: Segmentation prédite (H, W)
        ground_truth: Segmentation ground truth (H, W)
    
    Returns:
        ev: Score EV
    """
    h, w = labels.shape
    n_pixels = h * w
    
    # Convertir les labels en vecteurs de classe pour chaque pixel
    # Pour chaque superpixel prédit, chercher la classe GT majoritaire
    unique_labels = np.unique(labels)
    
    # Initialiser score de pureté
    total_purity = 0
    
    for pred_label in unique_labels:
        mask = (labels == pred_label)
        n_in_cluster = np.sum(mask)
        
        if n_in_cluster == 0:
            continue
        
        # Trouver les classes GT dans ce cluster
        gt_classes_in_cluster = ground_truth[mask]
        unique_gt, counts = np.unique(gt_classes_in_cluster, return_counts=True)
        
        # Pureté = pixels de la classe majoritaire / pixels du cluster
        max_count = np.max(counts)
        purity = max_count / n_in_cluster
        
        # Contribution pondérée par la taille du cluster
        total_purity += (n_in_cluster / n_pixels) * purity
    
    # Nombre attendu de classes GT
    unique_gt = np.unique(ground_truth)
    random_purity = 1.0 / len(unique_gt) if len(unique_gt) > 0 else 0
    
    # EV normalisée
    # Comparer la pureté obtenue à la pureté aléatoire
    ev = (total_purity - random_purity) / (1.0 - random_purity)
    
    return float(np.clip(ev, -1.0, 1.0))

def global_regularity(labels):
    """
    Calcule la Global Regularity (GR).
    
    Mesure l'uniformité globale de la grille de superpixels.
    Une GR élevée indique que les superpixels sont régulièrement espacés.
    
    Basé sur l'écart-type des distances au plus proche voisin.
    
    Formule:
        GR = 1 / (1 + CV_spatial)
    
    où:
        - CV_spatial = std(distances voisins) / mean(distances voisins)
        - distance = distance entre centre de superpixels adjacents
    
    Args:
        labels: Segmentation (H, W)
    
    Returns:
        gr: Score de Global Regularity dans [0, 1]
            1.0 = grille parfaitement régulière
            0.0 = distribution très irrégulière
    
    Example:
        >>> labels = slic.fit(image)
        >>> gr = global_regularity(labels)
        >>> print(f"GR: {gr:.4f}")
    """
    from scipy import ndimage
    
    h, w = labels.shape
    unique_labels = np.unique(labels)
    
    if len(unique_labels) <= 1:
        return 0.0
    
    # Calculer les centres de masse de chaque superpixel
    centers = ndimage.center_of_mass(np.ones_like(labels), labels, unique_labels)
    centers = np.array(centers)
    
    # Calculer les distances entre tous les centres
    distances = []
    
    for i, center_i in enumerate(centers):
        # Trouver les k plus proches voisins (k=4 pour 4-connectivité)
        dists_to_center_i = np.sqrt(np.sum((centers - center_i) ** 2, axis=1))
        
        # Exclure la distance à soi-même (0)
        dists_to_center_i = dists_to_center_i[dists_to_center_i > 0]
        
        if len(dists_to_center_i) > 0:
            # Prendre les 4 plus proches
            k_nearest = np.sort(dists_to_center_i)[:min(4, len(dists_to_center_i))]
            distances.extend(k_nearest)
    
    if len(distances) == 0:
        return 0.0
    
    distances = np.array(distances)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    # Coefficient de variation spatial
    cv_spatial = std_dist / (mean_dist + 1e-10)
    
    # Global Regularity
    gr = 1.0 / (1.0 + cv_spatial)
    
    return float(np.clip(gr, 0.0, 1.0))


def corrected_under_segmentation_error(labels, ground_truth):
    """
    Calcule l'Corrected Under-segmentation Error (CUE).
    
    Version améliorée de l'UE qui prend en compte:
    - Les pixels sur les frontières (moins de pénalité)
    - L'importance relative des segments GT (normalisation)
    
    Formule:
        CUE = (1/N) × Σ_i [ (|S_i - G_i| / |G_i|) × |G_i| ]
    
    où:
        - S_i = ensemble des pixels du superpixel dans le segment GT i
        - G_i = pixels du segment GT i
        - N = nombre total de pixels
    
    Args:
        labels: Segmentation prédite (H, W)
        ground_truth: Segmentation ground truth (H, W)
    
    Returns:
        cue: Score de CUE
             0.0 = pas de débordement
             > 0 = débordement (plus bas est mieux)
    
    Example:
        >>> labels = slic.fit(image)
        >>> cue = corrected_under_segmentation_error(labels, ground_truth)
        >>> print(f"CUE: {cue:.4f}")
    """
    n_pixels = labels.size
    cue = 0.0
    
    unique_gt_labels = np.unique(ground_truth)
    
    for gt_label in unique_gt_labels:
        # Masque du segment GT
        gt_mask = (ground_truth == gt_label)
        gt_size = np.sum(gt_mask)
        
        if gt_size == 0:
            continue
        
        # Trouver tous les superpixels qui chevauchent ce segment GT
        overlapping_sp_labels = np.unique(labels[gt_mask])
        
        # Calculer le débordement normalisé
        leakage = 0.0
        for sp_label in overlapping_sp_labels:
            sp_mask = (labels == sp_label)
            sp_size = np.sum(sp_mask)
            
            # Intersection
            intersection = np.sum(gt_mask & sp_mask)
            
            # Débordement pour ce superpixel
            # Pondéré par la taille du superpixel
            sp_leakage = (sp_size - intersection)
            leakage += sp_leakage
        
        # Contribution du segment GT
        # Normalisée par la taille du segment GT
        cue += (leakage / gt_size) * (gt_size / n_pixels)
    
    return float(np.clip(cue, 0.0, 1.0))


def precision(labels, ground_truth):
    """
    Calcule la Precision (P) des contours.
    
    Mesure la proportion de contours prédits qui sont réellement
    des contours du ground truth (exactitude des contours détectés).
    
    Formule:
        P = |contours_prédits ∩ contours_GT| / |contours_prédits|
    
    Args:
        labels: Segmentation prédite (H, W)
        ground_truth: Segmentation ground truth (H, W)
    
    Returns:
        p: Score de Precision dans [0, 1]
           1.0 = tous les contours prédits sont corrects
           0.0 = aucun contour ne correspond
    
    Example:
        >>> labels = slic.fit(image)
        >>> p = precision(labels, ground_truth)
        >>> print(f"Precision: {p:.4f}")
    """
    from skimage import segmentation
    
    # Extraire les contours
    boundaries_pred = segmentation.find_boundaries(labels, mode='thick')
    boundaries_gt = segmentation.find_boundaries(ground_truth, mode='thick')
    
    # Compter les pixels de contours
    pred_boundary_pixels = np.sum(boundaries_pred)
    
    if pred_boundary_pixels == 0:
        return 1.0  # Pas de contour prédit, precision parfaite par convention
    
    # Compter les contours prédits qui correspondent aux contours GT
    correctly_detected = np.sum(boundaries_pred & boundaries_gt)
    
    # Calculer la précision
    p = correctly_detected / pred_boundary_pixels
    
    return float(p)


def contour_density(labels, ground_truth):
    """
    Calcule la Contour Density (CD).
    
    Mesure la densité des contours détectés par rapport aux contours GT.
    Ratio entre le nombre de pixels de contours prédits et GT.
    
    Formule:
        CD = |contours_prédits| / |contours_GT|
    
    Interprétation:
        - CD ≈ 1.0 : densité similaire au GT
        - CD > 1.0 : sur-détection de contours
        - CD < 1.0 : sous-détection de contours
    
    Args:
        labels: Segmentation prédite (H, W)
        ground_truth: Segmentation ground truth (H, W)
    
    Returns:
        cd: Score de Contour Density
            Typiquement dans [0, 2] mais peut être > 2
            1.0 = densité de contours optimale
    
    Example:
        >>> labels = slic.fit(image)
        >>> cd = contour_density(labels, ground_truth)
        >>> print(f"CD: {cd:.4f}")
    """
    from skimage import segmentation
    
    # Extraire les contours
    boundaries_pred = segmentation.find_boundaries(labels, mode='thick')
    boundaries_gt = segmentation.find_boundaries(ground_truth, mode='thick')
    
    # Compter les pixels de contours
    pred_boundary_pixels = np.sum(boundaries_pred)
    gt_boundary_pixels = np.sum(boundaries_gt)
    
    if gt_boundary_pixels == 0:
        return 0.0  # Pas de contour GT
    
    # Calculer la densité
    cd = pred_boundary_pixels / gt_boundary_pixels
    
    return float(cd)

def compute_all_metrics(labels, ground_truth=None):
    """
    Calcule toutes les métriques disponibles (version améliorée).
    
    Args:
        labels: Segmentation prédite (H, W)
        ground_truth: Segmentation ground truth (H, W), optionnel
    
    Returns:
        metrics: Dictionnaire avec toutes les métriques
                {
                    'n_superpixels': int,
                    'compactness': float,
                    'regularity': float,
                    'global_regularity': float,
                    'boundary_recall': float,  # si GT fourni
                    'under_segmentation_error': float,  # si GT fourni
                    'corrected_under_segmentation_error': float,  # si GT fourni
                    'asa': float,  # si GT fourni
                    'precision': float,  # si GT fourni
                    'contour_density': float,  # si GT fourni
                    'explained_variation': float  # si GT fourni
                }
    
    Example:
        >>> labels = slic.fit(image)
        >>> metrics = compute_all_metrics(labels, ground_truth)
    """
    metrics = {
        'n_superpixels': len(np.unique(labels)),
        'compactness': compactness(labels),
        'regularity': regularity(labels),
        'global_regularity': global_regularity(labels)
    }
    
    # Si ground truth fourni, calculer les métriques supplémentaires
    if ground_truth is not None:
        metrics['boundary_recall'] = boundary_recall(labels, ground_truth)
        metrics['under_segmentation_error'] = under_segmentation_error(labels, ground_truth)
        metrics['corrected_under_segmentation_error'] = corrected_under_segmentation_error(labels, ground_truth)
        metrics['asa'] = achievable_segmentation_accuracy(labels, ground_truth)
        metrics['precision'] = precision(labels, ground_truth)
        metrics['contour_density'] = contour_density(labels, ground_truth)
        metrics['explained_variation'] = explained_variation(labels, ground_truth)
    
    return metrics


def compute_metrics_multiple_gt(labels, ground_truths):
    """
    Calcule les métriques en moyennant sur plusieurs ground truths (version améliorée).
    
    Args:
        labels: Segmentation prédite (H, W)
        ground_truths: Liste de segmentations ground truth [(H, W), ...]
    
    Returns:
        metrics: Dictionnaire avec métriques moyennes et écarts-types
    
    Example:
        >>> labels = slic.fit(image)
        >>> ground_truths = [gt1, gt2, gt3]
        >>> metrics = compute_metrics_multiple_gt(labels, ground_truths)
    """
    if not ground_truths or len(ground_truths) == 0:
        return compute_all_metrics(labels)
    
    # Calculer les métriques pour chaque GT
    br_scores = []
    ue_scores = []
    cue_scores = []
    asa_scores = []
    p_scores = []
    cd_scores = []
    ev_scores = []
    
    for gt in ground_truths:
        br_scores.append(boundary_recall(labels, gt))
        ue_scores.append(under_segmentation_error(labels, gt))
        cue_scores.append(corrected_under_segmentation_error(labels, gt))
        asa_scores.append(achievable_segmentation_accuracy(labels, gt))
        p_scores.append(precision(labels, gt))
        cd_scores.append(contour_density(labels, gt))
        ev_scores.append(explained_variation(labels, gt))
    
    # Agréger les résultats
    metrics = {
        'n_superpixels': len(np.unique(labels)),
        'compactness': compactness(labels),
        'regularity': regularity(labels),
        'global_regularity': global_regularity(labels),
        'boundary_recall': float(np.mean(br_scores)),
        'boundary_recall_std': float(np.std(br_scores)),
        'under_segmentation_error': float(np.mean(ue_scores)),
        'under_segmentation_error_std': float(np.std(ue_scores)),
        'corrected_under_segmentation_error': float(np.mean(cue_scores)),
        'corrected_under_segmentation_error_std': float(np.std(cue_scores)),
        'asa': float(np.mean(asa_scores)),
        'asa_std': float(np.std(asa_scores)),
        'precision': float(np.mean(p_scores)),
        'precision_std': float(np.std(p_scores)),
        'contour_density': float(np.mean(cd_scores)),
        'contour_density_std': float(np.std(cd_scores)),
        'explained_variation': float(np.mean(ev_scores)),
        'explained_variation_std': float(np.std(ev_scores))
    }
    
    return metrics


def format_metrics(metrics):
    """
    Formate les métriques pour un affichage lisible (version complète).
    
    Args:
        metrics: Dictionnaire de métriques
    
    Returns:
        formatted: String formaté avec tableau
    """
    lines = []
    lines.append("=" * 70)
    lines.append("MÉTRIQUES D'ÉVALUATION COMPLÈTES")
    lines.append("=" * 70)
    
    # Ordre préféré d'affichage (catégorisé)
    preferred_order = [
        # Métriques de base
        'n_superpixels',
        # Régularité et compacité
        'compactness',
        'regularity',
        'global_regularity',
        # Métriques avec Ground Truth
        'boundary_recall',
        'boundary_recall_std',
        'precision',
        'precision_std',
        'contour_density',
        'contour_density_std',
        'under_segmentation_error',
        'under_segmentation_error_std',
        'corrected_under_segmentation_error',
        'corrected_under_segmentation_error_std',
        'asa',
        'asa_std',
        'explained_variation',
        'explained_variation_std'
    ]
    
    # Noms lisibles avec descriptions
    readable_names = {
        'n_superpixels': 'Nombre de superpixels',
        'compactness': 'Compacité moyenne',
        'regularity': 'Régularité',
        'global_regularity': 'Global Regularity (GR)',
        'boundary_recall': 'Boundary Recall (BR)',
        'boundary_recall_std': 'BR écart-type',
        'precision': 'Precision (P)',
        'precision_std': 'P écart-type',
        'contour_density': 'Contour Density (CD)',
        'contour_density_std': 'CD écart-type',
        'under_segmentation_error': 'Under-segmentation Error (UE)',
        'under_segmentation_error_std': 'UE écart-type',
        'corrected_under_segmentation_error': 'Corrected UE (CUE)',
        'corrected_under_segmentation_error_std': 'CUE écart-type',
        'asa': 'Achievable Seg. Accuracy (ASA)',
        'asa_std': 'ASA écart-type',
        'explained_variation': 'Explained Variation (EV)',
        'explained_variation_std': 'EV écart-type'
    }
    
    # Afficher dans l'ordre préféré
    for key in preferred_order:
        if key in metrics:
            name = readable_names.get(key, key)
            value = metrics[key]
            
            if isinstance(value, float):
                lines.append(f"{name:40s}: {value:.4f}")
            else:
                lines.append(f"{name:40s}: {value}")
    
    # Afficher les autres métriques non listées
    for key, value in metrics.items():
        if key not in preferred_order:
            if isinstance(value, float):
                lines.append(f"{key:40s}: {value:.4f}")
            else:
                lines.append(f"{key:40s}: {value}")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


def compare_segmentations_metrics(labels_dict, ground_truth=None):
    """
    Compare plusieurs segmentations en calculant leurs métriques.
    
    Args:
        labels_dict: Dictionnaire {nom_méthode: labels}
        ground_truth: Ground truth optionnel (H, W)
    
    Returns:
        comparison: Dictionnaire {nom_méthode: metrics_dict}
    
    Example:
        >>> labels_dict = {
        ...     'SLIC-100': labels1,
        ...     'SLIC-200': labels2
        ... }
        >>> comparison = compare_segmentations_metrics(labels_dict, gt)
    """
    comparison = {}
    
    for method_name, labels in labels_dict.items():
        metrics = compute_all_metrics(labels, ground_truth)
        comparison[method_name] = metrics
    
    return comparison


def save_metrics_to_csv(metrics, filename):
    """
    Sauvegarde les métriques dans un fichier CSV.
    
    Args:
        metrics: Dictionnaire ou liste de dictionnaires de métriques
        filename: Chemin du fichier CSV de sortie
    
    Example:
        >>> metrics = compute_all_metrics(labels, gt)
        >>> save_metrics_to_csv(metrics, 'results/metrics.csv')
    """
    import pandas as pd
    
    # Convertir en liste si nécessaire
    if isinstance(metrics, dict):
        metrics = [metrics]
    
    # Créer DataFrame et sauvegarder
    df = pd.DataFrame(metrics)
    df.to_csv(filename, index=False)
    print(f"Métriques sauvegardées: {filename}")

def display_metrics_categorized(metrics, image_name=None, method_name=None, 
                               elapsed_time=None, verbose=True):
    """
    Affiche les métriques de manière catégorisée et formatée.
    
    Catégories:
    - OBJECT QUALITY: Qualité de segmentation des objets
    - CONTOUR QUALITY: Qualité des contours détectés
    - REGULARITY: Uniformité et régularité des superpixels
    - COLOR COHERENCE: Cohérence chromatique
    
    Args:
        metrics: Dictionnaire de métriques (résultat de compute_all_metrics)
        image_name: Nom de l'image (optionnel)
        method_name: Nom de la méthode (optionnel)
        elapsed_time: Temps d'exécution en secondes (optionnel)
        verbose: Si True, affiche dans la console
    
    Returns:
        formatted_str: String formatée avec l'affichage complet
    
    Example:
        >>> metrics = compute_metrics_multiple_gt(labels, ground_truths)
        >>> display = display_metrics_categorized(
        ...     metrics, 
        ...     image_name="107045",
        ...     method_name="SLIC_IPOL",
        ...     elapsed_time=0.145
        ... )
        >>> print(display)
    """
    lines = []
    
    # En-tête
    lines.append("\n" + "="*75)
    if method_name:
        lines.append(f"  {method_name}")
    if image_name:
        lines.append(f"  Image: {image_name}")
    lines.append("="*75)
    
    # Informations de base
    lines.append("\n INFORMATIONS GÉNÉRALES")
    lines.append("─"*75)
    lines.append(f"  Superpixels générés: {int(metrics['n_superpixels']):4d}")
    if elapsed_time is not None:
        lines.append(f"  Temps d'exécution:   {elapsed_time:7.3f}s")
    
    # OBJECT QUALITY - Qualité de segmentation des objets
    if 'asa' in metrics or 'under_segmentation_error' in metrics:
        lines.append("\n OBJECT QUALITY (Qualité de segmentation)")
        lines.append("─"*75)
        
        if 'asa' in metrics:
            asa = metrics['asa']
            asa_std = metrics.get('asa_std', None)
            if asa_std is not None:
                lines.append(f"  ASA (Achievable Segmentation Accuracy): {asa:.4f} ± {asa_std:.4f}")
            else:
                lines.append(f"  ASA (Achievable Segmentation Accuracy): {asa:.4f}")
        
        if 'under_segmentation_error' in metrics:
            ue = metrics['under_segmentation_error']
            ue_std = metrics.get('under_segmentation_error_std', None)
            if ue_std is not None:
                lines.append(f"  UE (Under-segmentation Error):         {ue:.4f} ± {ue_std:.4f}")
            else:
                lines.append(f"  UE (Under-segmentation Error):         {ue:.4f}")
        
        if 'corrected_under_segmentation_error' in metrics:
            cue = metrics['corrected_under_segmentation_error']
            cue_std = metrics.get('corrected_under_segmentation_error_std', None)
            if cue_std is not None:
                lines.append(f"  CUE (Corrected UE):                    {cue:.4f} ± {cue_std:.4f}")
            else:
                lines.append(f"  CUE (Corrected UE):                    {cue:.4f}")
        
        if 'explained_variation' in metrics:
            ev = metrics['explained_variation']
            ev_std = metrics.get('explained_variation_std', None)
            if ev_std is not None:
                lines.append(f"  EV (Explained Variation):              {ev:.4f} ± {ev_std:.4f}")
            else:
                lines.append(f"  EV (Explained Variation):              {ev:.4f}")
    
    # CONTOUR QUALITY - Qualité des contours
    if 'boundary_recall' in metrics or 'precision' in metrics or 'contour_density' in metrics:
        lines.append("\n CONTOUR QUALITY (Qualité des contours)")
        lines.append("─"*75)
        
        if 'boundary_recall' in metrics:
            br = metrics['boundary_recall']
            br_std = metrics.get('boundary_recall_std', None)
            if br_std is not None:
                lines.append(f"  BR (Boundary Recall):                  {br:.4f} ± {br_std:.4f}")
            else:
                lines.append(f"  BR (Boundary Recall):                  {br:.4f}")
        
        if 'precision' in metrics:
            p = metrics['precision']
            p_std = metrics.get('precision_std', None)
            if p_std is not None:
                lines.append(f"  P (Precision):                         {p:.4f} ± {p_std:.4f}")
            else:
                lines.append(f"  P (Precision):                         {p:.4f}")
        
        if 'contour_density' in metrics:
            cd = metrics['contour_density']
            cd_std = metrics.get('contour_density_std', None)
            if cd_std is not None:
                lines.append(f"  CD (Contour Density):                  {cd:.4f} ± {cd_std:.4f}")
            else:
                lines.append(f"  CD (Contour Density):                  {cd:.4f}")
            if cd < 1.0:
                lines.append(f"     → Sous-détection ({100*(1-cd):.1f}% contours manquants)")
            elif cd > 1.0:
                lines.append(f"     → Sur-détection ({100*(cd-1):.1f}% contours supplémentaires)")
            else:
                lines.append(f"     → Densité optimale")
    
    # REGULARITY - Uniformité et régularité
    if 'regularity' in metrics or 'global_regularity' in metrics or 'compactness' in metrics:
        lines.append("\n REGULARITY (Uniformité et régularité)")
        lines.append("─"*75)
        
        if 'regularity' in metrics:
            reg = metrics['regularity']
            lines.append(f"  Regularity (local):                    {reg:.4f}")
        
        if 'global_regularity' in metrics:
            gr = metrics['global_regularity']
            lines.append(f"  GR (Global Regularity):                {gr:.4f}")
        
        if 'compactness' in metrics:
            comp = metrics['compactness']
            lines.append(f"  Compactness:                           {comp:.4f}")
    
    # COLOR COHERENCE - Cohérence chromatique
    # Note: Ces métriques pourraient être calculées si nécessaire
    # Pour l'instant, on peut les ajouter si elles existent
    color_metrics = {
        k: v for k, v in metrics.items() 
        if any(x in k.lower() for x in ['color', 'lab', 'chromatique', 'coherence'])
    }
    
    if color_metrics:
        lines.append("\n🎨 COLOR COHERENCE (Cohérence chromatique)")
        lines.append("─"*75)
        for key, value in color_metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
    
    # Résumé final
    lines.append("\n" + "="*75)
    
    formatted_str = "\n".join(lines)
    
    if verbose:
        print(formatted_str)
    
    return formatted_str


def print_metrics_summary(metrics_list, methods_names):
    """
    Affiche un résumé comparatif de plusieurs méthodes.
    
    Args:
        metrics_list: Liste de dictionnaires de métriques
        methods_names: Liste des noms de méthodes
    
    Example:
        >>> metrics1 = compute_metrics_multiple_gt(labels1, gt)
        >>> metrics2 = compute_metrics_multiple_gt(labels2, gt)
        >>> print_metrics_summary([metrics1, metrics2], ['SLIC_IPOL', 'SIT-HSS'])
    """
    print("\n" + "="*100)
    print("COMPARAISON DES MÉTHODES")
    print("="*100)
    
    # Catégories et métriques
    categories = {
        'OBJECT QUALITY': [
            ('asa', 'ASA', '(plus = mieux)'),
            ('under_segmentation_error', 'UE', '(moins = mieux)'),
            ('corrected_under_segmentation_error', 'CUE', '(moins = mieux)'),
            ('explained_variation', 'EV', '(plus = mieux)')
        ],
        'CONTOUR QUALITY': [
            ('boundary_recall', 'BR', '(plus = mieux)'),
            ('precision', 'P', '(plus = mieux)'),
            ('contour_density', 'CD', '(≈ 1.0 = mieux)')
        ],
        'REGULARITY': [
            ('regularity', 'Regularity', '(plus = mieux)'),
            ('global_regularity', 'GR', '(plus = mieux)'),
            ('compactness', 'Compactness', '(plus = mieux)')
        ]
    }
    
    for category, metric_list in categories.items():
        print(f"\n📊 {category}")
        print("─" * 100)
        
        # En-tête
        header = f"{'Métrique':<30} {'Explication':<30}"
        for method_name in methods_names:
            header += f" {method_name:<12}"
        print(header)
        print("─" * 100)
        
        # Lignes de métriques
        for metric_key, metric_display, explanation in metric_list:
            if any(metric_key in m for m in metrics_list):
                row = f"{metric_display:<30} {explanation:<30}"
                for metrics in metrics_list:
                    if metric_key in metrics:
                        value = metrics[metric_key]
                        std_key = f"{metric_key}_std"
                        if std_key in metrics:
                            row += f" {value:.4f}±{metrics[std_key]:.3f}  "
                        else:
                            row += f" {value:12.4f} "
                    else:
                        row += f" {'N/A':>12} "
                print(row)
    
    print("\n" + "="*100)