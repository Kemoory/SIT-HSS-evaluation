"""
Métriques de distance pour SLIC et autres algorithmes de superpixels

La distance SLIC combine distance couleur (dans l'espace Lab) et distance spatiale:
    D_SLIC = sqrt(d_lab² + (d_xy/S)² × m²)

où:
    - d_lab: distance euclidienne dans l'espace Lab
    - d_xy: distance euclidienne spatiale (2D)
    - S: espacement de grille (normalisation spatiale)
    - m: paramètre de compacité (contrôle le poids relatif)
"""
import numpy as np


def compute_distance(pixel_lab, center_lab, pixel_xy, center_xy, m, S):
    """
    Calcule la distance SLIC entre un pixel et un centre de cluster.
    
    La distance SLIC est une combinaison pondérée de:
    - Distance couleur dans l'espace Lab (perceptuellement uniforme)
    - Distance spatiale normalisée par S
    
    Formule: D = sqrt(d_lab² + (d_xy/S)² × m²)
    
    Args:
        pixel_lab: Coordonnées Lab du pixel, array (3,) [L, a, b]
        center_lab: Coordonnées Lab du centre, array (3,) [L, a, b]
        pixel_xy: Coordonnées spatiales du pixel, array (2,) [x, y]
        center_xy: Coordonnées spatiales du centre, array (2,) [x, y]
        m: Paramètre de compacité (typiquement entre 1 et 40)
           - m faible: privilégie la cohérence couleur
           - m élevé: privilégie la compacité spatiale
        S: Espacement de grille, utilisé pour normaliser les distances spatiales
    
    Returns:
        distance: Distance SLIC scalaire
    
    Example:
        >>> pixel_lab = np.array([50.0, 10.0, 20.0])
        >>> center_lab = np.array([55.0, 12.0, 18.0])
        >>> pixel_xy = np.array([100, 150])
        >>> center_xy = np.array([105, 155])
        >>> d = compute_distance(pixel_lab, center_lab, pixel_xy, center_xy, m=10, S=20)
    """
    # Distance couleur dans l'espace Lab (distance euclidienne 3D)
    d_lab = np.sqrt(np.sum((pixel_lab - center_lab) ** 2))
    
    # Distance spatiale (distance euclidienne 2D)
    d_xy = np.sqrt(np.sum((pixel_xy - center_xy) ** 2))
    
    # Distance SLIC combinée
    # Le terme (d_xy / S) normalise la distance spatiale
    # Le terme m² contrôle le poids relatif spatial vs couleur
    D = np.sqrt(d_lab ** 2 + (d_xy / S) ** 2 * (m ** 2))
    
    return D


def compute_distance_vectorized(pixels_lab, center_lab, pixels_xy, center_xy, m, S):
    """
    Version vectorisée du calcul de distance SLIC pour multiple pixels.
    
    Cette version est optimisée pour traiter plusieurs pixels simultanément,
    ce qui est beaucoup plus rapide que d'appeler compute_distance en boucle.
    
    Args:
        pixels_lab: Coordonnées Lab des pixels, array (N, 3) où N = nombre de pixels
        center_lab: Coordonnées Lab du centre, array (3,) [L, a, b]
        pixels_xy: Coordonnées spatiales des pixels, array (N, 2) [x, y]
        center_xy: Coordonnées spatiales du centre, array (2,) [x, y]
        m: Paramètre de compacité
        S: Espacement de grille
    
    Returns:
        distances: Array de distances SLIC, shape (N,)
    
    Example:
        >>> n_pixels = 100
        >>> pixels_lab = np.random.randn(n_pixels, 3) * 20 + 50
        >>> center_lab = np.array([50.0, 10.0, 20.0])
        >>> pixels_xy = np.random.randn(n_pixels, 2) * 10 + 100
        >>> center_xy = np.array([100, 100])
        >>> distances = compute_distance_vectorized(pixels_lab, center_lab, 
        ...                                        pixels_xy, center_xy, m=10, S=20)
        >>> print(distances.shape)  # (100,)
    """
    # Distance couleur (broadcasting automatique)
    # Shape: (N, 3) - (3,) = (N, 3)
    d_lab = np.sqrt(np.sum((pixels_lab - center_lab) ** 2, axis=1))
    
    # Distance spatiale
    # Shape: (N, 2) - (2,) = (N, 2)
    d_xy = np.sqrt(np.sum((pixels_xy - center_xy) ** 2, axis=1))
    
    # Distance SLIC combinée
    # Shape: (N,)
    D = np.sqrt(d_lab ** 2 + (d_xy / S) ** 2 * (m ** 2))
    
    return D


def euclidean_distance(point1, point2):
    """
    Calcule la distance euclidienne entre deux points dans un espace N-D.
    
    Args:
        point1: Premier point, array (N,)
        point2: Deuxième point, array (N,)
    
    Returns:
        distance: Distance euclidienne scalaire
    
    Example:
        >>> p1 = np.array([0, 0, 0])
        >>> p2 = np.array([3, 4, 0])
        >>> d = euclidean_distance(p1, p2)
        >>> print(d)  # 5.0
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))


def euclidean_distance_vectorized(points1, points2):
    """
    Version vectorisée de la distance euclidienne pour plusieurs paires de points.
    
    Args:
        points1: Premiers points, array (N, D) où N = nombre de points, D = dimensions
        points2: Deuxièmes points, array (N, D)
    
    Returns:
        distances: Array de distances, shape (N,)
    
    Example:
        >>> n = 1000
        >>> p1 = np.random.randn(n, 3)
        >>> p2 = np.random.randn(n, 3)
        >>> distances = euclidean_distance_vectorized(p1, p2)
        >>> print(distances.shape)  # (1000,)
    """
    return np.sqrt(np.sum((points1 - points2) ** 2, axis=1))


def manhattan_distance(point1, point2):
    """
    Calcule la distance de Manhattan (L1) entre deux points.
    
    Args:
        point1: Premier point, array (N,)
        point2: Deuxième point, array (N,)
    
    Returns:
        distance: Distance de Manhattan scalaire
    
    Example:
        >>> p1 = np.array([0, 0])
        >>> p2 = np.array([3, 4])
        >>> d = manhattan_distance(p1, p2)
        >>> print(d)  # 7.0
    """
    return np.sum(np.abs(point1 - point2))


def lab_color_distance(lab1, lab2):
    """
    Calcule la distance de couleur dans l'espace Lab (dE).
    
    L'espace Lab est conçu pour être perceptuellement uniforme:
    une distance de 1.0 correspond à une différence de couleur
    à peine perceptible par l'œil humain.
    
    Args:
        lab1: Couleur Lab 1, array (3,) [L, a, b]
        lab2: Couleur Lab 2, array (3,) [L, a, b]
    
    Returns:
        delta_E: Distance de couleur (Delta E)
    
    Example:
        >>> color1 = np.array([50, 10, 20])
        >>> color2 = np.array([55, 12, 18])
        >>> dE = lab_color_distance(color1, color2)
    """
    dL = lab1[0] - lab2[0]
    da = lab1[1] - lab2[1]
    db = lab1[2] - lab2[2]
    
    delta_E = np.sqrt(dL**2 + da**2 + db**2)
    
    return delta_E


def lab_color_distance_vectorized(labs1, labs2):
    """
    Version vectorisée de la distance de couleur Lab.
    
    Args:
        labs1: Couleurs Lab 1, array (N, 3)
        labs2: Couleurs Lab 2, array (N, 3)
    
    Returns:
        delta_Es: Array de distances de couleur, shape (N,)
    """
    return np.sqrt(np.sum((labs1 - labs2) ** 2, axis=1))


def mahalanobis_distance(point, mean, cov_inv):
    """
    Calcule la distance de Mahalanobis.
    
    Utile pour mesurer la distance en tenant compte de la covariance
    (forme de la distribution).
    
    Args:
        point: Point à évaluer, array (D,)
        mean: Moyenne de la distribution, array (D,)
        cov_inv: Inverse de la matrice de covariance, array (D, D)
    
    Returns:
        distance: Distance de Mahalanobis
    
    Example:
        >>> point = np.array([1, 2])
        >>> mean = np.array([0, 0])
        >>> cov = np.array([[1, 0.5], [0.5, 1]])
        >>> cov_inv = np.linalg.inv(cov)
        >>> d = mahalanobis_distance(point, mean, cov_inv)
    """
    diff = point - mean
    distance = np.sqrt(diff.T @ cov_inv @ diff)
    return distance


def cosine_distance(vec1, vec2):
    """
    Calcule la distance cosinus (1 - similarité cosinus).
    
    Utile pour comparer des vecteurs de caractéristiques.
    Distance = 1 - cos(θ) où θ est l'angle entre les vecteurs.
    
    Args:
        vec1: Premier vecteur, array (N,)
        vec2: Deuxième vecteur, array (N,)
    
    Returns:
        distance: Distance cosinus dans [0, 2]
    
    Example:
        >>> v1 = np.array([1, 0, 0])
        >>> v2 = np.array([0, 1, 0])
        >>> d = cosine_distance(v1, v2)
        >>> print(d)  # 1.0 (orthogonaux)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 1.0  # Vecteurs nuls considérés comme orthogonaux
    
    cosine_similarity = dot_product / (norm1 * norm2)
    distance = 1.0 - cosine_similarity
    
    return distance


def squared_euclidean_distance(point1, point2):
    """
    Calcule la distance euclidienne au carré (plus rapide, évite sqrt).
    
    Utile quand seul l'ordre des distances importe (comparaisons).
    
    Args:
        point1: Premier point, array (N,)
        point2: Deuxième point, array (N,)
    
    Returns:
        squared_distance: Distance euclidienne au carré
    """
    return np.sum((point1 - point2) ** 2)


def squared_euclidean_distance_vectorized(points1, points2):
    """
    Version vectorisée de la distance euclidienne au carré.
    
    Args:
        points1: Premiers points, array (N, D)
        points2: Deuxièmes points, array (N, D)
    
    Returns:
        squared_distances: Array de distances au carré, shape (N,)
    """
    return np.sum((points1 - points2) ** 2, axis=1)


def pairwise_distances(points, metric='euclidean'):
    """
    Calcule la matrice de distances entre tous les points.
    
    Args:
        points: Array de points, shape (N, D)
        metric: Type de métrique ('euclidean', 'manhattan', 'squared_euclidean')
    
    Returns:
        dist_matrix: Matrice de distances, shape (N, N)
    
    Example:
        >>> points = np.random.randn(100, 5)
        >>> dist_mat = pairwise_distances(points, metric='euclidean')
        >>> print(dist_mat.shape)  # (100, 100)
    """
    n_points = points.shape[0]
    dist_matrix = np.zeros((n_points, n_points))
    
    if metric == 'euclidean':
        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = euclidean_distance(points[i], points[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
    
    elif metric == 'manhattan':
        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = manhattan_distance(points[i], points[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
    
    elif metric == 'squared_euclidean':
        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = squared_euclidean_distance(points[i], points[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
    
    else:
        raise ValueError(f"Métrique inconnue: {metric}")
    
    return dist_matrix


def nearest_neighbors(query_point, points, k=1, metric='euclidean'):
    """
    Trouve les k plus proches voisins d'un point.
    
    Args:
        query_point: Point de requête, array (D,)
        points: Ensemble de points, array (N, D)
        k: Nombre de voisins à retourner
        metric: Type de métrique
    
    Returns:
        indices: Indices des k plus proches voisins
        distances: Distances correspondantes
    
    Example:
        >>> query = np.array([0, 0])
        >>> points = np.random.randn(1000, 2)
        >>> indices, distances = nearest_neighbors(query, points, k=5)
    """
    n_points = points.shape[0]
    distances = np.zeros(n_points)
    
    if metric == 'euclidean':
        for i in range(n_points):
            distances[i] = euclidean_distance(query_point, points[i])
    elif metric == 'manhattan':
        for i in range(n_points):
            distances[i] = manhattan_distance(query_point, points[i])
    else:
        raise ValueError(f"Métrique inconnue: {metric}")
    
    # Trier et retourner les k plus proches
    indices = np.argsort(distances)[:k]
    
    return indices, distances[indices]


def compute_distance_matrix_region(pixels_lab, pixels_xy, center_lab, center_xy, m, S):
    """
    Calcule les distances SLIC pour une région entière de pixels.
    
    Fonction optimisée pour calculer efficacement les distances entre
    tous les pixels d'une région et un centre de cluster.
    
    Args:
        pixels_lab: Valeurs Lab des pixels, array (H, W, 3)
        pixels_xy: Coordonnées des pixels, array (H, W, 2)
        center_lab: Centre Lab, array (3,)
        center_xy: Centre spatial, array (2,)
        m: Paramètre de compacité
        S: Espacement de grille
    
    Returns:
        distances: Matrice de distances, shape (H, W)
    """
    h, w = pixels_lab.shape[:2]
    
    # Reshape pour traitement vectorisé
    pixels_lab_flat = pixels_lab.reshape(-1, 3)
    pixels_xy_flat = pixels_xy.reshape(-1, 2)
    
    # Calculer les distances
    distances_flat = compute_distance_vectorized(
        pixels_lab_flat, center_lab,
        pixels_xy_flat, center_xy,
        m, S
    )
    
    # Reshape au format original
    distances = distances_flat.reshape(h, w)
    
    return distances