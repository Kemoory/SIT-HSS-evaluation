"""
Conversions d'espaces colorimétriques pour SLIC
"""
import numpy as np
from skimage import color


def rgb_to_lab(image):
    """
    Convertit une image RGB en espace Lab.
    
    Args:
        image: Image RGB (H, W, 3) avec valeurs [0, 255] ou [0, 1]
    
    Returns:
        Image Lab (H, W, 3)
    """
    if image.max() > 1.0:
        image = image / 255.0
    
    return color.rgb2lab(image)


def lab_to_rgb(lab_image):
    """
    Convertit une image Lab en RGB.
    
    Args:
        lab_image: Image Lab (H, W, 3)
    
    Returns:
        Image RGB (H, W, 3) avec valeurs [0, 1]
    """
    return color.lab2rgb(lab_image)


def normalize_lab(lab_image):
    """
    Normalise les valeurs Lab pour le calcul de distances.
    
    Args:
        lab_image: Image Lab (H, W, 3)
    
    Returns:
        Image Lab normalisée
    """
    normalized = lab_image.copy()
    # L: [0, 100], a: [-127, 127], b: [-127, 127]
    normalized[:, :, 0] = normalized[:, :, 0] / 100.0
    normalized[:, :, 1] = (normalized[:, :, 1] + 127.0) / 254.0
    normalized[:, :, 2] = (normalized[:, :, 2] + 127.0) / 254.0
    
    return normalized