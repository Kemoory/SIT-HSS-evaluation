"""
Chargement et prétraitement des images
"""
import os
import numpy as np
from PIL import Image
from pathlib import Path


class ImageLoader:
    """
    Classe pour charger les images depuis différentes sources.
    """
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
    
    def load_image(self, image_path, resize=None):
        """
        Charge une image.
        
        Args:
            image_path: Chemin vers l'image
            resize: Tuple (width, height) pour redimensionner, None sinon
        
        Returns:
            image: Array numpy (H, W, 3) avec valeurs [0, 255]
        """
        img = Image.open(image_path).convert('RGB')
        
        if resize is not None:
            img = img.resize(resize, Image.LANCZOS)
        
        return np.array(img)
    
    def load_bsds500_images(self, split='train', max_images=None):
        """
        Charge les images du dataset BSDS500.
        
        Args:
            split: 'train', 'val', ou 'test'
            max_images: Nombre maximum d'images à charger (None = toutes)
        
        Returns:
            images: Liste d'images
            paths: Liste des chemins
        """
        images_dir = self.data_dir / 'BSDS500' / 'data' / 'images' / split
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Directory not found: {images_dir}")
        
        image_paths = sorted(list(images_dir.glob('*.jpg')))
        
        if max_images is not None:
            image_paths = image_paths[:max_images]
        
        images = []
        paths = []
        
        for img_path in image_paths:
            try:
                img = self.load_image(img_path)
                images.append(img)
                paths.append(str(img_path))
                print(f"Loaded: {img_path.name}")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        return images, paths
    
    # TODO: Si on fait du machine learning remplacer global par train/val/test, sinon surpimmer train/val/test
    def load_bsds500_groundtruth(self, split='global', image_name=None):
        """
        Charge les segmentations ground truth du BSDS500.
        
        Args:
            split: 'train', 'val', ou 'test'
            image_name: Nom de l'image (sans extension)
        
        Returns:
            groundtruth: Liste de segmentations (plusieurs par image)
        """
        gt_dir = self.data_dir / 'BSDS500' / 'data' / 'groundTruth' / split
        
        if not gt_dir.exists():
            raise FileNotFoundError(f"Directory not found: {gt_dir}")
        
        if image_name is None:
            # Charger tous les ground truths
            gt_paths = sorted(list(gt_dir.glob('*.mat')))
        else:
            # Charger le ground truth pour une image spécifique
            gt_path = gt_dir / f"{image_name}.mat"
            if not gt_path.exists():
                raise FileNotFoundError(f"Ground truth not found: {gt_path}")
            gt_paths = [gt_path]
        
        groundtruths = []
        
        for gt_path in gt_paths:
            try:
                import scipy.io
                mat = scipy.io.loadmat(gt_path)
                
                # Le format BSDS500 contient plusieurs segmentations par image
                # Structure: mat['groundTruth'][0][n]['Segmentation'][0][0]
                gt_list = []
                for i in range(len(mat['groundTruth'][0])):
                    segmentation = mat['groundTruth'][0][i]['Segmentation'][0][0]
                    gt_list.append(segmentation)
                
                groundtruths.append(gt_list)
                print(f"Loaded GT: {gt_path.name} ({len(gt_list)} segmentations)")
            except Exception as e:
                print(f"Error loading {gt_path}: {e}")
        
        return groundtruths
    
    def get_available_datasets(self):
        """
        Liste les datasets disponibles.
        
        Returns:
            datasets: Liste des noms de datasets
        """
        datasets = []
        
        if (self.data_dir / 'BSDS500').exists():
            datasets.append('BSDS500')
        
        return datasets


def preprocess_image(image, normalize=True, blur_sigma=0):
    """
    Prétraite une image.
    
    Args:
        image: Image RGB (H, W, 3)
        normalize: Si True, normalise entre [0, 1]
        blur_sigma: Écart-type pour le flou gaussien (0 = pas de flou)
    
    Returns:
        processed_image: Image prétraitée
    """
    from scipy import ndimage
    
    processed = image.copy().astype(np.float64)
    
    # Normalisation
    if normalize and image.max() > 1.0:
        processed = processed / 255.0
    
    # Flou gaussien optionnel
    if blur_sigma > 0:
        for i in range(3):
            processed[:, :, i] = ndimage.gaussian_filter(processed[:, :, i], sigma=blur_sigma)
    
    return processed