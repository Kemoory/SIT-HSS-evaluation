"""
Script de démarrage rapide pour tester SIT-HSS (Corrigé)
Structure identique à quick_start.py
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from PIL import Image
from skimage import segmentation

from src.methods.hierarchical.hierarchical_seg import SITHSS
from src.evaluation.metrics import compute_all_metrics, format_metrics
from src.evaluation.visualize import visualize_segmentation

def create_test_image():
    """Crée une image test synthétique avec des régions colorées (Identique à SLIC)"""
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    image[:, :] = [100, 150, 200] # Fond bleu
    
    y, x = np.ogrid[:300, :400]
    mask_circle = (x - 150)**2 + (y - 150)**2 <= 50**2
    image[mask_circle] = [220, 50, 50] # Cercle rouge
    
    image[50:100, 250:350] = [50, 200, 50] # Rectangle vert
    image[200:250, 250:350] = [230, 230, 50] # Rectangle jaune
    
    noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return image

def example_1_basic_usage():
    """Utilisation de base sur image synthétique"""
    print("\n--- Exemple 1: Utilisation de base (Synthétique) ---")
    image = create_test_image()
    
    # Initialisation SIT-HSS
    model = SITHSS(n_segments=200, t=0.1)
    
    start_time = time.time()
    labels = model.fit(image)
    elapsed = time.time() - start_time
    
    print(f"SIT-HSS terminé en {elapsed:.3f}s pour {len(np.unique(labels))} superpixels")
    
    visualize_segmentation(image, labels, "SIT-HSS (K=200, t=0.1)")
    plt.savefig('images/hss_example_basic.png', dpi=150, bbox_inches='tight')
    print("Figure sauvegardée: images/hss_example_basic.png")
    plt.show()

def example_2_parameter_comparison():
    """Comparaison de l'influence du paramètre t"""
    print("\n--- Exemple 2: Comparaison du paramètre de lissage 't' ---")
    image = create_test_image()
    t_values = [0.05, 0.1, 0.3]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, t_val in enumerate(t_values):
        model = SITHSS(n_segments=100, t=t_val)
        labels = model.fit(image)
        
        out = segmentation.mark_boundaries(image, labels, color=(1, 1, 0))
        axes[i].imshow(out)
        axes[i].set_title(f"t = {t_val}")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig('images/hss_example_parameters.png', bbox_inches='tight')
    print("Figure sauvegardée: images/hss_example_parameters.png")
    plt.show()

def example_3_metrics_evaluation():
    """Évaluation des métriques sur SIT-HSS"""
    print("\n--- Exemple 3: Évaluation des métriques ---")
    image = create_test_image()
    model = SITHSS(n_segments=200, t=0.1)
    labels = model.fit(image)
    
    metrics = compute_all_metrics(labels)
    print("Métriques pour SIT-HSS :")
    print(format_metrics(metrics))

def example_4_real_image():
    """Chargement et test sur image réelle"""
    print("\n--- Exemple 4: Image réelle (BSDS500) ---")
    img_path = 'data/BSDS500/data/images/test/16068.jpg'
    
    try:
        if os.path.exists(img_path):
            image = np.array(Image.open(img_path))
            
            # Application SIT-HSS
            model = SITHSS(n_segments=200, t=0.1)
            labels = model.fit(image)
            
            visualize_segmentation(image, labels, "SIT-HSS sur BSDS500")
            plt.savefig('images/hss_example_real_image.png', dpi=150, bbox_inches='tight')
            print("Figure sauvegardée: images/hss_example_real_image.png")
            plt.show()
            
            metrics = compute_all_metrics(labels)
            print(format_metrics(metrics))
        else:
            print(f"Aucune image trouvée à {img_path}")
            print("Veuillez vérifier le chemin data/BSDS500/")
            
    except Exception as e:
        print(f"Erreur lors du chargement de l'image réelle: {e}")

def main():
    """Fonction principale"""
    # Créer le dossier de résultats
    os.makedirs('images', exist_ok=True)
    
    print("\n" + "#"*70)
    print("# DÉMONSTRATION DE LA MÉTHODE SIT-HSS")
    print("#"*70)
    
    # Exécuter les exemples
    
    example_1_basic_usage()
    example_2_parameter_comparison()
    example_3_metrics_evaluation()
    example_4_real_image()

    
    print("\n" + "#"*70)
    print("# DÉMONSTRATION TERMINÉE")
    print("#"*70)
    print("\nLes résultats SIT-HSS sont dans le dossier 'images/'")

if __name__ == "__main__":
    main()
