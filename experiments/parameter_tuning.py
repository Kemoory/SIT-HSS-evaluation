"""
Optimisation et analyse des paramètres SLIC
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from src.methods.slic.slic_original import SLIC
from src.preprocessing.image_loader import ImageLoader
from src.evaluation.metrics import compute_all_metrics, compute_metrics_multiple_gt
from src.evaluation.visualize import plot_parameter_study


def study_n_segments(image, ground_truths=None, 
                    n_segments_values=[50, 100, 200, 400, 800],
                    compactness=10.0):
    """
    Étudie l'effet du nombre de superpixels.
    
    Args:
        image: Image RGB (H, W, 3)
        ground_truths: Liste de segmentations GT (optionnel)
        n_segments_values: Liste des valeurs à tester
        compactness: Paramètre de compacité fixe
    
    Returns:
        results: Dictionnaire avec labels et métriques
    """
    print("=" * 60)
    print("ÉTUDE DU PARAMÈTRE N_SEGMENTS")
    print("=" * 60)
    
    results = {
        'n_segments_values': n_segments_values,
        'labels_list': [],
        'metrics_list': [],
        'times': []
    }
    
    for n_seg in n_segments_values:
        print(f"\nTest avec n_segments={n_seg}...")
        
        # Mesurer le temps
        start_time = time.time()
        
        # Appliquer SLIC
        slic = SLIC(n_segments=n_seg, compactness=compactness, max_iter=10)
        labels = slic.fit(image)
        
        elapsed = time.time() - start_time
        results['times'].append(elapsed)
        results['labels_list'].append(labels)
        
        # Calculer les métriques
        if ground_truths:
            metrics = compute_metrics_multiple_gt(labels, ground_truths)
        else:
            metrics = compute_all_metrics(labels)
        
        results['metrics_list'].append(metrics)
        
        print(f"  Temps: {elapsed:.2f}s")
        print(f"  Nombre réel: {metrics['n_superpixels']}")
        print(f"  Compacité: {metrics['compactness']:.4f}")
        if 'boundary_recall' in metrics:
            print(f"  BR: {metrics['boundary_recall']:.4f}")
            print(f"  UE: {metrics['under_segmentation_error']:.4f}")
    
    return results


def study_compactness(image, ground_truths=None,
                     compactness_values=[1, 5, 10, 20, 40],
                     n_segments=200):
    """
    Étudie l'effet du paramètre de compacité.
    
    Args:
        image: Image RGB (H, W, 3)
        ground_truths: Liste de segmentations GT (optionnel)
        compactness_values: Liste des valeurs à tester
        n_segments: Nombre de superpixels fixe
    
    Returns:
        results: Dictionnaire avec labels et métriques
    """
    print("=" * 60)
    print("ÉTUDE DU PARAMÈTRE COMPACTNESS")
    print("=" * 60)
    
    results = {
        'compactness_values': compactness_values,
        'labels_list': [],
        'metrics_list': [],
        'times': []
    }
    
    for comp in compactness_values:
        print(f"\nTest avec compactness={comp}...")
        
        start_time = time.time()
        
        slic = SLIC(n_segments=n_segments, compactness=comp, max_iter=10)
        labels = slic.fit(image)
        
        elapsed = time.time() - start_time
        results['times'].append(elapsed)
        results['labels_list'].append(labels)
        
        if ground_truths:
            metrics = compute_metrics_multiple_gt(labels, ground_truths)
        else:
            metrics = compute_all_metrics(labels)
        
        results['metrics_list'].append(metrics)
        
        print(f"  Temps: {elapsed:.2f}s")
        print(f"  Compacité mesurée: {metrics['compactness']:.4f}")
        if 'boundary_recall' in metrics:
            print(f"  BR: {metrics['boundary_recall']:.4f}")
    
    return results


def grid_search(image, ground_truths=None,
               n_segments_values=[100, 200, 400],
               compactness_values=[5, 10, 20]):
    """
    Recherche en grille des meilleurs paramètres.
    
    Args:
        image: Image RGB (H, W, 3)
        ground_truths: Liste de segmentations GT (optionnel)
        n_segments_values: Liste des valeurs de n_segments
        compactness_values: Liste des valeurs de compactness
    
    Returns:
        results: Dictionnaire avec tous les résultats
    """
    print("=" * 60)
    print("RECHERCHE EN GRILLE")
    print("=" * 60)
    print(f"n_segments: {n_segments_values}")
    print(f"compactness: {compactness_values}")
    
    results = {
        'n_segments_values': n_segments_values,
        'compactness_values': compactness_values,
        'grid': {}
    }
    
    best_score = -1
    best_params = None
    
    for n_seg in n_segments_values:
        for comp in compactness_values:
            print(f"\nTest: n_segments={n_seg}, compactness={comp}")
            
            slic = SLIC(n_segments=n_seg, compactness=comp, max_iter=10)
            labels = slic.fit(image)
            
            if ground_truths:
                metrics = compute_metrics_multiple_gt(labels, ground_truths)
                # Score combiné (BR et ASA élevés, UE faible)
                score = (metrics['boundary_recall'] + metrics['asa'] - 
                        metrics['under_segmentation_error']) / 2
            else:
                metrics = compute_all_metrics(labels)
                score = metrics['compactness']
            
            results['grid'][(n_seg, comp)] = {
                'labels': labels,
                'metrics': metrics,
                'score': score
            }
            
            print(f"  Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_params = (n_seg, comp)
    
    print("\n" + "=" * 60)
    print("MEILLEURS PARAMÈTRES")
    print("=" * 60)
    print(f"n_segments: {best_params[0]}")
    print(f"compactness: {best_params[1]}")
    print(f"Score: {best_score:.4f}")
    
    results['best_params'] = best_params
    results['best_score'] = best_score
    
    return results


def visualize_parameter_study(results, param_name):
    """
    Visualise les résultats d'une étude de paramètre.
    
    Args:
        results: Résultats de l'étude
        param_name: Nom du paramètre ('n_segments' ou 'compactness')
    """
    param_key = f'{param_name}_values'
    param_values = results[param_key]
    metrics_list = results['metrics_list']
    
    # Graphique des métriques
    fig = plot_parameter_study(param_name, param_values, metrics_list)
    
    # Graphique du temps d'exécution
    fig_time, ax = plt.subplots(figsize=(8, 5))
    ax.plot(param_values, results['times'], marker='o', linewidth=2, markersize=8)
    ax.set_xlabel(param_name)
    ax.set_ylabel('Temps (secondes)')
    ax.set_title(f'Temps d\'exécution vs {param_name}')
    ax.grid(True, alpha=0.3)
    
    return fig, fig_time


def visualize_grid_search(results, image):
    """
    Visualise les résultats de la recherche en grille.
    
    Args:
        results: Résultats de grid_search
        image: Image originale
    """
    n_segments_values = results['n_segments_values']
    compactness_values = results['compactness_values']
    
    n_rows = len(n_segments_values)
    n_cols = len(compactness_values)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    if image.max() > 1.0:
        image = image / 255.0
    
    for i, n_seg in enumerate(n_segments_values):
        for j, comp in enumerate(compactness_values):
            labels = results['grid'][(n_seg, comp)]['labels']
            score = results['grid'][(n_seg, comp)]['score']
            
            from skimage import segmentation
            marked = segmentation.mark_boundaries(image, labels, 
                                                 color=(1, 1, 0), mode='thick')
            
            axes[i, j].imshow(marked)
            axes[i, j].set_title(f'n={n_seg}, m={comp}\nScore: {score:.3f}')
            axes[i, j].axis('off')
            
            # Encadrer les meilleurs paramètres
            if (n_seg, comp) == results['best_params']:
                for spine in axes[i, j].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(4)
    
    plt.tight_layout()
    return fig


def main():
    """
    Point d'entrée principal.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimisation des paramètres SLIC')
    parser.add_argument('--image', type=str, required=True, 
                       help='Chemin vers l\'image')
    parser.add_argument('--study', type=str, default='all',
                       choices=['n_segments', 'compactness', 'grid', 'all'],
                       help='Type d\'étude à réaliser')
    parser.add_argument('--save', action='store_true',
                       help='Sauvegarder les résultats')
    
    args = parser.parse_args()
    
    # Charger l'image
    loader = ImageLoader()
    image = loader.load_image(args.image)
    print(f"Image chargée: {image.shape}")
    
    # Ground truth si disponible (BSDS500)
    ground_truths = None
    try:
        image_name = Path(args.image).stem
        gt_data = loader.load_bsds500_groundtruth(split='test', image_name=image_name)
        if gt_data:
            ground_truths = gt_data[0]
            print(f"Ground truth chargé: {len(ground_truths)} segmentations")
    except:
        print("Pas de ground truth disponible")
    
    # Exécuter les études
    if args.study in ['n_segments', 'all']:
        results_nseg = study_n_segments(image, ground_truths)
        fig1, fig2 = visualize_parameter_study(results_nseg, 'n_segments')
        
        if args.save:
            save_dir = Path('results/slic/parameters')
            save_dir.mkdir(parents=True, exist_ok=True)
            fig1.savefig(save_dir / 'n_segments_metrics.png', dpi=150, bbox_inches='tight')
            fig2.savefig(save_dir / 'n_segments_time.png', dpi=150, bbox_inches='tight')
    
    if args.study in ['compactness', 'all']:
        results_comp = study_compactness(image, ground_truths)
        fig3, fig4 = visualize_parameter_study(results_comp, 'compactness')
        
        if args.save:
            save_dir = Path('results/slic/parameters')
            save_dir.mkdir(parents=True, exist_ok=True)
            fig3.savefig(save_dir / 'compactness_metrics.png', dpi=150, bbox_inches='tight')
            fig4.savefig(save_dir / 'compactness_time.png', dpi=150, bbox_inches='tight')
    
    if args.study in ['grid', 'all']:
        results_grid = grid_search(image, ground_truths)
        fig5 = visualize_grid_search(results_grid, image)
        
        if args.save:
            save_dir = Path('results/slic/parameters')
            save_dir.mkdir(parents=True, exist_ok=True)
            fig5.savefig(save_dir / 'grid_search.png', dpi=150, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    main()