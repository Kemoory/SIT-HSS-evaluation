"""
Script de benchmark complet comparant SLIC, SLIC_IPOL et SIT-HSS
Génère des images individuelles pour chaque segmentation
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import pandas as pd
from glob import glob

from src.methods.slic.slic_original import SLIC
from src.methods.slic.slic_ipol import SLIC_IPOL
from src.methods.hierarchical.hierarchical_seg import SITHSS
from src.preprocessing.image_loader import ImageLoader
from src.evaluation.metrics import *
from skimage import segmentation, color as skcolor


def save_individual_segmentation_images(image, labels, metrics, method_name, 
                                       image_name, results_dir, elapsed_time,
                                       gt_list=None):
    """
    Génère et sauvegarde les images individuelles de segmentation.
    
    Crée 4 sous-figures:
    1. Image originale
    2. Image avec contours visibles
    3. Image avec couleurs moyennes des superpixels
    4. Distribution des métriques
    
    Args:
        image: Image originale (H, W, 3)
        labels: Labels des superpixels (H, W)
        metrics: Dictionnaire de métriques
        method_name: Nom de la méthode
        image_name: Nom de l'image
        results_dir: Répertoire de destination
        elapsed_time: Temps d'exécution
        gt_list: Ground truth optionnel
    """
    # Normaliser l'image
    if image.max() > 1:
        img_norm = image / 255.0
    else:
        img_norm = image
    
    # Créer la figure principale (4 sous-figures)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Image originale
    axes[0, 0].imshow(img_norm)
    axes[0, 0].set_title('Image Originale', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Image avec contours
    marked = segmentation.mark_boundaries(
        img_norm, labels, color=(1, 1, 0), mode='thick'
    )
    axes[0, 1].imshow(marked)
    axes[0, 1].set_title(
        f'Contours des Superpixels\n'
        f'{int(metrics["n_superpixels"])} SP | Temps: {elapsed_time:.3f}s',
        fontsize=12, fontweight='bold'
    )
    axes[0, 1].axis('off')
    
    # 3. Couleurs moyennes
    colored = skcolor.label2rgb(labels, img_norm, kind='avg')
    axes[1, 0].imshow(colored)
    axes[1, 0].set_title(
        f'Couleurs Moyennes des Superpixels\n'
        f'Compacité: {metrics["compactness"]:.4f} | '
        f'Régularité: {metrics["regularity"]:.4f}',
        fontsize=11, fontweight='bold'
    )
    axes[1, 0].axis('off')
    
    # 4. Métriques textuelles
    axes[1, 1].axis('off')
    
    metrics_text = f"{'═'*40}\n"
    metrics_text += f"{method_name}\n"
    metrics_text += f"{'═'*40}\n\n"
    metrics_text += f" GÉNÉRALITÉS\n"
    metrics_text += f"  Superpixels: {int(metrics['n_superpixels'])}\n"
    metrics_text += f"  Temps: {elapsed_time:.3f}s\n"
    metrics_text += f"  GR: {metrics.get('global_regularity', 'N/A'):.4f}\n\n"
    
    metrics_text += f" QUALITÉ\n"
    if 'compactness' in metrics:
        metrics_text += f"  Compacité: {metrics['compactness']:.4f}\n"
    if 'regularity' in metrics:
        metrics_text += f"  Régularité: {metrics['regularity']:.4f}\n"
    
    if 'boundary_recall' in metrics:
        metrics_text += f"\n GT METRICS\n"
        metrics_text += f"  BR: {metrics['boundary_recall']:.4f}\n"
        if 'precision' in metrics:
            metrics_text += f"  P: {metrics['precision']:.4f}\n"
        if 'contour_density' in metrics:
            metrics_text += f"  CD: {metrics['contour_density']:.4f}\n"
        if 'under_segmentation_error' in metrics:
            metrics_text += f"  UE: {metrics['under_segmentation_error']:.4f}\n"
        if 'asa' in metrics:
            metrics_text += f"  ASA: {metrics['asa']:.4f}\n"
    
    axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(
        f'{method_name} - {image_name}',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    
    # Sauvegarder
    output_path = results_dir / f'{image_name}_{method_name.replace(" ", "_").lower()}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_comparison_figure_with_gt(image, labels_dict, metrics_dict, 
                                    image_name, results_dir, gt_list=None):
    """
    Crée une figure comparative avec les 3 méthodes + GT.
    
    Args:
        image: Image originale
        labels_dict: {method_name: labels}
        metrics_dict: {method_name: metrics}
        image_name: Nom de l'image
        results_dir: Répertoire de destination
        gt_list: Ground truth optionnel
    """
    # Normaliser l'image
    if image.max() > 1:
        img_norm = image / 255.0
    else:
        img_norm = image
    
    n_methods = len(labels_dict)
    n_cols = n_methods + 2  # +1 pour l'image originale, +1 pour GT si disponible
    
    fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 10))
    
    col_idx = 0
    
    # Colonne 0: Image originale
    axes[0, col_idx].imshow(img_norm)
    axes[0, col_idx].set_title('Image Originale', fontsize=11, fontweight='bold')
    axes[0, col_idx].axis('off')
    
    axes[1, col_idx].imshow(img_norm)
    axes[1, col_idx].set_title('Image Originale', fontsize=11, fontweight='bold')
    axes[1, col_idx].axis('off')
    
    col_idx += 1
    
    # Colonnes pour chaque méthode
    methods_order = ['SLIC', 'SLIC_IPOL', 'SIT-HSS']
    colors = ['steelblue', 'limegreen', 'coral']
    
    for method_idx, (method_name, color) in enumerate(zip(methods_order, colors)):
        if method_name not in labels_dict:
            continue
        
        labels = labels_dict[method_name]
        metrics = metrics_dict[method_name]
        
        # Ligne 0: Contours
        marked = segmentation.mark_boundaries(
            img_norm, labels, color=(1, 1, 0), mode='thick'
        )
        axes[0, col_idx].imshow(marked)
        
        title = f'{method_name}\n'
        title += f'{int(metrics["n_superpixels"])} SP'
        if 'execution_time' in metrics:
            title += f' | {metrics["execution_time"]:.3f}s'
        
        axes[0, col_idx].set_title(title, fontsize=11, fontweight='bold', color=color)
        axes[0, col_idx].axis('off')
        
        # Ligne 1: Couleurs moyennes
        colored = skcolor.label2rgb(labels, img_norm, kind='avg')
        axes[1, col_idx].imshow(colored)
        
        metrics_info = f'Compacité: {metrics["compactness"]:.3f}\n'
        if 'boundary_recall' in metrics:
            metrics_info += f'BR: {metrics["boundary_recall"]:.3f}'
        
        axes[1, col_idx].set_title(metrics_info, fontsize=10)
        axes[1, col_idx].axis('off')
        
        col_idx += 1
    
    # Dernière colonne: Ground Truth (si disponible)
    if gt_list is not None and len(gt_list) > 0:
        gt = gt_list[0]  # Premier annotateur
        
        axes[0, col_idx].imshow(gt, cmap='tab20')
        axes[0, col_idx].set_title('GT Segmentation\n(Annotateur 1)', 
                                   fontsize=11, fontweight='bold', color='red')
        axes[0, col_idx].axis('off')
        
        marked_gt = segmentation.mark_boundaries(
            img_norm, gt, color=(1, 0, 0), mode='thick'
        )
        axes[1, col_idx].imshow(marked_gt)
        axes[1, col_idx].set_title('GT Contours', fontsize=11, fontweight='bold', color='red')
        axes[1, col_idx].axis('off')
    else:
        axes[0, col_idx].axis('off')
        axes[1, col_idx].axis('off')
    
    plt.suptitle(
        f'Comparaison des Méthodes - {image_name}',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    
    # Sauvegarder
    output_path = results_dir / f'{image_name}_comparison_all.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def benchmark_single_method(image_path, method_name, method_instance, 
                           n_segments=200, compactness=10, gt_dir=None):
    """
    Exécute un benchmark d'une méthode sur une seule image.
    """
    loader = ImageLoader(data_dir=gt_dir if gt_dir else 'data')
    image = loader.load_image(image_path)
    image_name = Path(image_path).stem
    
    # Exécuter la méthode
    start_time = time.time()
    labels = method_instance.fit(image)
    elapsed_time = time.time() - start_time
    
    # Charger ground truth si disponible
    gt_list = None
    if gt_dir:
        try:
            gt_all = loader.load_bsds500_groundtruth(split='global', image_name=image_name)
            if len(gt_all) > 0:
                gt_list = gt_all[0]
        except Exception as e:
            pass
    
    # Calculer les métriques
    if gt_list:
        metrics = compute_metrics_multiple_gt(labels, gt_list)
    else:
        metrics = compute_all_metrics(labels)
    
    # Ajouter les informations
    metrics['method'] = method_name
    metrics['image'] = image_name
    metrics['execution_time'] = elapsed_time
    metrics['n_segments_target'] = n_segments
    
    # Affichage catégorisé
    display_metrics_categorized(
        metrics,
        image_name=image_name,
        method_name=method_name,
        elapsed_time=elapsed_time,
        verbose=True
    )
    
    return metrics, labels, image, gt_list


def benchmark_all(image_dir=None, split='test', n_segments=200, compactness=10,
                 max_images=None, gt_dir=None, save_results=True, 
                 save_individual_images=True):
    """
    Lance un benchmark complet comparant les trois méthodes.
    """
    print("\n" + "#"*75)
    print("# BENCHMARK COMPLET: SLIC vs SLIC_IPOL vs SIT-HSS")
    print("#"*75)
    
    # Charger les images
    loader = ImageLoader(data_dir=gt_dir if gt_dir else 'data')
    
    if image_dir:
        image_paths = sorted(glob(os.path.join(image_dir, '*.jpg')) + 
                           glob(os.path.join(image_dir, '*.png')))
    else:
        try:
            images, image_paths = loader.load_bsds500_images(split=split, 
                                                            max_images=max_images)
        except Exception as e:
            print(f"Erreur lors du chargement BSDS500: {e}")
            return
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"\nTrouvé {len(image_paths)} images\n")
    
    # Résultats pour les trois méthodes
    results = {
        'SLIC': [],
        'SLIC_IPOL': [],
        'SIT-HSS': []
    }
    
    all_images = {}
    
    # Répertoire de résultats
    results_dir = Path('results/benchmark_individual')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Benchmark sur chaque image
    for img_idx, img_path in enumerate(image_paths, 1):
        image_name = Path(img_path).stem
        
        print(f"\n{'█'*75}")
        print(f"Image {img_idx}/{len(image_paths)}: {image_name}")
        print(f"{'█'*75}")
        
        # Dictionnaires temporaires pour cette image
        labels_dict = {}
        metrics_dict = {}
        gt_list = None
        
        # SLIC Original
        print(f"\n{'▼'*75}")
        slic = SLIC(n_segments=n_segments, compactness=compactness, max_iter=10)
        metrics_slic, labels_slic, image, gt_list = benchmark_single_method(
            img_path, 'SLIC', slic, n_segments=n_segments, 
            compactness=compactness, gt_dir=gt_dir
        )
        results['SLIC'].append((metrics_slic, labels_slic, image))
        labels_dict['SLIC'] = labels_slic
        metrics_dict['SLIC'] = metrics_slic
        all_images[image_name] = image
        
        # Sauvegarder image individuelle SLIC
        if save_individual_images:
            save_individual_segmentation_images(
                image, labels_slic, metrics_slic, 'SLIC', image_name,
                results_dir, metrics_slic['execution_time'], gt_list
            )
        
        # SLIC_IPOL
        print(f"\n{'▼'*75}")
        slic_ipol = SLIC_IPOL(n_segments=n_segments, compactness=compactness, max_iter=10)
        metrics_slic_ipol, labels_slic_ipol, _, _ = benchmark_single_method(
            img_path, 'SLIC_IPOL', slic_ipol, n_segments=n_segments, 
            compactness=compactness, gt_dir=gt_dir
        )
        results['SLIC_IPOL'].append((metrics_slic_ipol, labels_slic_ipol, image))
        labels_dict['SLIC_IPOL'] = labels_slic_ipol
        metrics_dict['SLIC_IPOL'] = metrics_slic_ipol
        
        # Sauvegarder image individuelle SLIC_IPOL
        if save_individual_images:
            save_individual_segmentation_images(
                image, labels_slic_ipol, metrics_slic_ipol, 'SLIC_IPOL', image_name,
                results_dir, metrics_slic_ipol['execution_time'], gt_list
            )
        
        # SIT-HSS
        print(f"\n{'▼'*75}")
        sit_hss = SITHSS(
            n_segments=n_segments, t=0.15, tau=1e-8,
            max_radius=7
        )
        metrics_sit, labels_sit, _, _ = benchmark_single_method(
            img_path, 'SIT-HSS', sit_hss, n_segments=n_segments, 
            compactness=compactness, gt_dir=gt_dir
        )
        results['SIT-HSS'].append((metrics_sit, labels_sit, image))
        labels_dict['SIT-HSS'] = labels_sit
        metrics_dict['SIT-HSS'] = metrics_sit
        
        # Sauvegarder image individuelle SIT-HSS
        if save_individual_images:
            save_individual_segmentation_images(
                image, labels_sit, metrics_sit, 'SIT-HSS', image_name,
                results_dir, metrics_sit['execution_time'], gt_list
            )
        
        # Figure comparative pour cette image
        print(f"\n{'▬'*75}")
        print("Génération de la figure comparative...")
        comparison_fig = create_comparison_figure_with_gt(
            image, labels_dict, metrics_dict, image_name, results_dir, gt_list
        )
        print(f"  Sauvegardée: {comparison_fig}")
    
    # Tableau de résultats
    print("\n" + "="*75)
    print("RÉSUMÉ GLOBAL")
    print("="*75)
    
    all_metrics = []
    for method, method_results in results.items():
        for metrics_dict, _, _ in method_results:
            all_metrics.append(metrics_dict)
    
    df = pd.DataFrame(all_metrics)
    
    # Résumé comparatif final
    print(f"\n{'='*75}")
    print("COMPARAISON GLOBALE DES MÉTHODES")
    print(f"{'='*75}")
    
    # Préparer les données
    methods_list = ['SLIC', 'SLIC_IPOL', 'SIT-HSS']
    metrics_by_method = {}
    
    for method in methods_list:
        metrics_list = [m[0] for m in results[method]]
        
        # Calculer les moyennes
        result_dict = {}
        for metric_key in metrics_list[0].keys():
            values = [m.get(metric_key) for m in metrics_list 
                     if isinstance(m.get(metric_key), (int, float))]
            if values:
                result_dict[metric_key] = float(np.mean(values))
                result_dict[f"{metric_key}_std"] = float(np.std(values)) if len(values) > 1 else 0.0
        
        metrics_by_method[method] = result_dict
    
    # Afficher la comparaison
    print_metrics_summary(
        [metrics_by_method['SLIC'], 
         metrics_by_method['SLIC_IPOL'], 
         metrics_by_method['SIT-HSS']],
        ['SLIC', 'SLIC_IPOL', 'SIT-HSS']
    )
    
    # Sauvegarder les résultats
    if save_results:
        csv_path = results_dir / 'benchmark_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Résultats sauvegardés: {csv_path}")
        print(f"✓ Images individuelles sauvegardées dans: {results_dir}")
    
    print("\n" + "#"*75)
    print("# BENCHMARK TERMINÉ")
    print("#"*75)


def main():
    """Point d'entrée principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Benchmark complet avec images individuelles'
    )
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Répertoire contenant les images')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Split du BSDS500')
    parser.add_argument('--n_segments', type=int, default=200,
                       help='Nombre de superpixels')
    parser.add_argument('--compactness', type=float, default=10.0,
                       help='Paramètre de compacité')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Nombre maximum d\'images')
    parser.add_argument('--g', type=str, default=None,
                       help='Répertoire du ground truth')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Sauvegarder les résultats')
    parser.add_argument('--no_individual', action='store_true', default=False,
                       help='Ne pas sauvegarder les images individuelles')
    
    args = parser.parse_args()
    
    benchmark_all(
        image_dir=args.image_dir,
        split=args.split,
        n_segments=args.n_segments,
        compactness=args.compactness,
        max_images=args.max_images,
        gt_dir=args.g,
        save_results=args.save,
        save_individual_images=not args.no_individual
    )


if __name__ == "__main__":
    main()