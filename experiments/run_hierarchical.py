"""
Script d'execution SIT-HSS avec traitement parallele
Genere des images individuelles pour chaque segmentation
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
import torch
from multiprocessing import Pool, cpu_count
import multiprocessing

from src.methods.hierarchical.hierarchical_seg import SITHSS
from src.preprocessing.image_loader import ImageLoader
from src.evaluation.metrics import *
from skimage import segmentation, color as skcolor



def process_sithss_single_image(args):
    img_path, n_segments, t, tau, max_radius, gt_dir = args

    loader = ImageLoader(data_dir=gt_dir if gt_dir else 'data')
    image = loader.load_image(img_path)
    image_name = Path(img_path).stem

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sit_hss = SITHSS(
        n_segments=n_segments,
        t=t,
        tau=tau,
        max_radius=max_radius,
        device=device
    )

    start_time = time.time()
    labels = sit_hss.fit(image)
    elapsed_time = time.time() - start_time

    gt_list = None
    if gt_dir:
        try:
            gt_all = loader.load_bsds500_groundtruth(split='global', image_name=image_name)
            if len(gt_all) > 0:
                gt_list = gt_all[0]
        except Exception as e:
            pass

    if gt_list:
        metrics = compute_metrics_multiple_gt(labels, gt_list)
    else:
        metrics = compute_all_metrics(labels)

    metrics['method'] = 'SIT-HSS'
    metrics['image'] = image_name
    metrics['execution_time'] = elapsed_time
    metrics['n_segments_target'] = n_segments

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        'metrics': metrics,
        'labels': labels,
        'image': image,
        'gt_list': gt_list,
        'image_name': image_name,
        'img_path': img_path
    }


def process_all_sithss_parallel(image_paths, n_segments, t, tau, max_radius,
                                gt_dir, n_workers=None):
    if n_workers is None:
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            n_workers = min(n_gpus, 4) if n_gpus > 1 else 2
        else:
            n_workers = min(cpu_count(), 4)

    print(f"\n{'='*75}")
    print(f"TRAITEMENT PARALLELE SITHSS: {len(image_paths)} images avec {n_workers} worker(s)")
    print(f"{'='*75}\n")

    tasks = [(img_path, n_segments, t, tau, max_radius, gt_dir)
             for img_path in image_paths]

    start_time = time.time()

    results_dict = {}
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap(process_sithss_single_image, tasks), 1):
            image_name = result['image_name']
            results_dict[image_name] = result

            elapsed = time.time() - start_time
            eta = (elapsed / i) * (len(tasks) - i)

            print(f"[{i}/{len(tasks)}] {image_name} | "
                  f"Temps: {result['metrics']['execution_time']:.2f}s | "
                  f"ETA: {eta/60:.1f} min", flush=True)

    total_time = time.time() - start_time
    avg_time = total_time / len(image_paths)

    print(f"\n{'='*75}")
    print(f"SITHSS TERMINE: {len(image_paths)} images en {total_time/60:.1f} min")
    print(f"Temps moyen: {avg_time:.2f}s/image")
    print(f"{'='*75}\n")

    return results_dict


def save_individual_segmentation_images(image, labels, metrics, image_name,
                                       results_dir, elapsed_time, gt_list=None):
    if image.max() > 1:
        img_norm = image / 255.0
    else:
        img_norm = image

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].imshow(img_norm)
    axes[0, 0].set_title('Image Originale', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

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

    colored = skcolor.label2rgb(labels, img_norm, kind='avg')
    axes[1, 0].imshow(colored)
    axes[1, 0].set_title(
        f'Couleurs Moyennes des Superpixels\n'
        f'Compacite: {metrics["compactness"]:.4f} | '
        f'Regularite: {metrics["regularity"]:.4f}',
        fontsize=11, fontweight='bold'
    )
    axes[1, 0].axis('off')

    axes[1, 1].axis('off')

    metrics_text = f"{'='*40}\n"
    metrics_text += f"SIT-HSS\n"
    metrics_text += f"{'='*40}\n\n"
    metrics_text += f" GENERALITES\n"
    metrics_text += f"  Superpixels: {int(metrics['n_superpixels'])}\n"
    metrics_text += f"  Temps: {elapsed_time:.3f}s\n\n"

    metrics_text += f" QUALITE GEOMETRIQUE\n"
    if 'global_regularity' in metrics:
        metrics_text += f"  GR: {metrics['global_regularity']:.4f}\n"
    if 'compactness' in metrics:
        metrics_text += f"  CO: {metrics['compactness']:.4f}\n"
    if 'regularity' in metrics:
        metrics_text += f"  RE: {metrics['regularity']:.4f}\n"

    if 'boundary_recall' in metrics:
        metrics_text += f"\n GT METRICS\n"
        metrics_text += f"  BR: {metrics['boundary_recall']:.4f}\n"
        if 'precision' in metrics:
            metrics_text += f"  P: {metrics['precision']:.4f}\n"
        if 'contour_density' in metrics:
            metrics_text += f"  CD: {metrics['contour_density']:.4f}\n"
        if 'under_segmentation_error' in metrics:
            metrics_text += f"  UE: {metrics['under_segmentation_error']:.4f}\n"
        if 'corrected_under_segmentation_error' in metrics:
            metrics_text += f"  CUE: {metrics['corrected_under_segmentation_error']:.4f}\n"
        if 'achievable_segmentation_accuracy' in metrics:
            metrics_text += f"  ASA: {metrics['achievable_segmentation_accuracy']:.4f}\n"
        if 'explained_variation' in metrics:
            metrics_text += f"  EV: {metrics['explained_variation']:.4f}\n"

    axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle(
        f'SIT-HSS - {image_name}',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    output_path = results_dir / f'{image_name}_sithss.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def execute_sithss(image_dir=None, split='test', n_segments=200,
                   t=0.15, tau=1e-8, max_radius=7,
                   max_images=None, gt_dir=None, save_results=True,
                   save_individual_images=True, use_parallel=True,
                   n_workers=None):
    print("\n" + "#"*75)
    print("# EXECUTION SIT-HSS")
    if use_parallel:
        print("# MODE: TRAITEMENT PARALLELE")
    else:
        print("# MODE: TRAITEMENT SEQUENTIEL")
    print("#"*75)

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

    print(f"\nTrouve {len(image_paths)} images\n")

    results_dir = Path('results/sit_hss')
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    if use_parallel:
        sithss_results = process_all_sithss_parallel(
            image_paths=image_paths,
            n_segments=n_segments,
            t=t,
            tau=tau,
            max_radius=max_radius,
            gt_dir=gt_dir,
            n_workers=n_workers
        )

        if save_individual_images:
            print(f"\n{'='*75}")
            print("GENERATION DES IMAGES INDIVIDUELLES")
            print(f"{'='*75}\n")

            for i, (image_name, result) in enumerate(sithss_results.items(), 1):
                print(f"[{i}/{len(sithss_results)}] Generation image pour {image_name}...", flush=True)

                save_individual_segmentation_images(
                    result['image'],
                    result['labels'],
                    result['metrics'],
                    image_name,
                    results_dir,
                    result['metrics']['execution_time'],
                    result['gt_list']
                )

                all_results.append(result['metrics'])

    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for img_idx, img_path in enumerate(image_paths, 1):
            image_name = Path(img_path).stem

            print(f"\n{'█'*75}")
            print(f"Image {img_idx}/{len(image_paths)}: {image_name}")
            print(f"{'█'*75}")

            image = loader.load_image(img_path)

            sit_hss = SITHSS(
                n_segments=n_segments,
                t=t,
                tau=tau,
                max_radius=max_radius,
                device=device
            )

            start_time = time.time()
            labels = sit_hss.fit(image)
            elapsed_time = time.time() - start_time

            gt_list = None
            if gt_dir:
                try:
                    gt_all = loader.load_bsds500_groundtruth(split='global', image_name=image_name)
                    if len(gt_all) > 0:
                        gt_list = gt_all[0]
                except Exception as e:
                    pass

            if gt_list:
                metrics = compute_metrics_multiple_gt(labels, gt_list)
            else:
                metrics = compute_all_metrics(labels)

            metrics['method'] = 'SIT-HSS'
            metrics['image'] = image_name
            metrics['execution_time'] = elapsed_time
            metrics['n_segments_target'] = n_segments

            print(f"\n  === RESULTATS ===")
            print(f"  Superpixels: {int(metrics['n_superpixels'])}")
            print(f"  Temps: {elapsed_time:.3f}s")

            print(f"\n  QUALITE GEOMETRIQUE:")
            if 'global_regularity' in metrics:
                print(f"    GR (Global Regularity): {metrics['global_regularity']:.4f}")
            if 'compactness' in metrics:
                print(f"    CO (Compactness): {metrics['compactness']:.4f}")
            if 'regularity' in metrics:
                print(f"    RE (Regularity): {metrics['regularity']:.4f}")

            if 'boundary_recall' in metrics:
                print(f"\n  METRIQUES AVEC GT:")
                print(f"    BR (Boundary Recall): {metrics['boundary_recall']:.4f}")
                if 'precision' in metrics:
                    print(f"    P (Precision): {metrics['precision']:.4f}")
                if 'contour_density' in metrics:
                    print(f"    CD (Contour Density): {metrics['contour_density']:.4f}")
                if 'under_segmentation_error' in metrics:
                    print(f"    UE (Undersegmentation Error): {metrics['under_segmentation_error']:.4f}")
                if 'corrected_under_segmentation_error' in metrics:
                    print(f"    CUE (Corrected UE): {metrics['corrected_under_segmentation_error']:.4f}")
                if 'achievable_segmentation_accuracy' in metrics:
                    print(f"    ASA (Achievable Seg. Accuracy): {metrics['achievable_segmentation_accuracy']:.4f}")
                if 'explained_variation' in metrics:
                    print(f"    EV (Explained Variation): {metrics['explained_variation']:.4f}")

            if save_individual_images:
                save_individual_segmentation_images(
                    image, labels, metrics, image_name,
                    results_dir, elapsed_time, gt_list
                )

            all_results.append(metrics)

            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("\n" + "="*75)
    print("RESUME GLOBAL")
    print("="*75)

    df = pd.DataFrame(all_results)

    print(f"\nNombre d'images traitees: {len(all_results)}")
    print(f"\n{'='*75}")
    print("METRIQUES MOYENNES")
    print(f"{'='*75}")

    print(f"\nGENERALITES:")
    print(f"  Superpixels: {df['n_superpixels'].mean():.1f} ± {df['n_superpixels'].std():.1f}")
    print(f"  Temps execution: {df['execution_time'].mean():.3f}s ± {df['execution_time'].std():.3f}s")

    print(f"\nQUALITE GEOMETRIQUE:")
    if 'global_regularity' in df.columns:
        print(f"  GR (Global Regularity): {df['global_regularity'].mean():.4f} ± {df['global_regularity'].std():.4f}")
    if 'compactness' in df.columns:
        print(f"  CO (Compactness): {df['compactness'].mean():.4f} ± {df['compactness'].std():.4f}")
    if 'regularity' in df.columns:
        print(f"  RE (Regularity): {df['regularity'].mean():.4f} ± {df['regularity'].std():.4f}")

    if 'boundary_recall' in df.columns:
        print(f"\nMETRIQUES AVEC GROUND TRUTH:")
        print(f"  BR (Boundary Recall): {df['boundary_recall'].mean():.4f} ± {df['boundary_recall'].std():.4f}")
        if 'precision' in df.columns:
            print(f"  P (Precision): {df['precision'].mean():.4f} ± {df['precision'].std():.4f}")
        if 'contour_density' in df.columns:
            print(f"  CD (Contour Density): {df['contour_density'].mean():.4f} ± {df['contour_density'].std():.4f}")
        if 'under_segmentation_error' in df.columns:
            print(f"  UE (Undersegmentation Error): {df['under_segmentation_error'].mean():.4f} ± {df['under_segmentation_error'].std():.4f}")
        if 'corrected_under_segmentation_error' in df.columns:
            print(f"  CUE (Corrected UE): {df['corrected_under_segmentation_error'].mean():.4f} ± {df['corrected_under_segmentation_error'].std():.4f}")
        if 'achievable_segmentation_accuracy' in df.columns:
            print(f"  ASA (Achievable Seg. Accuracy): {df['achievable_segmentation_accuracy'].mean():.4f} ± {df['achievable_segmentation_accuracy'].std():.4f}")
        if 'explained_variation' in df.columns:
            print(f"  EV (Explained Variation): {df['explained_variation'].mean():.4f} ± {df['explained_variation'].std():.4f}")

    if save_results:
        csv_path = results_dir / 'sithss_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Resultats sauvegardes: {csv_path}")
        if save_individual_images:
            print(f"✓ Images individuelles sauvegardees dans: {results_dir}")

    print("\n" + "#"*75)
    print("# EXECUTION TERMINEE")
    print("#"*75)

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Execution SIT-HSS avec images individuelles'
    )
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Repertoire contenant les images')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Split du BSDS500')
    parser.add_argument('--n_segments', type=int, default=200,
                       help='Nombre de superpixels')
    parser.add_argument('--t', type=float, default=0.15,
                       help='Parametre t de SITHSS')
    parser.add_argument('--tau', type=float, default=1e-8,
                       help='Parametre tau de SITHSS')
    parser.add_argument('--max_radius', type=int, default=7,
                       help='Rayon maximal pour SITHSS')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Nombre maximum d\'images')
    parser.add_argument('--g', type=str, default=None,
                       help='Repertoire du ground truth')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Sauvegarder les resultats')
    parser.add_argument('--no_individual', action='store_true', default=False,
                       help='Ne pas sauvegarder les images individuelles')
    parser.add_argument('--no_parallel', action='store_true', default=False,
                       help='Desactiver le traitement parallele')
    parser.add_argument('--workers', type=int, default=None,
                       help='Nombre de workers pour le parallele (auto si non specifie)')

    args = parser.parse_args()

    execute_sithss(
        image_dir=args.image_dir,
        split=args.split,
        n_segments=args.n_segments,
        t=args.t,
        tau=args.tau,
        max_radius=args.max_radius,
        max_images=args.max_images,
        gt_dir=args.g,
        save_results=args.save,
        save_individual_images=not args.no_individual,
        use_parallel=not args.no_parallel,
        n_workers=args.workers
    )


if __name__ == "__main__":
    main()
