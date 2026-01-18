# Segmentation Hiérarchique de Superpixels via Théorie de l'Information Structurelle (SIT-HSS)

**Projet M2 VMI - Modélisation de systèmes intelligents**  
**Etudiant: Yanis (SIT-HSS)**  

---

## Présentation du sujet

Ce projet implémente et évalue **SIT-HSS** (*Hierarchical Superpixel Segmentation via Structural Information Theory*), une méthode innovante de segmentation hiérarchique de superpixels basée sur la théorie de l'information structurelle. Les superpixels sont des regroupements de pixels adjacents partageant des caractéristiques visuelles similaires (couleur, texture), constituant une représentation intermédiaire entre les pixels bruts et la segmentation sémantique complète.

### Contexte et motivation

Malgré la dominance du deep learning en vision par ordinateur, les méthodes de superpixelisation conservent leur pertinence pour plusieurs raisons :

- **Efficacité computationnelle** : Réduction drastique du nombre de primitives à traiter
- **Interprétabilité** : Représentation mid-level compréhensible et manipulable
- **Apprentissage faiblement supervisé** : Support pour annotations partielles
- **Prétraitement pour réseaux de neurones** : Segmentation initiale pour architectures graph-based
- **Applications contraintes** : Imagerie satellite, systèmes embarqués où les ressources sont limitées

**Note importante** : Ce dépôt est une copie extraite du projet complet développé en collaboration (Yanis & Samy), disponible sur : https://github.com/Evowind/slic-hierarchical-superpixels  
Cette version contient l'implémentation SIT-HSS développée par Yanis.

---

## Article de référence

### Hierarchical Superpixel Segmentation via Structural Information Theory (2025)
**Minhui Xie, Hao Peng, et al.**  
[arXiv:2501.07069](https://arxiv.org/abs/2501.07069)  

**Contributions principales** :
- Stratégie de construction de graphe basée sur la maximisation de l'entropie structurelle 1D
- Algorithme de partitionnement hiérarchique guidé par la minimisation de l'entropie structurelle 2D
- Performances state-of-the-art sur trois benchmarks (BSDS500, SBD, PASCAL-S)
- Complexité linéaire O(|E|) avec support GPU

---

## Théorie de l'Information Structurelle

SIT-HSS repose sur la **théorie de l'information structurelle** qui quantifie la complexité et les patterns de connectivité d'un graphe via des arbres d'encodage.

### Concepts clés

**Entropie Structurelle** : Nombre moyen minimum de communautés nécessaires pour encoder les nœuds accessibles lors d'une marche aléatoire sur le graphe.

**Entropie 1D** (H⁽¹⁾) : Mesure l'information contenue dans le graphe
- Utilisée pour la construction du graphe
- Maximiser H⁽¹⁾ → retenir plus d'information

**Entropie 2D** (H⁽²⁾) : Mesure la qualité du partitionnement
- Utilisée pour le partitionnement hiérarchique
- Minimiser H⁽²⁾ → partitions optimales

---

## Méthode SIT-HSS

### 1. Construction du Graphe (Maximisation H⁽¹⁾)

**Principe** : Contrairement aux méthodes classiques qui connectent uniquement les pixels adjacents directs, SIT-HSS explore progressivement le voisinage pour capturer les relations non-adjacentes.

**Algorithme** :

1. **Initialisation** : Graphe sans arêtes, pixels = nœuds
2. **Expansion progressive** : 
   - Rayon r = 1 : 8 voisins directs
   - Rayon r = 2, 3, ... : voisinage étendu
3. **Critère d'arrêt** : ΔH⁽¹⁾ < τ (plateau d'entropie)
4. **Sélection** : Rayon optimal qui maximise l'information retenue

**Distance entre pixels** :
```
ρᵢⱼ = ||cᵢ - cⱼ||² · ||sᵢ - sⱼ||²
```
où :
- `cᵢ, cⱼ` : caractéristiques couleur (espace Lab)
- `sᵢ, sⱼ` : coordonnées spatiales

**Poids normalisés** :
```
Wᵢⱼ = exp(-ρᵢⱼ / (t · moyenne(ρ)))
```

### 2. Partitionnement Hiérarchique (Minimisation H⁽²⁾)

**Principe** : Fusion itérative des partitions pour minimiser l'entropie structurelle 2D.

**Algorithme** :

1. **Initialisation** : Chaque pixel = partition indépendante
2. **Itération** :
   - Pour chaque superpixel pᵢ
   - Trouver voisin adjacent pⱼ maximisant ΔH⁽²⁾
   - Fusionner pᵢ et pⱼ
3. **Arrêt** : Nombre de partitions = K (taille cible)

**Réduction d'entropie** :
```
ΔH⁽²⁾ₚᵢ,ₚⱼ = H⁽²⁾_avant - H⁽²⁾_après
```

**Contrainte de connectivité** : Fusion uniquement entre superpixels spatialement adjacents.

---

## Métriques d'évaluation

### Métriques avec Ground Truth

**Achievable Segmentation Accuracy (ASA)** : Précision théorique maximale
- Assigne chaque superpixel à la classe GT majoritaire
- Plage : [0, 1], plus élevé = meilleur

**Boundary Recall (BR)** : Proportion de contours GT correctement détectés
- Formule : `BR = contours_detectes / total_contours_GT`
- Plage : [0, 1], plus élevé = meilleur

**Under-segmentation Error (UE)** : Débordement des superpixels
- Mesure les pixels débordant hors des segments GT
- Plage : [0, ∞], plus bas = meilleur

**Explained Variation (EV)** : Variation expliquée par les superpixels
- Mesure la cohérence interne des superpixels
- Plage : [0, 1], plus élevé = meilleur

### Métriques intrinsèques (sans GT)

**Compactness (CO)** : Mesure la régularité des formes
- Formule : `C = 4π × aire / périmètre²`
- Plage : [0, 1], 1.0 = cercle parfait

**Regularity (RE)** : Uniformité des tailles
- Basée sur coefficient de variation des tailles
- Plage : [0, 1], 1.0 = toutes tailles identiques

**Global Regularity (GR)** : Uniformité de la grille spatiale
- Basée sur écart-type des distances inter-centres
- Plage : [0, 1], 1.0 = grille parfaitement régulière

---

## Résultats

### Visualisation
**Images BSDS500** (K = 200 superpixels)

![Exemple SIT-HSS](results/sit_hss/103029_sithss.png)

**Observations** :

* SIT-HSS capture efficacement les frontières des objets grâce à la prise en compte des relations non-adjacentes
* Excellente adhérence aux contours
* Gestion des petits objets et détails fins
* Superpixels cohérents avec les structures sémantiques de l'image

---

### Métriques quantitatives

#### Sans ground truth

| Métrique          | Scores        |
| ----------------- | ------------- |
| Compacité         | 0.7477        |
| Régularité        | 0.4507        |
| Global Regularity | 0.5944        |

#### Avec ground truth

| Métrique                       | Scores        |
| ------------------------------ | ------------- |
| Boundary Recall (BR)           | 0.8925        |
| Under-segmentation Error (UE)  | 0.4479        |
| Corrected UE                   | 0.4479        |
| Achievable Seg. Accuracy (ASA) | 0.9720        |
| Precision (P)                  | 0.0760        |
| Contour Density (CD)           | 7.5411        |
| Explained Variation            | 0.9681        |

### Temps d'exécution

Image : 481 × 321

Superpixels : 200 

* SIT-HSS : 132.196s

### Remarque

Le nombre de superpixels voulus est important notamment par rapport au nombre de rayons testés lors de la construction du graphe. En effet moins on a de superpixels plus ils sont grands et inversement, ainsi lorsque l'on en a moins on a besoin de plus d'informations donc d'un rayon plus grand. Cependant plus le rayon est important plus le temps d'exécution sera long, il faut donc faire un compromis sur le entre essayer de trouver le nombre de rayons optimal et une borne maximum à ne pas dépasser pour ne pas impacter trop lourdement le temps d'exécution. Avec ceci il se peut que dans certains cas où le rayon optimal est grand pour une image donnée on est pas les meilleurs résultats possibles. 

---

## Evaluation qualitative (12 testeurs)

### Protocole

Douze testeurs ont évalué **5 images** en utilisant **trois méthodes de segmentation en superpixels** : **SLIC**, **SLIC_IPOL** et **SIT-HSS**.
Chaque méthode a été notée selon les critères suivants (échelle de 1 à 5) :

1. **Qualité des contours** : respect des frontières naturelles des objets
2. **Régularité et uniformité** : homogénéité de la taille et de la forme des superpixels
3. **Cohérence chromatique** : similarité des couleurs à l'intérieur d'un même superpixel
4. **Equilibre global** : méthode jugée la plus satisfaisante pour chaque image

Les réponses manquantes (questions modifiées en cours de formulaire) ont été ignorées dans les calculs.

---

### Profil des testeurs

* **9 étudiants en Vision par Ordinateur** (75 %)
* **3 testeurs issus d'autres spécialités** (25 %)
* **4 testeurs avec expérience préalable des superpixels** (33 %)
* **8 testeurs sans expérience préalable** (67 %)

---

### Scores moyens par méthode (échelle 1–5)

| Critère                 | SLIC     | SLIC_IPOL | SIT-HSS  |
| ----------------------- | -------- | --------- | -------- |
| Qualité des contours    | 3.27     | 3.37      | **3.55** |
| Régularité / uniformité | **3.32** | 3.33      | 3.20     |
| Cohérence chromatique   | 3.28     | 3.28      | **3.47** |
| **Moyenne générale**    | **3.29** | **3.33**  | **3.41** |

**Analyse SIT-HSS** :
- **Meilleurs scores** en qualité des contours (+5.3% vs SLIC_IPOL)
- **Meilleurs scores** en cohérence chromatique (+5.8% vs SLIC/SLIC_IPOL)
- Légèrement inférieur en régularité (-3.6% vs SLIC)
- **Score global le plus élevé** (+2.4% vs SLIC_IPOL, +3.6% vs SLIC)

---

### Préférences globales

#### Classement global des méthodes

(1 = meilleure méthode, 3 = moins performante)

| Méthode       | Rang moyen | 
| ------------- | ---------- |
| **SLIC**      | **1.58**   |
| **SIT-HSS**   | 1.83       |
| **SLIC_IPOL** | 2.58       |

Bien que SIT-HSS obtienne de meilleurs scores qualitatifs, SLIC est la méthode la plus fréquemment bien classée, suggérant une perception de robustesse et de régularité.

---

### Meilleur équilibre par image

(**60 évaluations au total**)

| Méthode               | Votes  | Pourcentage |
| --------------------- | ------ | ----------- |
| **SIT-HSS**           | **34** | **56.7 %**  |
| SLIC_IPOL             | 12     | 20.0 %      |
| SLIC                  | 10     | 16.7 %      |
| Aucune ne se démarque | 4      | 6.6 %       |

SIT-HSS est majoritairement perçue comme offrant le meilleur compromis global.

---

### Comparaison SIT-HSS vs SLIC/SLIC_IPOL

#### SIT-HSS vs SLIC

| Critère               | SIT-HSS meilleur | SLIC meilleur | Equivalent |
| --------------------- | ---------------- | ------------- | ---------- |
| Qualité des contours  | **58%**          | 19%           | 23%        |
| Régularité            | 24%              | **51%**       | 25%        |
| Cohérence chromatique | **53%**          | 22%           | 25%        |

#### SIT-HSS vs SLIC_IPOL

| Critère               | SIT-HSS meilleur | SLIC_IPOL meilleur | Equivalent |
| --------------------- | ---------------- | ------------------ | ---------- |
| Qualité des contours  | **62%**          | 15%                | 23%        |
| Régularité            | 29%              | 34%                | 37%        |
| Cohérence chromatique | **56%**          | 18%                | 26%        |

---

### Commentaires qualitatifs récurrents

**SLIC**

* Bon équilibre général
* Régularité visuelle
* Couleurs perçues comme stables

**SLIC_IPOL**

* Contours plus précis sur certaines images
* Meilleure perception de relief
* Capacité à faire ressortir des détails fins spécifiques

**SIT-HSS**

* Fidélité accrue à la structure de l’image
* Meilleure séparation des objets
* Détails et contrastes plus marqués

**Limitations observées**

* SLIC : fusion d’éléments distincts, contours externes parfois flous
* SLIC_IPOL : résultats inconstants selon les images
* SIT-HSS : sur-segmentation locale dans certains cas

---

### Applications suggérées

* Imagerie satellite (détection, analyse de scènes)
* Traitement artistique et stylisation
* Analyse médicale (simulation de troubles visuels)
* Analyse picturale et décomposition de compositions

---

### Limites de l’évaluation

Cette évaluation repose sur un nombre limité de testeurs et d’images, ce qui restreint la portée statistique des résultats. Les jugements restent subjectifs et dépendent fortement du contenu des images évaluées. Enfin, certaines réponses (5) ont été ignorées en raison de modifications du formulaire en cours d’évaluation, ce qui peut introduire un léger biais.


---

### Conclusion de l'évaluation qualitative


1. **SIT-HSS est la méthode la plus appréciée globalement**, notamment pour la qualité visuelle perçue
2. **SLIC et SLIC_IPOL présentent des performances moyennes très proches**
3. **SLIC se distingue par sa régularité et sa stabilité**
4. **SLIC_IPOL offre de meilleurs contours dans certains cas spécifiques**
5. **Aucune méthode n’est universellement supérieure** : le choix dépend fortement du type d’image
6. Le protocole a été jugé clair et compréhensible par les testeurs (note de 4.17/5)

---

## Structure du code

```
slic-ipol-implementation/
├── src/
│   ├── methods/
│   │   └── slic/
│   │       ├── hierarchical_seg.py   # Implémentation SIT-HSS original
│   │       └── utils.py              # Utilitaires 
│   ├── evaluation/
│   │   ├── metrics.py                # Métriques (BR, UE, ASA, etc.)
│   │   └── visualize.py              # Fonctions de visualisation
│   ├── preprocessing/
│   │   └── image_loader.py           # Chargement BSDS500
│   └── utils/
│       ├── color_space.py            # Conversions RGB/Lab
│       └── distance.py               # Distance
├── experiments/
│   ├── parameter_tuning.py           # Optimisation paramètres
│   └── run_hierarchical.py           # Exécution simple
├── data/
│   └── BSDS500/                      # Dataset (non inclus)
├── images/                           # Sortie démo
├── results/                          # Résultats générés + Etude Humain
├── quick_start_hss.py                # Démo rapide
└── requirements.txt                  # Dépendances
```

---

## Installation et utilisation

### Prérequis

```bash
python >= 3.8
torch >= 1.10.0
CUDA (optionnel, recommandé)
```

### Installation

```bash
# Cloner le dépôt
git clone git@github.com:Kemoory/SIT-HSS-evaluation.git
cd SIT-HSS-evaluation

# Créer environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows

# Installer dépendances
pip install -r requirements.txt
```

### Dataset BSDS500

Télécharger depuis : [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)

Extraire dans `data/BSDS500/`

### Exemples d'utilisation

**1. Démo rapide**

```bash
python quick_start_hss.py
```

**2. Execution SIT-HSS sur 3 images (test)**

```bash
python experiments/run_hierarchical.py --split test --max_images 3 --g data --save
```

**3. Execution complète sur split de test**

```bash
python experiments/run_hierarchical.py --split test --g data --save
```

**4. Avec paramètres personnalisés**

```bash
python experiments/run_hierarchical.py \
    --split test \
    --n_segments 600 \
    --t 0.1 \
    --tau 2e-7 \
    --max_radius 7 \
    --g data \
    --save
```

**6. Utilisation dans votre code**

```python
from src.methods.hierarchical.hierarchical_seg import SITHSS
from PIL import Image
import numpy as np

# Charger image
image = np.array(Image.open('image.jpg'))

# SIT-HSS
sithss = SITHSS(
    n_segments=600,
    t=0.1,
    tau=2e-7,
    max_radius=7,
    device='cuda'  # ou 'cpu'
)
labels = sithss.fit(image)
```

---

## Avantages de SIT-HSS

### Par rapport aux méthodes classiques

| Aspect | Méthodes classiques | SIT-HSS |
|--------|-------------------|---------|
| **Information captée** | Relations adjacentes uniquement | Relations non-adjacentes + globales |
| **Construction graphe** | Statique, rayon fixe | Adaptive, maximisation H⁽¹⁾ |
| **Partitionnement** | Heuristiques simples | Optimisation théorique (H⁽²⁾) |

---

## Conclusion

### Synthèse

1. **SIT-HSS surpasse les méthodes state-of-the-art** sur tous les critères évalués selon l'article
2. **Innovation théorique** : entropie structurelle effective pour superpixelisation
3. **Bon compromis qualité/efficacité** comparable aux méthodes les plus rapides (si bien paramétré et optimisé)
4. **Validation expérimentale** : métriques quantitatives et évaluation humaine qui sont très encourageantes vis-à-vis de l'efficacité de la méthode

### Pertinence à l'ère de l'IA

Les superpixels via SIT-HSS restent pertinents car :
- **Complémentarité avec DL** : prétraitement optimal pour Graph Neural Networks
- **Interprétabilité** : arbres d'encodage explicites
- **Efficacité** : ressources limitées (edge computing, satellite)

### Perspectives

- Intégration dans pipelines deep learning (graph pooling)
- Extension aux vidéos (cohérence temporelle via H⁽³⁾)
- Optimisation multi-GPU pour très grandes images
- Adaptation domaines spécifiques (médical, satellite)

---

## Références

[1] Xie, M., Peng, H., et al. "Hierarchical Superpixel Segmentation via Structural Information Theory", arXiv:2501.07069, 2025

[5] BSDS500 Dataset: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html

[6] Code source officiel: https://github.com/SELGroup/SIT-HSS

---

## Contact

Pour toute question concernant cette implémentation :
- Ouvrir une issue sur GitHub
- Consulter la documentation dans `docs/`
- Voir le projet complet : https://github.com/Evowind/slic-hierarchical-superpixels

---

**Licence** : MIT

**Acknowledgments** : Ce travail s'inscrit dans le cadre du cours de Modélisation de systèmes intelligents du M2 VMI. Merci aux auteurs des articles originaux et à l'équipe IPOL pour leurs travaux de référence.
