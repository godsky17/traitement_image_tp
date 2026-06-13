# Étape 1 — Prétraitement des images

> Module de préparation des images de chiffres manuscrits avant reconnaissance.

---

## Objectif

Transformer une image brute (photo de papier, scan) en une image **binaire propre**,
prête à être consommée par un modèle de reconnaissance de chiffres.

---

## Pipeline appliqué

```
Image originale (RGB)
        │
        ▼
Niveaux de gris          ← réduit 3 canaux → 1 canal
        │
        ▼
Filtre Gaussien 5×5      ← atténue le bruit haute fréquence
        │
        ▼
Binarisation Otsu        ← seuil automatique, chiffre blanc / fond noir
        │
        ▼
Ouverture morphologique  ← supprime les artefacts résiduels
        │
        ▼
Image binaire finale ✔
```

---

## Structure du projet

```
etape-1-pretraitement/
├── pretraitement.py        # Module principal (fonctions + pipeline)
├── test_pretraitement.py   # Tests unitaires (19 tests)
├── requirements.txt        # Dépendances Python
├── README.md               # Ce fichier
├── samples/                # Dossier des images à traiter  ← tes images ici
│   ├── chiffre_3.png
│   └── ...
└── sorties/                # Figures avant/après générées automatiquement
    ├── chiffre_3_avant_apres.png
    └── ...
```

---

## Installation

### Prérequis

- Python **3.10+**
- `pip` à jour

### 1. Cloner / télécharger les fichiers

Place tous les fichiers dans un même dossier.

### 2. (Recommandé) Créer un environnement virtuel

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Utilisation

### Traiter tes propres images

1. Place tes images (`.png`, `.jpg`, `.jpeg`) dans le dossier `samples/`
2. Lance le script :

```bash
python pretraitement.py
```

Les figures avant/après sont automatiquement sauvegardées dans `sorties/`.

### Utiliser le pipeline dans ton propre code

```python
from pretraitement import pretraiter_image, generer_capture_avant_apres

# Pipeline complet sur une image
resultats = pretraiter_image("samples/mon_chiffre.jpg")

# Accéder aux étapes intermédiaires
image_finale  = resultats["propre"]       # image prête pour le modèle ML
image_gris    = resultats["gris"]         # niveaux de gris
seuil         = resultats["seuil_otsu"]   # seuil calculé par Otsu

# Générer la figure avant/après
generer_capture_avant_apres(resultats, "mon_chiffre", dossier_sortie="sorties")
```

### Ajuster les paramètres

```python
resultats = pretraiter_image(
    "samples/mon_chiffre.jpg",
    taille_noyau_bruit=7,   # noyau Gaussien plus large → plus de lissage
    taille_noyau_morpho=5   # ouverture plus agressive → supprime plus d'artefacts
)
```

| Paramètre             | Valeur par défaut | Quand l'augmenter                        |
|-----------------------|:-----------------:|------------------------------------------|
| `taille_noyau_bruit`  | `5`               | Image très bruitée / grain de papier fort |
| `taille_noyau_morpho` | `3`               | Beaucoup d'artefacts résiduels après Otsu |

---

## Lancer les tests

```bash
# Avec pytest (recommandé)
pip install pytest
python -m pytest test_pretraitement.py -v

# Sans pytest
python test_pretraitement.py
```

Résultat attendu :

```
collected 19 items

test_pretraitement.py::TestChargerImage::test_chargement_valide          PASSED
test_pretraitement.py::TestChargerImage::test_fichier_inexistant         PASSED
test_pretraitement.py::TestConvertirNiveauxDeGris::test_dimensions_...   PASSED
...
============================== 19 passed in 1.49s ==============================
```

---

## Dépendances

| Package          | Version minimale | Rôle                                    |
|------------------|:----------------:|-----------------------------------------|
| `opencv-python`  | 4.13.0           | Traitement d'image (lecture, filtres, morphologie) |
| `numpy`          | 2.4.4            | Manipulation des tableaux de pixels     |
| `matplotlib`     | 3.10.8           | Génération des figures avant/après      |
| `scikit-image`   | 0.26.0           | Utilitaires image complémentaires       |
| `Pillow`         | 12.1.1           | Support de formats d'image additionnels |

---

## Formats d'image acceptés

`.png` · `.jpg` · `.jpeg`

Pour ajouter d'autres formats (`.bmp`, `.tiff`…), modifier la ligne suivante dans `pretraitement.py` :

```python
# Ligne ~130 dans la fonction main()
extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
```

---

## Prochaine étape

➜ **Étape 2 — Extraction de caractéristiques** : segmentation des chiffres,
normalisation de taille, préparation des tenseurs d'entrée pour le modèle ML.