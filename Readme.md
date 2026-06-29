# Projet OCR — Reconnaissance Automatique de Codes Postaux Manuscrits

> Système complet de traitement d'images et de classification supervisée pour la reconnaissance de chiffres manuscrits isolés ou collés.

---

## 🚀 Objectif du Projet

Ce projet vise à extraire et reconnaître automatiquement une suite de chiffres manuscrits (ex: un code postal) à partir d'une image brute (photo, scan). Il intègre un pipeline complet de Vision par Ordinateur et de Machine Learning divisé en trois phases majeures :

1. **Prétraitement & Nettoyage** (Filtre Gaussien, Binarisation d'Otsu, Morphologie).
2. **Segmentation & Extraction de caractéristiques** (Contours OpenCV, Fusion de boîtes englobantes robustes, Descripteurs HOG).
3. **Classification Multi-Modèles** (Entraînement et évaluation comparative de classifieurs **SVM** et **KNN**).

---

## 📊 Pipeline Global d'Exécution

```
      [Image Brute] (Photo / Scan dans samples/)
            │
            ▼
┌───────────────────────┐
│ 1. PRÉTRAITEMENT      │ ← Module 'pretraitement.py'
└───────────────────────┘
            │ 1. Niveaux de gris (Luminance)
            │ 2. Filtre Gaussien 5×5 (Réduction du bruit)
            │ 3. Binarisation Adaptative d'Otsu (Inversée)
            │ 4. Ouverture Morphologique (Nettoyage des poussières)
            ▼
      [Image Binaire Propre] (Sauvegardée dans resultats/)
            │
            ▼
┌───────────────────────┐
│ 2. SEGMENTATION ROBUSTE│ ← Algorithme 'fusionner_boites'
└───────────────────────┘
            │ • Détection des contours externes
            │ • Fusion des rectangles trop proches horizontalement
            │   (Résout la coupure des chiffres comme le '4' ou le '7')
            ▼
      [Chiffres Individuels Segmentés]
            │
            ▼
┌───────────────────────┐
│ 3. EXTRACTION HOG     │ ← skimage.feature.hog
└───────────────────────┘
            │ • Redimensionnement uniforme au format standard (28×28)
            │ • Extraction des vecteurs de caractéristiques de formes (HOG)
            ▼
┌───────────────────────┐
│ 4. CLASSIFICATION     │ ← Modèles entraînés et interchangeables
└───────────────────────┘
            ├───> [ Option 1 : SVM ] ───> Code Postal Prédit 🔮
            └───> [ Option 2 : KNN ] ───> Code Postal Prédit 🔮

```

---

## 📁 Structure du Projet mis à jour

```
TRAITEMENT_IMAGE/
│
├── pretraitement.py          # Module de nettoyage et binarisation d'images
├── entrainement.py          # Script d'extraction HOG et d'entraînement (SVM + KNN)
├── test.py                   # Script de test de reconnaissance via le Terminal (CLI)
│
├── test_pretraitement.py     # Tests unitaires du pipeline de prétraitement
├── requirements.txt          # Dépendances Python du projet
├── README.md                 # Documentation du projet
│
├── samples/                  # DOSSIER ENTRÉE : Images de chiffres pour l'entraînement
│   ├── 00000.jpg
│   └── 88888.jpg
│
├── resultats/                # Images intermédiaires binarisées (générées par pretraitement.py)
│   ├── 00000.jpg
│   └── 88888.jpg
│
├── modeles/                  # Fichiers binaires des modèles ML entraînés
│   ├── svm_chiffres.pkl      # Modèle Support Vector Machine
│   └── knn_chiffres.pkl      # Modèle k-Nearest Neighbors
│
├── data/
│   └── test/                 # DOSSIER TEST : Vos images manuscrites réelles à classifier
│       └── code_postal_test.jpg
│
└── sorties/                  # Graphiques d'analyse "Avant/Après" générés
    └── 88888_avant_apres.png

```

---

## 🛠️ Installation et Configuration

### Prérequis

* Python **3.10+**
* Gestionnaire de paquets `pip`

### 1. Environnement Virtuel

```bash
# Création de l'environnement venv
python -m venv venv

# Activation de l'environnement
# Sur Windows :
venv\Scripts\activate
# Sur Linux / macOS :
source venv/bin/activate

```

### 2. Installation des Dépendances

Installez l'ensemble des bibliothèques nécessaires (incluant désormais `scikit-learn` pour les modèles ML) :

```bash
pip install -r requirements.txt

```

---

## 💻 Guide d'Utilisation

Le projet s'exécute de manière séquentielle en 3 étapes claires :

### Étape 1 : Lancer le Prétraitement des images d'entraînement

Placez vos images composites de chiffres (ex: `88888.jpg`) dans `samples/` puis exécutez :

```bash
python pretraitement.py

```

*Effets :* Génère les figures comparatives dans `sorties/` et enregistre les images binaires nettoyées dans le dossier `resultats/`.

### Étape 2 : Entraîner les modèles (SVM & KNN)

Ce script extrait les descripteurs HOG à partir des images segmentées du dossier `resultats/` et entraîne simultanément les deux algorithmes :

```bash
python entrainement.py

```

*Effets :* Crée le dossier `modeles/` contenant `svm_chiffres.pkl` et `knn_chiffres.pkl`.

### Étape 3 : Tester la reconnaissance sur vos images réelles (CLI)

Placez vos images de test dans `data/test/`. Vous pouvez lancer la prédiction en choisissant dynamiquement votre méthode directement depuis votre terminal :

* **Tester avec le modèle SVM (par défaut) :**
```bash
python test.py svm

```


* **Tester avec le modèle KNN :**
```bash
python test.py knn

```



---

## ⚙️ Robustesse face à la Sur-Segmentation (Détails Techniques)

L'une des forces de cette version réside dans sa tolérance aux imperfections de l'écriture manuscrite humaine.

Lorsqu'un chiffre est écrit en traits disjoints (par exemple, la barre verticale d'un **4** ou d'un **7** détachée du reste), une détection de contours classique génère plusieurs rectangles pour un seul caractère, faussant le compteur et la prédiction.

Le projet intègre désormais une fonction algorithmique de **fusion de boîtes englobantes** (`fusionner_boites`) :

* Elle analyse l'espacement sur l'axe horizontal ($X$).
* Si deux contours se chevauchent ou sont situés à une distance inférieure à un seuil de tolérance configurable (par défaut `15 pixels`), ils sont automatiquement combinés en une seule entité cohérente avant d'être envoyés à l'extraction HOG.

---

## 📦 Liste des Dépendances Requises (`requirements.txt`)

Assurez-vous que votre fichier `requirements.txt` contient au minimum les packages suivants :

```text
opencv-python>=4.13.0
numpy>=2.4.4
matplotlib>=3.10.8
scikit-image>=0.26.0
scikit-learn>=1.4.0
joblib>=1.3.0
Pillow>=12.1.1

```