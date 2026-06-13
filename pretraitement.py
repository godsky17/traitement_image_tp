"""
pretraitement.py
================
Module de prétraitement d'images pour la reconnaissance de chiffres manuscrits.

Étapes appliquées :
    1. Chargement de l'image originale (RGB / couleur)
    2. Conversion en niveaux de gris
    3. Réduction du bruit par filtre Gaussien
    4. Binarisation adaptative (méthode d'Otsu)
    5. Nettoyage morphologique (ouverture) pour éliminer les artefacts résiduels

Auteur  : Ingénieur ML
Version : 1.0.0
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # backend sans affichage (scripts / serveurs)
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# Fonctions de prétraitement
# ---------------------------------------------------------------------------

def charger_image(chemin: str) -> np.ndarray:
    """
    Charge une image depuis le disque en format BGR (OpenCV).

    Args:
        chemin (str): Chemin complet vers le fichier image.

    Returns:
        np.ndarray: Image chargée en BGR (uint8).

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si l'image ne peut pas être lue.
    """
    if not os.path.exists(chemin):
        raise FileNotFoundError(f"Image introuvable : {chemin}")

    image = cv2.imread(chemin)
    if image is None:
        raise ValueError(f"Impossible de lire l'image : {chemin}")

    return image


def convertir_niveaux_de_gris(image_bgr: np.ndarray) -> np.ndarray:
    """
    Convertit une image BGR en niveaux de gris.

    La conversion utilise la formule de luminance perceptuelle pondérée :
        Gris = 0.114·B + 0.587·G + 0.299·R
    Cette pondération respecte la sensibilité de l'œil humain aux couleurs.

    Args:
        image_bgr (np.ndarray): Image couleur (H, W, 3) en BGR.

    Returns:
        np.ndarray: Image en niveaux de gris (H, W), uint8.
    """
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def reduire_bruit(image_gris: np.ndarray, taille_noyau: int = 5) -> np.ndarray:
    """
    Applique un filtre Gaussien pour réduire le bruit haute fréquence.

    Le flou Gaussien lisse les variations brutales de pixels (bruit de capteur,
    grain de papier, poussières) sans trop dégrader les contours du chiffre.

    Args:
        image_gris (np.ndarray): Image en niveaux de gris.
        taille_noyau (int): Taille du noyau Gaussien (doit être impair). Défaut : 5.

    Returns:
        np.ndarray: Image lissée (uint8).

    Raises:
        ValueError: Si taille_noyau est pair.
    """
    if taille_noyau % 2 == 0:
        raise ValueError("taille_noyau doit être un entier impair (ex: 3, 5, 7).")

    return cv2.GaussianBlur(image_gris, (taille_noyau, taille_noyau), sigmaX=0)


def binariser_otsu(image_lissee: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Binarise l'image par la méthode d'Otsu.

    La méthode d'Otsu cherche automatiquement le seuil optimal qui maximise
    la variance inter-classes (pixels sombres vs pixels clairs). Elle est
    particulièrement adaptée aux images à histogramme bimodal (fond clair /
    encre sombre).

    L'image résultante est inversée (THRESH_BINARY_INV) : le chiffre devient
    blanc (255) et le fond devient noir (0), convention standard pour la
    reconnaissance de formes.

    Args:
        image_lissee (np.ndarray): Image en niveaux de gris lissée.

    Returns:
        tuple[np.ndarray, float]:
            - Image binaire (uint8), chiffre blanc sur fond noir.
            - Seuil calculé automatiquement par Otsu.
    """
    seuil, image_binaire = cv2.threshold(
        image_lissee,
        0,                          # valeur ignorée avec OTSU
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return image_binaire, seuil


def nettoyer_morphologie(image_binaire: np.ndarray,
                          taille_noyau: int = 3,
                          iterations: int = 1) -> np.ndarray:
    """
    Applique une ouverture morphologique pour supprimer les petits artefacts.

    L'ouverture = érosion suivie d'une dilatation.
    - L'érosion supprime les petits points isolés (poussières, micro-taches).
    - La dilatation restaure la taille des régions conservées.

    Args:
        image_binaire (np.ndarray): Image binaire (uint8).
        taille_noyau (int): Taille de l'élément structurant carré. Défaut : 3.
        iterations (int): Nombre de fois que l'opération est appliquée. Défaut : 1.

    Returns:
        np.ndarray: Image nettoyée (uint8).
    """
    noyau = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (taille_noyau, taille_noyau)
    )
    return cv2.morphologyEx(
        image_binaire,
        cv2.MORPH_OPEN,
        noyau,
        iterations=iterations
    )


def pretraiter_image(chemin: str,
                      taille_noyau_bruit: int = 5,
                      taille_noyau_morpho: int = 3) -> dict:
    """
    Pipeline complet de prétraitement d'une image de chiffre manuscrit.

    Enchaîne toutes les étapes dans l'ordre :
        1. Chargement
        2. Niveaux de gris
        3. Réduction du bruit (Gaussien)
        4. Binarisation (Otsu)
        5. Nettoyage morphologique (ouverture)

    Args:
        chemin (str): Chemin vers l'image source.
        taille_noyau_bruit (int): Noyau du filtre Gaussien. Défaut : 5.
        taille_noyau_morpho (int): Noyau de l'opération morphologique. Défaut : 3.

    Returns:
        dict: Dictionnaire contenant toutes les étapes intermédiaires :
            - "originale"  : image BGR chargée
            - "gris"       : image en niveaux de gris
            - "lissee"     : image après filtre Gaussien
            - "binaire"    : image binarisée (Otsu)
            - "propre"     : image après nettoyage morphologique (résultat final)
            - "seuil_otsu" : seuil calculé par Otsu
    """
    originale = charger_image(chemin)
    gris      = convertir_niveaux_de_gris(originale)
    lissee    = reduire_bruit(gris, taille_noyau=taille_noyau_bruit)
    binaire, seuil = binariser_otsu(lissee)
    propre    = nettoyer_morphologie(binaire, taille_noyau=taille_noyau_morpho)

    return {
        "originale":  originale,
        "gris":       gris,
        "lissee":     lissee,
        "binaire":    binaire,
        "propre":     propre,
        "seuil_otsu": seuil,
    }


# ---------------------------------------------------------------------------
# Visualisation avant / après
# ---------------------------------------------------------------------------

def generer_capture_avant_apres(resultats: dict,
                                  nom_fichier: str,
                                  dossier_sortie: str = ".") -> str:
    """
    Génère et sauvegarde une figure comparative « avant / après » en 5 étapes.

    Args:
        resultats (dict): Dictionnaire retourné par `pretraiter_image`.
        nom_fichier (str): Nom de base du fichier de sortie (sans extension).
        dossier_sortie (str): Dossier où sauvegarder la figure. Défaut : ".".

    Returns:
        str: Chemin absolu de la figure sauvegardée.
    """
    os.makedirs(dossier_sortie, exist_ok=True)

    # Convertir BGR → RGB pour Matplotlib
    originale_rgb = cv2.cvtColor(resultats["originale"], cv2.COLOR_BGR2RGB)

    etapes = [
        (originale_rgb,         "1. Originale (RGB)",         "viridis"),
        (resultats["gris"],     "2. Niveaux de gris",         "gray"),
        (resultats["lissee"],   "3. Après filtre Gaussien",   "gray"),
        (resultats["binaire"],  f"4. Binarisation Otsu\n(seuil={resultats['seuil_otsu']:.0f})", "gray"),
        (resultats["propre"],   "5. Nettoyage morphologique\n(résultat final)", "gray"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(
        f"Pipeline de prétraitement — {nom_fichier}",
        fontsize=14, fontweight="bold", y=1.02
    )

    for ax, (image, titre, cmap) in zip(axes, etapes):
        if len(image.shape) == 3:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap=cmap)
        ax.set_title(titre, fontsize=9)
        ax.axis("off")

    plt.tight_layout()

    chemin_sortie = os.path.join(dossier_sortie, f"{nom_fichier}_avant_apres.png")
    plt.savefig(chemin_sortie, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return os.path.abspath(chemin_sortie)


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def main():
    """
    Traite un ensemble d'images de test et génère les captures avant/après.
    """
    DOSSIER_ENTREE = "samples"
    DOSSIER_SORTIE = "sorties"

    # Récupérer toutes les images PNG/JPG du dossier
    extensions = (".png", ".jpg", ".jpeg")
    images = [
        str(p) for p in Path(DOSSIER_ENTREE).iterdir()
        if p.suffix.lower() in extensions
    ]

    if not images:
        print(f"[AVERTISSEMENT] Aucune image trouvée dans '{DOSSIER_ENTREE}'.")
        return

    print(f"{'='*55}")
    print(f"  Pipeline de prétraitement — {len(images)} image(s) détectée(s)")
    print(f"{'='*55}\n")

    for chemin in sorted(images):
        nom = Path(chemin).stem
        print(f"[→] Traitement de : {chemin}")

        resultats = pretraiter_image(chemin)

        print(f"    ✔ Niveaux de gris       : OK")
        print(f"    ✔ Filtre Gaussien (5×5) : OK")
        print(f"    ✔ Binarisation Otsu     : seuil = {resultats['seuil_otsu']:.1f}")
        print(f"    ✔ Nettoyage morpho      : OK")

        chemin_figure = generer_capture_avant_apres(resultats, nom, DOSSIER_SORTIE)
        print(f"    ✔ Figure sauvegardée    : {chemin_figure}\n")

    print(f"{'='*55}")
    print(f"  Prétraitement terminé. Résultats dans : {DOSSIER_SORTIE}/")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()