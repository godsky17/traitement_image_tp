"""
test_pretraitement.py
=====================
Tests unitaires du module de prétraitement d'images.

Couvre :
    - Chargement d'image (valide / invalide)
    - Conversion en niveaux de gris
    - Réduction du bruit
    - Binarisation Otsu
    - Nettoyage morphologique
    - Pipeline complet (pretraiter_image)
    - Génération de la figure avant/après

Utilisation :
    python -m pytest test_pretraitement.py -v
    # ou directement :
    python test_pretraitement.py

Auteur  : Ingénieur ML
Version : 1.0.0
"""

import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from pretraitement import (
    binariser_otsu,
    charger_image,
    convertir_niveaux_de_gris,
    generer_capture_avant_apres,
    nettoyer_morphologie,
    pretraiter_image,
    reduire_bruit,
)


# ---------------------------------------------------------------------------
# Utilitaires de test
# ---------------------------------------------------------------------------

def _creer_image_test(chemin: str, largeur: int = 64, hauteur: int = 64) -> np.ndarray:
    """
    Crée une image synthétique RGB (fond gris clair, carré sombre au centre)
    et la sauvegarde sur le disque.

    Args:
        chemin (str): Chemin de sauvegarde (.png).
        largeur (int): Largeur en pixels.
        hauteur (int): Hauteur en pixels.

    Returns:
        np.ndarray: Image BGR créée (telle qu'OpenCV la chargerait).
    """
    image = np.full((hauteur, largeur, 3), 220, dtype=np.uint8)  # fond gris clair
    # Dessiner un carré sombre au centre pour simuler un chiffre
    image[16:48, 16:48] = 30
    cv2.imwrite(chemin, image)
    return image


# ---------------------------------------------------------------------------
# Tests unitaires
# ---------------------------------------------------------------------------

class TestChargerImage(unittest.TestCase):
    """Tests de la fonction charger_image."""

    def setUp(self):
        self.dossier_tmp = tempfile.mkdtemp()
        self.chemin_valide = os.path.join(self.dossier_tmp, "test.png")
        _creer_image_test(self.chemin_valide)

    def tearDown(self):
        shutil.rmtree(self.dossier_tmp)

    def test_chargement_valide(self):
        """Vérifie qu'une image valide est chargée correctement."""
        image = charger_image(self.chemin_valide)
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(len(image.shape), 3)   # H × W × C
        self.assertEqual(image.shape[2], 3)      # 3 canaux BGR

    def test_fichier_inexistant(self):
        """Vérifie qu'une FileNotFoundError est levée si le fichier manque."""
        with self.assertRaises(FileNotFoundError):
            charger_image("/chemin/qui/nexiste/pas.png")


class TestConvertirNiveauxDeGris(unittest.TestCase):
    """Tests de la fonction convertir_niveaux_de_gris."""

    def test_sortie_2d(self):
        """Vérifie que la sortie est une image 2D (un seul canal)."""
        image_bgr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        gris = convertir_niveaux_de_gris(image_bgr)
        self.assertEqual(len(gris.shape), 2)

    def test_dimensions_preservees(self):
        """Vérifie que les dimensions H×W sont conservées après conversion."""
        image_bgr = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        gris = convertir_niveaux_de_gris(image_bgr)
        self.assertEqual(gris.shape, (100, 80))

    def test_plage_valeurs(self):
        """Vérifie que les valeurs restent dans [0, 255]."""
        image_bgr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        gris = convertir_niveaux_de_gris(image_bgr)
        self.assertGreaterEqual(gris.min(), 0)
        self.assertLessEqual(gris.max(), 255)


class TestReduireBruit(unittest.TestCase):
    """Tests de la fonction reduire_bruit."""

    def setUp(self):
        self.image_gris = np.random.randint(0, 255, (64, 64), dtype=np.uint8)

    def test_sortie_meme_forme(self):
        """Vérifie que le filtre Gaussien conserve la forme de l'image."""
        lissee = reduire_bruit(self.image_gris, taille_noyau=5)
        self.assertEqual(lissee.shape, self.image_gris.shape)

    def test_noyau_pair_leve_erreur(self):
        """Vérifie qu'un noyau pair déclenche une ValueError."""
        with self.assertRaises(ValueError):
            reduire_bruit(self.image_gris, taille_noyau=4)

    def test_lissage_reduit_variance(self):
        """Vérifie que le lissage réduit la variance (bruit atténué)."""
        lissee = reduire_bruit(self.image_gris, taille_noyau=5)
        self.assertLess(float(lissee.std()), float(self.image_gris.std()))


class TestBinariserOtsu(unittest.TestCase):
    """Tests de la fonction binariser_otsu."""

    def setUp(self):
        # Image bimodale simple : moitié noire, moitié blanche
        self.image_bimodale = np.zeros((64, 64), dtype=np.uint8)
        self.image_bimodale[:, 32:] = 255

    def test_valeurs_binaires(self):
        """Vérifie que l'image résultante ne contient que 0 et 255."""
        binaire, _ = binariser_otsu(self.image_bimodale)
        valeurs_uniques = set(np.unique(binaire))
        self.assertTrue(valeurs_uniques.issubset({0, 255}))

    def test_seuil_est_float(self):
        """Vérifie que le seuil retourné est un nombre flottant."""
        _, seuil = binariser_otsu(self.image_bimodale)
        self.assertIsInstance(seuil, float)

    def test_sortie_meme_forme(self):
        """Vérifie que la forme de l'image est conservée."""
        binaire, _ = binariser_otsu(self.image_bimodale)
        self.assertEqual(binaire.shape, self.image_bimodale.shape)


class TestNettoyerMorphologie(unittest.TestCase):
    """Tests de la fonction nettoyer_morphologie."""

    def setUp(self):
        # Image binaire avec un petit point de bruit isolé
        self.image_binaire = np.zeros((64, 64), dtype=np.uint8)
        self.image_binaire[30:50, 20:45] = 255  # région principale
        self.image_binaire[5, 5] = 255           # artefact isolé (1 pixel)

    def test_suppression_artefact(self):
        """Vérifie que le petit artefact isolé est supprimé."""
        propre = nettoyer_morphologie(self.image_binaire, taille_noyau=3)
        self.assertEqual(propre[5, 5], 0, "L'artefact isolé doit être supprimé.")

    def test_region_principale_preservee(self):
        """Vérifie que la région principale (chiffre) est conservée."""
        propre = nettoyer_morphologie(self.image_binaire, taille_noyau=3)
        self.assertGreater(propre[38, 30], 0, "La région principale doit rester visible.")

    def test_sortie_meme_forme(self):
        """Vérifie que la forme est conservée après nettoyage."""
        propre = nettoyer_morphologie(self.image_binaire)
        self.assertEqual(propre.shape, self.image_binaire.shape)


class TestPretraiterImage(unittest.TestCase):
    """Tests du pipeline complet pretraiter_image."""

    def setUp(self):
        self.dossier_tmp = tempfile.mkdtemp()
        self.chemin_image = os.path.join(self.dossier_tmp, "chiffre.png")
        _creer_image_test(self.chemin_image)

    def tearDown(self):
        shutil.rmtree(self.dossier_tmp)

    def test_cles_retournees(self):
        """Vérifie que toutes les clés attendues sont dans le résultat."""
        cles_attendues = {"originale", "gris", "lissee", "binaire", "propre", "seuil_otsu"}
        resultats = pretraiter_image(self.chemin_image)
        self.assertEqual(set(resultats.keys()), cles_attendues)

    def test_image_finale_binaire(self):
        """Vérifie que l'image finale ne contient que 0 et 255."""
        resultats = pretraiter_image(self.chemin_image)
        valeurs = set(np.unique(resultats["propre"]))
        self.assertTrue(valeurs.issubset({0, 255}))

    def test_seuil_otsu_dans_plage(self):
        """Vérifie que le seuil Otsu est dans [0, 255]."""
        resultats = pretraiter_image(self.chemin_image)
        self.assertGreaterEqual(resultats["seuil_otsu"], 0)
        self.assertLessEqual(resultats["seuil_otsu"], 255)


class TestGenererCaptureAvantApres(unittest.TestCase):
    """Tests de la génération de la figure avant/après."""

    def setUp(self):
        self.dossier_tmp = tempfile.mkdtemp()
        chemin_image = os.path.join(self.dossier_tmp, "chiffre.png")
        _creer_image_test(chemin_image)
        self.resultats = pretraiter_image(chemin_image)

    def tearDown(self):
        shutil.rmtree(self.dossier_tmp)

    def test_fichier_cree(self):
        """Vérifie que la figure PNG est bien créée sur le disque."""
        chemin = generer_capture_avant_apres(
            self.resultats, "test_figure", self.dossier_tmp
        )
        self.assertTrue(os.path.exists(chemin))

    def test_extension_png(self):
        """Vérifie que le fichier généré a bien l'extension .png."""
        chemin = generer_capture_avant_apres(
            self.resultats, "test_figure", self.dossier_tmp
        )
        self.assertTrue(chemin.endswith(".png"))


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)