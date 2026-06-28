import os
import sys
import cv2
import numpy as np
import joblib
from pathlib import Path
from skimage.feature import hog

# Importation directe du pipeline de prétraitement du Binôme 1
from pretraitement import pretraiter_image

def fusionner_boites(rectangles, tolerance=15):
    """
    Fusionne les rectangles de contours trop proches sur l'axe horizontal (X)
    pour éviter la sur-segmentation (ex: un '4' ou un '7' écrit en deux traits disjoints).
    """
    if not rectangles:
        return []
        
    # Crucial : On trie de gauche à droite selon l'axe X pour lire dans le bon ordre
    rectangles = sorted(rectangles, key=lambda r: r[0])
    fusionnes = []
    
    cx, cy, cw, ch = rectangles[0]
    for nx, ny, nw, nh in rectangles[1:]:
        # Si le rectangle suivant commence avant la fin du précédent + la tolérance en pixels
        if nx <= cx + cw + tolerance:
            x_min = min(cx, nx)
            y_min = min(cy, ny)
            x_max = max(cx + cw, nx + nw)
            y_max = max(cy + ch, ny + nh)
            cx, cy, cw, ch = x_min, y_min, x_max - x_min, y_max - y_min
        else:
            fusionnes.append((cx, cy, cw, ch))
            cx, cy, cw, ch = nx, ny, nw, nh
            
    fusionnes.append((cx, cy, cw, ch))
    return fusionnes

def main():
    # 1. Gestion dynamique du choix de l'algorithme via le terminal
    # Par défaut, on utilise le SVM si l'utilisateur ne précise rien
    CHOIX_ALGO = "svm" 
    
    if len(sys.argv) > 1:
        argument = sys.argv[1].lower()
        if argument in ["svm", "knn"]:
            CHOIX_ALGO = argument
        else:
            print(f"[⚠️] '{argument}' n'est pas un algorithme valide. Choix possibles : 'svm' ou 'knn'.")
            print("[→] Utilisation de 'svm' par défaut.\n")

    DOSSIER_TEST = os.path.join("data", "test")
    TAILLE_CIBLE = (28, 28)
    CHEMIN_MODELE = f"modeles/{CHOIX_ALGO}_chiffres.pkl"
    
    print("-" * 60)
    print(f"  Phase de Test — Méthode Sélectionnée : [ {CHOIX_ALGO.upper()} ]")
    print("-" * 60)
    
    # 2. Vérification et chargement du modèle sélectionné
    if not os.path.exists(CHEMIN_MODELE):
        print(f"[✘] Erreur : Le fichier '{CHEMIN_MODELE}' est introuvable. Veuillez d'abord lancer 'entrainement.py'.")
        return
        
    modele = joblib.load(CHEMIN_MODELE)
    print(f"    [✔] Modèle {CHOIX_ALGO.upper()} chargé avec succès.")
    
    # 3. Récupération des images présentes dans le dossier de test
    extensions = (".png", ".jpg", ".jpeg")
    images_test = [p for p in Path(DOSSIER_TEST).iterdir() if p.suffix.lower() in extensions]
    
    if not images_test:
        print(f"    [⚠️] Aucune image trouvée dans '{DOSSIER_TEST}'. Déposez-y vos images à tester.")
        return
        
    # 4. Traitement et prédiction pour chaque image de test
    for chemin in images_test:
        print(f"\n[→] Analyse de l'image : {chemin.name}")
        
        # Appel automatique du pipeline du Binôme 1
        resultats = pretraiter_image(str(chemin))
        img_propre = resultats["propre"]  # Image binarisée (chiffres blancs sur fond noir)
        
        # Détection des contours des formes blanches
        contours, _ = cv2.findContours(img_propre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Filtrage des micro-bruits parasites (formes trop petites)
            if w > 5 and h > 10:
                rectangles.append((x, y, w, h))
        
        # Application de l'algorithme de fusion de boîtes englobantes
        rectangles_fusionnes = fusionner_boites(rectangles, tolerance=15)
        print(f"    [📊] Nombre de chiffres détectés après fusion : {len(rectangles_fusionnes)}")
        
        code_predit = []
        
        # Extraction, redimensionnement et classification de chaque chiffre isolé
        for (x, y, w, h) in rectangles_fusionnes:
            # Extraction de la Région d'Intérêt (ROI) du chiffre courant
            roi = img_propre[y:y+h, x:x+w]
            
            # Ajout d'une petite marge noire (padding) tout autour du chiffre
            # Cela évite les distorsions brutales et aide l'extraction HOG à capter la forme
            pad = 4
            roi_decore = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            
            # Redimensionnement uniforme au format 28x28 attendu par nos classifieurs
            img_redimensionnee = cv2.resize(roi_decore, TAILLE_CIBLE, interpolation=cv2.INTER_AREA)
            
            # Extraction des caractéristiques de formes HOG
            features = hog(img_redimensionnee, orientations=9, pixels_per_cell=(4, 4),
                           cells_per_block=(2, 2), visualize=False).reshape(1, -1)
            
            # Prédiction par le modèle chargé (SVM ou KNN)
            prediction = modele.predict(features)[0]
            code_predit.append(str(prediction))
            
        # Assemblage final et affichage du code postal détecté
        resultat_final = "".join(code_predit)
        print(f"    [🔮] RÉSULTAT DU CODE POSTAL : {resultat_final}")
        print("-" * 60)

if __name__ == "__main__":
    main()