import os
import cv2
import numpy as np
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier  # Étape KNN
from skimage.feature import hog

def fusionner_boites(rectangles, tolerance=15):
    if not rectangles:
        return []
    rectangles = sorted(rectangles, key=lambda r: r[0])
    fusionnes = []
    cx, cy, cw, ch = rectangles[0]
    for nx, ny, nw, nh in rectangles[1:]:
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

def preparer_donnees(dossier_images: str, taille_cible=(28, 28)):
    X, y = [], []
    # Extension de la liste pour couvrir toutes les variantes possibles de la promo
    extensions = (".png", ".jpg", ".jpeg", ".jfif", ".bmp")
    chemins = [p for p in Path(dossier_images).iterdir() if p.suffix.lower() in extensions]
    
    if not chemins:
        raise FileNotFoundError(f"Aucune image trouvée dans '{dossier_images}'.")
        
    for chemin in chemins:
        # CORRECTION 1 : Sécurisation de la nomenclature (évite le crash avec les lettres comme 'G')
        premier_caractere = chemin.stem[0]
        if not premier_caractere.isdigit():
            print(f"[ATTENTION] Fichier ignoré (nomenclature incorrecte) : {chemin.name}")
            continue
            
        label = int(premier_caractere)
        
        # CORRECTION 2 : Lecture binaire robuste pour supporter les accents (é, è, ç...) sous Windows
        try:
            img = cv2.imdecode(np.fromfile(str(chemin), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            print(f"[ERREUR] Impossible de charger le fichier {chemin.name}. Détails : {e}")
            continue
            
        # CORRECTION 3 : Sécurité anti-image corrompue (évite le plantage de findContours)
        if img is None:
            print(f"[ERREUR] Image illisible ou vide : {chemin.name}")
            continue
            
        # Détection des contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = []
        for c in contours:
            x, y_b, w, h = cv2.boundingRect(c)
            if w > 5 and h > 10:
                rectangles.append((x, y_b, w, h))
        
        # Fusion des boîtes englobantes proches
        rectangles = fusionner_boites(rectangles, tolerance=15)
        
        # Extraction et traitement des Régions d'Intérêt (ROI)
        for (x, y_b, w, h) in rectangles:
            roi = img[y_b:y_b+h, x:x+w]
            pad = 4
            roi_decore = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            img_redimensionnee = cv2.resize(roi_decore, taille_cible, interpolation=cv2.INTER_AREA)
            
            # Extraction des caractéristiques HOG
            features = hog(img_redimensionnee, orientations=9, pixels_per_cell=(4, 4),
                           cells_per_block=(2, 2), visualize=False)
            X.append(features)
            y.append(label)
            
    return np.array(X), np.array(y)

def main():
    DOSSIER_TRAIN = "resultats"
    os.makedirs("modeles", exist_ok=True)
    
    print("[→] Extraction des caractéristiques HOG...")
    X, y = preparer_donnees(DOSSIER_TRAIN)
    print(f"    ✔ {len(X)} échantillons de chiffres prêts.")
    
    # 1. Entraînement du SVM
    print("[→] Entraînement du modèle SVM...")
    modele_svm = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    modele_svm.fit(X, y)
    joblib.dump(modele_svm, "modeles/svm_chiffres.pkl")
    print("    ✔ Modèle SVM sauvegardé.")
    
    # 2. Entraînement du KNN
    print("[→] Entraînement du modèle KNN...")
    # n_neighbors=3 est une excellente valeur par défaut pour les chiffres isolés
    modele_knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
    modele_knn.fit(X, y)
    joblib.dump(modele_knn, "modeles/knn_chiffres.pkl")
    print("    ✔ Modèle KNN sauvegardé.")
    
    print("\n[✔] Tous les modèles sont prêts dans le dossier 'modeles/' !")

if __name__ == "__main__":
    main()