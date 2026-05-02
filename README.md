# SAFE DRIVE PRO
**Système de surveillance conducteur · Intelligence Artificielle**

`Version 2.0` · `Python 3.12` · `PyTorch` · `YOLOv8` · `MediaPipe`

---

## Description du projet

Safe Drive Pro est un système intelligent de surveillance conducteur en temps réel. Il analyse en continu le flux vidéo de la caméra pour détecter les comportements à risque et déclenche une **alerte vocale automatique** dès que le score de risque composite dépasse **50%**.

Le système détecte et analyse simultanément :

-  La **fatigue** du conducteur (modèle IA ResNet-18 entraîné)
-  Les **bâillements** (MediaPipe FaceMesh)
-  La **position anormale de la tête** (angles Yaw / Pitch / Roll)
-  L'**utilisation du téléphone** au volant (YOLOv8)

---

##  Modèle IA — `drowsiness_model.pth`

>  **Le fichier `drowsiness_model.pth` n'est pas inclus dans ce dépôt** en raison de sa taille volumineuse (~44 MB).
> Il doit être téléchargé séparément et placé dans le dossier `files/` du projet.

###  Télécharger le modèle

**[🔗 Télécharger drowsiness_model.pth — Google Drive](https://drive.google.com/file/d/194BfTVBInsMwTmdgKvJQXUK6N_Ec9gIM/view?usp=sharing)**

### Emplacement après téléchargement

```
projet/
└── files/
    └── drowsiness_model.pth   ← placer ici
```

### Caractéristiques du modèle

| Propriété | Valeur |
|-----------|--------|
| Architecture | ResNet-18 (torchvision) |
| Classes | 2 — Éveillé / Somnolent |
| Framework | PyTorch 2.x |
| Entrée | Image RGB 224×224 |
| Format | `.pth` (state_dict) |
| Taille | ~44 MB |

---

##  Formule du score composite

```
Score = ( 0.25 × Fatigue + 0.25 × Bâillement + 0.25 × Tête + 0.50 × Téléphone ) ÷ 1.25 × 100
```

| Indicateur | Poids | Calcul |
|------------|-------|--------|
|  Fatigue IA | 0.25 | Confidence du modèle ResNet-18 (0 → 1) |
|  Bâillement | 0.25 | Proportion dans une fenêtre glissante de 30s |
|  Position tête | 0.25 | Intensité angulaire normalisée (Yaw/Pitch/Roll) |
|  Téléphone | **0.50** | Confidence YOLO — double importance |

###  Comportement de l'alerte vocale

- L'alerte se déclenche à **chaque fois** que le score dépasse **50%**
- Cooldown configurable entre deux alertes (défaut : **6 secondes**)
- Messages variés pour éviter la monotonie (4 messages normaux + 3 messages danger)
- Score ≥ **75%** → message **DANGER CRITIQUE**

---

## ⚙️ Installation

### 1. Prérequis

| Outil | Version |
|-------|---------|
| Python | 3.10 ou supérieur (testé sur 3.12) |
| GPU | Optionnel — CUDA compatible pour accélération |
| Caméra | Webcam intégrée ou USB |
| OS | Windows 10/11 (testé), Linux, macOS |

### 2. Cloner le projet

```bash
git clone https://github.com/votre-username/safe-drive-pro.git
cd safe-drive-pro/files
```

### 3. Créer un environnement virtuel

```bash
python -m venv venv312

# Windows
venv312\Scripts\activate

# Linux / macOS
source venv312/bin/activate
```



### 4. Télécharger le modèle IA 

Télécharger depuis Google Drive :

**[🔗 Télécharger drowsiness_model.pth](https://drive.google.com/file/d/194BfTVBInsMwTmdgKvJQXUK6N_Ec9gIM/view?usp=sharing)**

Puis placer le fichier dans `files/drowsiness_model.pth`.

---

##  Lancement

### Interface Tkinter (bureau)

```bash
python main.py
```


##  Architecture du projet

```
files/
├── main.py                     → Interface Tkinter 
├── alert_engine.py             → Score composite + alertes vocales (pyttsx3)
├── detection.py                → MediaPipe FaceMesh — fatigue, bâillement, tête
├── video_capture.py            → Capture caméra threadée (ThreadedCapture)
├── yolo_detector.py            → Détection téléphone YOLOv8 (Ultralytics)
├── drowsiness_model_loader.py  → Chargeur du modèle ResNet-18
├── drowsiness_model.pth        → ⬇ À télécharger (Google Drive)
├── requirements.txt            → Dépendances Python
├── session_stats.json          → Statistiques sessions (généré automatiquement)
├── logs/                       → Logs CSV des sessions avec alertes

```

---

## Dépendances principales

| Package | Rôle |
|---------|------|
| `torch` / `torchvision` | Modèle de détection somnolence (ResNet-18) |
| `ultralytics` | YOLOv8 — détection téléphone en temps réel |
| `mediapipe` | FaceMesh — 478 landmarks faciaux |
| `opencv-python` | Traitement vidéo et affichage flux caméra |
| `pyttsx3` | Synthèse vocale hors ligne (alertes Tkinter) |
| `streamlit >= 1.32` | Interface web avec composants HTML natifs |
| `Pillow` | Conversion frames pour affichage Tkinter |
| `pandas` | Analyse des sessions CSV |

---

## Utilisation

### Interface Tkinter

1. Lancer `python main.py`
2. Le tableau de bord affiche les statistiques des sessions précédentes
3. Cliquer sur ** Démarrer** pour activer la caméra
4. Le score de risque s'affiche en temps réel avec la jauge circulaire
5. Cliquer sur ** Arrêter** pour terminer la session (stats sauvegardées automatiquement)


 
## 🔧 Configuration des paramètres

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| Seuil d'alerte (%) | 50 | Score déclenchant l'alerte vocale |
| Seuil danger (%) | 75 | Score critique — message d'urgence renforcé |
| Cooldown (s) | 6 | Délai minimum entre deux alertes vocales |
| Poids Fatigue | 0.25 | Contribution fatigue IA dans le score |
| Poids Bâillement | 0.25 | Contribution bâillements dans le score |
| Poids Tête | 0.25 | Contribution position tête dans le score |
| Poids Téléphone | 0.50 | Contribution téléphone dans le score |

---

##  Auteur

| | |
|--|--|
| **Projet** | Safe Drive Pro — Surveillance conducteur IA |
| **Cadre** | Vision par ordinateur — Projet universitaire |
| **Outils** | Python 3.12 · PyTorch · YOLOv8 · MediaPipe · Streamlit · Tkinter |
| **Contributeurs** | Amina FARIS, Fatima Iboubkarne, Salma JEGHLOUL, Salma CHLIH|

---

*Safe Drive Pro · 2026 · Tous droits réservés*
