"""
detection.py — Tous les algorithmes de détection
MediaPipe FaceMesh, MAR, pose de tête, inférence IA

Ce fichier contient tous les algorithmes de vision par ordinateur pour :
- Détecter le visage et les points clés (landmarks)
- Calculer l'ouverture de la bouche (MAR - Mouth Aspect Ratio)
- Analyser la pose de la tête (yaw, pitch, roll)
- Détecter la fatigue via un modèle IA (ResNet)
- Compter les bâillements
"""

import cv2
import math
import numpy as np
import mediapipe as mp
import torch
import torch.nn.functional as F
from scipy.spatial import distance as dist
from torchvision import transforms
from typing import Tuple, Optional

from alert_engine import DetectionState


class FaceDetector:
    """
    Détecteur facial complet utilisant MediaPipe FaceMesh + IA de fatigue.
    
    Cette classe est le module central de détection qui analyse chaque frame
    de la caméra pour extraire toutes les informations nécessaires :
    - Position et orientation du visage
    - Ouverture de la bouche (pour détecter les bâillements)
    - Niveau de fatigue via réseau de neurones
    - Pose de la tête (angles)
    """

    # ──────────────────────────────────────────────────────────────────
    # INDICES DES LANDMARKS (points clés du visage)
    # ──────────────────────────────────────────────────────────────────
    # Ces indices correspondent aux points spécifiques du modèle MediaPipe
    # _LEFT_EYE  : Points autour de l'œil gauche
    # _RIGHT_EYE : Points autour de l'œil droit  
    # _MOUTH     : Points autour de la bouche (pour calculer l'ouverture)
    # ──────────────────────────────────────────────────────────────────
    
    _LEFT_EYE  = [362, 385, 387, 263, 373, 380]   # Points de l'œil gauche
    _RIGHT_EYE = [33, 160, 158, 133, 153, 144]    # Points de l'œil droit
    _MOUTH     = [13, 14, 78, 308]                # Points de la bouche

    # ──────────────────────────────────────────────────────────────────
    # SEUILS PAR DÉFAUT (peuvent être modifiés par calibration)
    # ──────────────────────────────────────────────────────────────────
    # MAR_THRESHOLD : Seuil d'ouverture de bouche pour détecter un bâillement
    # YAW_RANGE     : Plage normale de rotation horizontale de la tête (-30° à +30°)
    # PITCH_RANGE   : Plage normale d'inclinaison verticale (-25° à +25°)
    # ROLL_RANGE    : Plage normale d'inclinaison latérale (-20° à +20°)
    # ──────────────────────────────────────────────────────────────────
    
    DEFAULT_MAR_THRESHOLD = 0.60      # 0.60 = 60% d'ouverture = bâillement
    DEFAULT_YAW_RANGE     = (-30, 30) # Rotation tête gauche/droite
    DEFAULT_PITCH_RANGE   = (-25, 25) # Inclinaison tête haut/bas
    DEFAULT_ROLL_RANGE    = (-20, 20) # Inclinaison oreille/oreille

    def __init__(
        self,
        drowsiness_model=None,      # Modèle IA de détection de fatigue
        device=None,                # CPU 
        model_loaded: bool = False, # Le modèle a-t-il été chargé ?
        drowsy_threshold: float = 0.70,  # Seuil pour considérer fatigué (70%)
        mar_threshold: float = DEFAULT_MAR_THRESHOLD,
        yaw_range: Tuple = DEFAULT_YAW_RANGE,
        pitch_range: Tuple = DEFAULT_PITCH_RANGE,
        roll_range: Tuple = DEFAULT_ROLL_RANGE,
    ):
        """
        Initialise le détecteur facial.
        
        Args:
            drowsiness_model: Modèle PyTorch entraîné pour la détection de fatigue
            device: CPU
            model_loaded: True si le modèle a été chargé avec succès
            drowsy_threshold: Seuil de confiance pour alerte fatigue (0.7 = 70%)
            mar_threshold: Seuil d'ouverture de bouche pour bâillement
            yaw_range: Plage normale de rotation horizontale (min, max)
            pitch_range: Plage normale d'inclinaison verticale (min, max)
            roll_range: Plage normale d'inclinaison latérale (min, max)
        """
        # Configuration du modèle IA
        self.drowsiness_model = drowsiness_model
        self.device = device or torch.device("cpu")
        self.model_loaded = model_loaded
        self.drowsy_threshold = drowsy_threshold
        
        # Seuils de détection
        self.mar_threshold = mar_threshold
        self.yaw_range = yaw_range
        self.pitch_range = pitch_range
        self.roll_range = roll_range

        # ──────────────────────────────────────────────────────────────
        # INITIALISATION DE MEDIAPIPE FACEMESH
        # ──────────────────────────────────────────────────────────────
        # MediaPipe FaceMesh est un modèle qui détecte 468 points (landmarks)
        # sur le visage en temps réel avec une grande précision.
        # refine_landmarks=True donne des points plus précis autour des yeux
        # ──────────────────────────────────────────────────────────────
        _mp = mp.solutions.face_mesh
        self.face_mesh = _mp.FaceMesh(
            max_num_faces=1,              # Détecte un seul visage (le conducteur)
            refine_landmarks=True,        # Points plus précis autour des yeux
            min_detection_confidence=0.5, # Confiance minimum pour détection
            min_tracking_confidence=0.5,  # Confiance minimum pour tracking
        )

        # ──────────────────────────────────────────────────────────────
        # PRÉTRAITEMENT POUR LE MODÈLE IA
        # ──────────────────────────────────────────────────────────────
        # Le modèle IA (ResNet) attend des images redimensionnées à 224x224
        # et normalisées avec les moyennes et écarts-types d'ImageNet
        # ──────────────────────────────────────────────────────────────
        self.transform = transforms.Compose([
            transforms.ToPILImage(),                     # Convertit numpy → PIL
            transforms.Resize((224, 224)),               # Redimensionne à 224x224
            transforms.ToTensor(),                       # Convertit PIL → Tensor
            transforms.Normalize(                        # Normalisation ImageNet
                mean=[0.485, 0.456, 0.406],              # Moyennes RGB
                std=[0.229, 0.224, 0.225]                # Écarts-types RGB
            ),
        ])

        # ──────────────────────────────────────────────────────────────
        # LISSAGE DES PRÉDICTIONS (ANTI-OSCILLATION)
        # ──────────────────────────────────────────────────────────────
        # Buffer circulaire qui stocke les 5 dernières prédictions de fatigue
        # Permet d'éviter les changements brusques (faux positifs)
        # ──────────────────────────────────────────────────────────────
        self._confidence_buffer = []   # Stocke les confiances récentes
        self._buffer_size = 5          # Taille du buffer (5 frames)

        # ──────────────────────────────────────────────────────────────
        # DEBOUNCE POUR BÂILLEMENTS
        # ──────────────────────────────────────────────────────────────
        # Empêche la détection multiple d'un même bâillement.
        # Sans debounce, un bâillement serait détecté sur ~10 frames consécutives.
        # ──────────────────────────────────────────────────────────────
        self._yawn_debounce = False

    # ──────────────────────────────────────────────────────────────────
    # MÉTHODE PUBLIQUE : MISE À JOUR DES SEUILS
    # ──────────────────────────────────────────────────────────────────
    def update_thresholds(self, **kwargs):
        """
        Met à jour dynamiquement les seuils de détection.
        Utilisé par la fenêtre de calibration pour ajuster la sensibilité.
        
        Args:
            **kwargs: Paramètres à modifier (mar_threshold, yaw_range, etc.)
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    # ──────────────────────────────────────────────────────────────────
    # POINT D'ENTRÉE PRINCIPAL
    # ──────────────────────────────────────────────────────────────────
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, DetectionState]:
        """
        Traite une frame de la caméra et retourne l'état complet de détection.
        
        C'est la méthode principale appelée à chaque frame par main.py.
        
        Args:
            frame: Image BGR de la caméra (format OpenCV)
            
        Returns:
            Tuple (frame_annotated, DetectionState)
            - frame_annotated: Frame avec overlay graphique (texte, cercles, barres)
            - DetectionState: Objet contenant tous les résultats de détection
        """
        h, w = frame.shape[:2]  # Hauteur et largeur de l'image
        state = DetectionState()  # État initial vide

        # ──────────────────────────────────────────────────────────────
        # 1. CONVERSION RGB + DÉTECTION FACEMESH
        # ──────────────────────────────────────────────────────────────
        # MediaPipe travaille en RGB (OpenCV utilise BGR par défaut)
        # ──────────────────────────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        # ──────────────────────────────────────────────────────────────
        # 2. SI AUCUN VISAGE DÉTECTÉ
        # ──────────────────────────────────────────────────────────────
        if not results.multi_face_landmarks:
            # Nettoie les buffers
            self._confidence_buffer.clear()
            self._yawn_debounce = False
            
            # Affiche un message d'alerte sur la frame
            cv2.putText(
                frame, "AUCUN VISAGE DETECTE",
                (50, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 165, 255), 2
            )
            return frame, state  # Retourne sans détection

        # ──────────────────────────────────────────────────────────────
        # 3. VISAGE DÉTECTÉ
        # ──────────────────────────────────────────────────────────────
        state.face_detected = True

        # Parcourt les visages détectés (normalement 1 seul)
        for lm_set in results.multi_face_landmarks:
            # Convertit les landmarks normalisés (0-1) en pixels (0-w, 0-h)
            coords = np.array(
                [(int(l.x * w), int(l.y * h)) for l in lm_set.landmark]
            )

            # ══════════════════════════════════════════════════════════
            # 3a. DÉTECTION DES BÂILLEMENTS (MAR - Mouth Aspect Ratio)
            # ══════════════════════════════════════════════════════════
            # MAR = (distance verticale entre lèvres) / (distance horizontale)
            # Plus la bouche est ouverte, plus le MAR est élevé
            # ══════════════════════════════════════════════════════════
            mar = self._mar(coords)
            
            if mar > self.mar_threshold:
                # Si seuil dépassé et pas déjà en debounce
                if not self._yawn_debounce:
                    state.yawn_detected = True      # Bâillement détecté !
                    self._yawn_debounce = True      # Active debounce
                cv2.putText(frame, "BAILLEMENT", (50, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                self._yawn_debounce = False         # Réinitialise debounce

            # ══════════════════════════════════════════════════════════
            # 3b. DÉTECTION DE FATIGUE PAR IA
            # ══════════════════════════════════════════════════════════
            # Extrait la région du visage (ROI) pour l'analyser
            # ══════════════════════════════════════════════════════════
            face_roi = self._extract_roi(frame, coords, w, h)
            drowsy, conf = self._detect_drowsiness(face_roi)
            
            # Met à jour l'état avec les résultats IA
            state.is_drowsy_ai = drowsy
            state.drowsy_confidence = conf

            # Affiche la barre de fatigue
            self._draw_fatigue_bar(frame, conf)

            # Si fatigue détectée, affiche alerte
            if drowsy:
                cv2.putText(frame, "!!! ALERTE FATIGUE  !!!", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                cv2.putText(frame, f"Confiance: {conf:.0%}", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # ══════════════════════════════════════════════════════════
            # 3c. ANALYSE DE LA POSE DE TÊTE (angles Yaw, Pitch, Roll)
            # ══════════════════════════════════════════════════════════
            # Yaw   : Rotation gauche/droite (secouer la tête "non")
            # Pitch : Inclinaison haut/bas (hocher la tête "oui")
            # Roll  : Inclinaison oreille/oreille (pencher la tête)
            # ══════════════════════════════════════════════════════════
            yaw, pitch, roll = self._head_pose(lm_set.landmark, w, h)
            state.yaw, state.pitch, state.roll = yaw, pitch, roll
            
            # Vérifie si la tête est dans une position normale
            normal, status = self._is_normal(yaw, pitch, roll)
            state.head_abnormal = not normal  # True = position anormale
            state.head_status = status

            # Affiche l'alerte si tête anormale
            if not normal:
                cv2.putText(frame, f"TETE: {status}", (50, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Affichage des angles
            cv2.putText(frame, f"Yaw:{int(yaw)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Pitch:{int(pitch)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Roll:{int(roll)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Indicateur visuel de la direction du regard
            self._draw_head_indicator(frame, yaw, pitch, w, h, normal)

        return frame, state

    # ══════════════════════════════════════════════════════════════════
    # MÉTHODES PRIVÉES (HELPERS)
    # ══════════════════════════════════════════════════════════════════

    def _mar(self, coords) -> float:
        """
        Calcule le Mouth Aspect Ratio (MAR).
        
        MAR = distance verticale entre les lèvres / distance horizontale
        
        Plus la bouche est ouverte, plus le MAR est élevé.
        Un MAR > 0.6 indique généralement un bâillement.
        
        Args:
            coords: Coordonnées des landmarks du visage
            
        Returns:
            float: Rapport d'aspect de la bouche (0 = bouche fermée, >0.6 = bâillement)
        """
        M = self._MOUTH
        # distance verticale (lèvre supérieure - lèvre inférieure)
        vertical = dist.euclidean(coords[M[0]], coords[M[1]])
        # distance horizontale (commissures des lèvres)
        horizontal = dist.euclidean(coords[M[2]], coords[M[3]])
        return vertical / (horizontal + 1e-6)  # +1e-6 évite division par zéro

    def _extract_roi(self, frame, coords, w, h):
        """
        Extrait la région du visage (ROI - Region of Interest).
        
        Prend une marge de 30 pixels autour du visage pour avoir un contexte.
        
        Args:
            frame: Image originale
            coords: Coordonnées des landmarks
            w, h: Largeur et hauteur de l'image
            
        Returns:
            numpy.ndarray: Région du visage ou None si invalide
        """
        xs, ys = [p[0] for p in coords], [p[1] for p in coords]
        x1 = max(0, min(xs) - 30)   # Bord gauche (avec sécurité)
        x2 = min(w, max(xs) + 30)   # Bord droit
        y1 = max(0, min(ys) - 30)   # Bord haut
        y2 = min(h, max(ys) + 30)   # Bord bas
        roi = frame[y1:y2, x1:x2]
        return roi if roi.size > 0 else None

    def _detect_drowsiness(self, face_roi) -> Tuple[bool, float]:
        """
        Détecte la somnolence via le modèle IA (ResNet).
        
        Le modèle analyse l'image du visage et retourne la probabilité que
        le conducteur soit fatigué.
        
        Utilise un buffer de lissage pour éviter les faux positifs.
        
        Args:
            face_roi: Image du visage extraite (ROI)
            
        Returns:
            Tuple (is_drowsy, confidence)
            - is_drowsy: True si le conducteur est fatigué
            - confidence: Niveau de confiance (0-1)
        """
        # Si modèle non chargé ou ROI invalide
        if not self.model_loaded or face_roi is None:
            return False, 0.0
        
        try:
            # Prétraitement de l'image pour le modèle
            tensor = self.transform(face_roi).unsqueeze(0).to(self.device)
            
            # Inférence
            with torch.no_grad():
                out = self.drowsiness_model(tensor)
                probs = F.softmax(out, dim=1)
            
            # La sortie est [prob_éveillé, prob_fatigué]
            # Nous prenons la probabilité d'être fatigué
            prob = probs[0][1].item()  # [1] = classe fatigué

            # Lissage : moyenne des dernières prédictions
            self._confidence_buffer.append(prob)
            if len(self._confidence_buffer) > self._buffer_size:
                self._confidence_buffer.pop(0)

            avg = float(np.mean(self._confidence_buffer))
            return avg > self.drowsy_threshold, avg

        except Exception as e:
            print(f"[_detect_drowsiness] erreur: {e}")
            return False, 0.0

    @staticmethod
    def _draw_fatigue_bar(frame, conf):
        """
        Dessine une barre de progression pour la fatigue.
        
        Args:
            frame: Image sur laquelle dessiner
            conf: Niveau de fatigue (0-1)
        """
        # Couleur selon le niveau : vert (<50%), orange (50-70%), rouge (>70%)
        if conf < 0.5:
            color = (0, 255, 0)      # Vert = peu fatigué
        elif conf < 0.7:
            color = (0, 165, 255)    # Orange = fatigue modérée
        else:
            color = (0, 0, 255)      # Rouge = fatigue élevée
        
        cv2.putText(frame, f"FATIGUE: {conf:.0%}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Barre de progression (200 pixels max)
        fill = int(200 * conf)
        cv2.rectangle(frame, (10, 135), (210, 150), (50, 50, 50), -1)  # Fond
        cv2.rectangle(frame, (10, 135), (10 + fill, 150), color, -1)    # Remplissage

    def _head_pose(self, landmarks, w, h) -> Tuple[float, float, float]:
        """
        Calcule les angles de la tête (yaw, pitch, roll) via solvePnP.
        
        Utilise la correspondance entre points 2D (image) et points 3D (modèle)
        pour estimer la rotation de la tête.
        
        Args:
            landmarks: Landmarks MediaPipe (format normalisé)
            w, h: Dimensions de l'image
            
        Returns:
            Tuple (yaw, pitch, roll): Angles en degrés
        """
        # Points 3D du modèle de tête (en mm, approximatif)
        points_3d = np.array([
            [0, 0, 0],           # Nez
            [0, -330, -65],      # Menton
            [-225, 170, -135],   # Œil gauche
            [225, 170, -135],    # Œil droit
            [-150, -150, -125],  # Bouche gauche
            [150, -150, -125]    # Bouche droite
        ], dtype=np.float64)
        
        # Indices correspondants dans les landmarks MediaPipe
        idx = [1, 152, 33, 263, 61, 291]
        
        # Points 2D correspondants dans l'image
        points_2d = np.array(
            [(landmarks[i].x * w, landmarks[i].y * h) for i in idx],
            dtype=np.float64
        )
        
        # Matrice de caméra
        focal = w  # Distance focale approximative
        cam = np.array([
            [focal, 0, w/2],
            [0, focal, h/2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Résolution PnP (Perspective-n-Point)
        ok, rvec, _ = cv2.solvePnP(
            points_3d, points_2d, cam, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not ok:
            return 0, 0, 0
        
        # Convertit vecteur rotation en matrice
        rmat, _ = cv2.Rodrigues(rvec)
        
        # Calcule les angles (formules standard)
        sy = math.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
        pitch = math.atan2(-rmat[2, 0], sy) * 180 / math.pi
        yaw   = math.atan2(rmat[1, 0], rmat[0, 0]) * 180 / math.pi
        roll  = math.atan2(rmat[2, 1], rmat[2, 2]) * 180 / math.pi
        
        # Normalisation des angles entre -180° et 180°
        yaw   = (yaw + 180) % 360 - 180
        pitch = (pitch + 180) % 360 - 180
        roll  = (roll + 180) % 360 - 180
        
        # Correction pour les angles extrêmes
        if abs(roll) > 160:
            roll = 0
        
        return yaw, pitch, roll

    def _is_normal(self, yaw, pitch, roll) -> Tuple[bool, str]:
        """
        Vérifie si la tête est dans une position normale.
        
        Args:
            yaw, pitch, roll: Angles de la tête
            
        Returns:
            Tuple (normal, status)
            - normal: True si position normale, False si anormale
            - status: Description textuelle de l'anomalie
        """
        # Vérification de la rotation horizontale (gauche/droite)
        if not (self.yaw_range[0] <= yaw <= self.yaw_range[1]):
            return False, f"Tournee ({int(yaw)}deg)"
        
        # Vérification de l'inclinaison verticale (haut/bas)
        if not (self.pitch_range[0] <= pitch <= self.pitch_range[1]):
            direction = "baissee" if pitch > 0 else "levee"
            return False, f"Tete {direction} ({int(abs(pitch))}deg)"
        
        # Vérification de l'inclinaison latérale (penchée)
        if not (self.roll_range[0] <= roll <= self.roll_range[1]):
            direction = "droite" if roll > 0 else "gauche"
            return False, f"Inclinee {direction} ({int(abs(roll))}deg)"
        
        return True, "Normale"

    @staticmethod
    def _draw_head_indicator(frame, yaw, pitch, w, h, normal):
        """
        Dessine un indicateur visuel de la direction du regard.
        
        Un cercle central et un point qui se déplace selon l'orientation.
        
        Args:
            frame: Image sur laquelle dessiner
            yaw, pitch: Angles de la tête
            w, h: Dimensions de l'image
            normal: Position normale ou non
        """
        cx, cy = w // 2, h // 2  # Centre de l'image
        
        # Couleur : vert si normal, rouge si anormal
        color = (0, 255, 0) if normal else (0, 0, 255)
        
        # Déplacement proportionnel aux angles (max 150 pixels)
        ox = int(max(-150, min(150, (yaw / 30) * 100)))
        oy = int(max(-150, min(150, (pitch / 30) * 100)))
        
        # Dessine le centre et le point de direction
        cv2.circle(frame, (cx, cy), 15, (100, 100, 100), 1)  # Cercle central
        cv2.circle(frame, (cx + ox, cy + oy), 8, color, -1)   # Point direction
        cv2.line(frame, (cx, cy), (cx + ox, cy + oy), color, 2)  # Ligne