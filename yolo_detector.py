"""
yolo_detector.py — Détection d'objets avec YOLOv8
Gère la détection de distractions (téléphone, etc.)
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional

# Import conditionnel YOLO
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("⚠️ YOLO non disponible. Installer: pip install ultralytics")


class YOLODetector:
    """
    Détecteur d'objets avec YOLOv8
    Spécialisé dans la détection de distractions au volant
    """
    
    # Classes d'intérêt pour la conduite
    DISTRACTION_CLASSES = {
        39: "bottle",           # Bouteille
        67: "cell phone",       # Téléphone portable
    }
    
    # Couleurs par type de distraction
    COLORS = {
        "cell phone": (0, 0, 255),      # Rouge
        "bottle": (255, 165, 0),        # Orange
        "default": (0, 165, 255)        # Orange par défaut
    }
    
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5):
        """
        Initialise le détecteur YOLO
        
        Args:
            model_path: Chemin vers le modèle YOLO
            conf_threshold: Seuil de confiance pour les détections
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Charge le modèle YOLO"""
        if not _YOLO_AVAILABLE:
            print("❌ YOLO non disponible")
            return
            
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ YOLOv8 chargé avec succès")
        except Exception as e:
            print(f"❌ Erreur chargement YOLO: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Vérifie si YOLO est disponible"""
        return self.model is not None and _YOLO_AVAILABLE
    
    def detect_distractions(self, frame: np.ndarray) -> Tuple[List[dict], np.ndarray]:
        """
        Détecte les distractions dans une frame
        
        Args:
            frame: Image BGR
            
        Returns:
            Tuple[List[dict], np.ndarray]: (détections, frame annotée)
        """
        distractions = []
        
        if not self.is_available():
            return distractions, frame
        
        try:
            # Inférence YOLO
            results = self.model(
                frame, 
                conf=self.conf_threshold, 
                imgsz=320,
                classes=list(self.DISTRACTION_CLASSES.keys()),
                verbose=False
            )
            
            # Traitement des résultats
            for result in results:
                if result.boxes is None:
                    continue
                    
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    label = self.DISTRACTION_CLASSES.get(cls_id, "unknown")
                    
                    # Coordonnées
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Informations de détection
                    distraction = {
                        'label': label,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'class_id': cls_id
                    }
                    distractions.append(distraction)
                    
                    # Dessiner sur la frame
                    frame = self._draw_distraction(frame, distraction)
                    
        except Exception as e:
            print(f"⚠️ Erreur détection YOLO: {e}")
            
        return distractions, frame
    
    def _draw_distraction(self, frame: np.ndarray, distraction: dict) -> np.ndarray:
        """
        Dessine les annotations de distraction sur la frame
        
        Args:
            frame: Image BGR
            distraction: Dictionnaire contenant les infos de détection
            
        Returns:
            np.ndarray: Frame annotée
        """
        label = distraction['label']
        confidence = distraction['confidence']
        x1, y1, x2, y2 = distraction['bbox']
        
        # Couleur selon le type de distraction
        color = self.COLORS.get(label, self.COLORS['default'])
        
        # Rectangle de détection
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Texte d'alerte
        text = f"⚠ {label.upper()} ({confidence:.0%})"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Fond du texte
        cv2.rectangle(
            frame, 
            (x1, y1 - text_size[1] - 10), 
            (x1 + text_size[0] + 10, y1 - 5), 
            color, 
            -1
        )
        
        # Texte
        cv2.putText(
            frame, text,
            (x1 + 5, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )
        
        return frame
    
    def get_distraction_count(self, distractions: List[dict]) -> int:
        """Retourne le nombre de distractions détectées"""
        return len(distractions)
    
    def has_cellphone(self, distractions: List[dict]) -> bool:
        """Vérifie si un téléphone portable a été détecté"""
        return any(d['label'] == 'cell phone' for d in distractions)
    
    def has_bottle(self, distractions: List[dict]) -> bool:
        """Vérifie si une bouteille a été détectée"""
        return any(d['label'] == 'bottle' for d in distractions)
    
    def get_distraction_labels(self, distractions: List[dict]) -> List[str]:
        """Retourne les labels des distractions détectées"""
        return list(set(d['label'] for d in distractions))
    
    def get_highest_confidence(self, distractions: List[dict]) -> Optional[dict]:
        """Retourne la distraction avec la plus haute confiance"""
        if not distractions:
            return None
        return max(distractions, key=lambda x: x['confidence'])
    


# Classe de test
class YOLODetectorTest:
    """Classe de test pour le détecteur YOLO"""
    
    @staticmethod
    def test_loading():
        """Test le chargement du modèle"""
        print("=== Test YOLODetector ===")
        detector = YOLODetector()
        
        if detector.is_available():
            print("✅ Détecteur YOLO initialisé avec succès")
        else:
            print("❌ Échec de l'initialisation YOLO")
        
        return detector
    
    @staticmethod
    def test_with_image(detector: YOLODetector, image_path: str):
        """Test la détection sur une image"""
        if not detector.is_available():
            print("YOLO non disponible")
            return
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Impossible de lire l'image: {image_path}")
            return
        
        distractions, annotated = detector.detect_distractions(img)
        
        print(f"Détections: {len(distractions)}")
        for d in distractions:
            print(f"  - {d['label']}: {d['confidence']:.2%}")
        
        cv2.imshow("YOLO Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Test du détecteur
    detector = YOLODetectorTest.test_loading()
    
    # Optionnel: tester avec une image
    # detector = YOLODetector()
    # YOLODetectorTest.test_with_image(detector, "test_image.jpg")