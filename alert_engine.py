"""
alert_engine.py — Moteur d'alertes pour Safe Drive Pro
Gère le score composite, les alertes vocales et les statistiques
Version robuste avec gestion des erreurs TTS
"""

import threading
import time
import csv
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

# Tentative d'importation de pyttsx3 pour la synthèse vocale
try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False
    print(" pyttsx3 non disponible ")

# Tentative d'importation de winsound pour les bips sonores (Windows)
try:
    import winsound
    _WINSOUND_AVAILABLE = True
except ImportError:
    _WINSOUND_AVAILABLE = False


@dataclass
class DetectionState:
    """
    Classe de données qui contient l'état complet des détections à un instant T.
    Utilisée pour transmettre toutes les informations du module de détection au moteur d'alertes.
    """
    drowsy_confidence: float = 0.0      # Niveau de fatigue (0-1)
    yawn_detected: bool = False         # Bâillement détecté ou non
    phone_detected: bool = False        # Téléphone détecté ou non
    phone_confidence: float = 0.0       # Confiance de détection du téléphone
    face_detected: bool = False         # Visage détecté ou non
    head_status: str = "Normal"         # Statut de la tête (Normal, Inclinée, etc.)
    head_abnormal: bool = False         # Position anormale de la tête
    yaw: float = 0.0                    # Rotation horizontale de la tête
    pitch: float = 0.0                  # Rotation verticale de la tête
    roll: float = 0.0                   # Inclinaison latérale de la tête
    is_drowsy_ai: bool = False          # Fatigue détectée par l'IA


@dataclass
class AlertConfig:
    """
    Configuration des seuils et poids pour le calcul du score.
    Permet de personnaliser la sensibilité du système d'alerte.
    """
    alert_threshold: int = 50           # Seuil d'alerte (déclenchement)
    danger_score: int = 75              # Seuil de danger critique
    voice_cooldown: float = 3.0         # Délai entre deux alertes vocales (secondes)
    weight_fatigue: float = 0.25        # Poids de la fatigue dans le score
    weight_yawn: float = 0.25           # Poids des bâillements dans le score
    weight_head: float = 0.25           # Poids de la position de tête
    weight_phone: float = 0.50          # Poids du téléphone (double par défaut)


class AlertEngine:
    """
    Moteur principal d'alertes.
    Calcule un score composite (0-100) et déclenche des alertes vocales
    lorsque le score dépasse le seuil configuré.
    
    Score = (0.25×Fatigue + 0.25×Bâillement + 0.25×Tête + 0.50×Téléphone) ÷ 1.25 × 100
    """

    # Messages d'alerte normaux (rotation cyclique)
    _ALERT_MESSAGES = [
        "Attention conducteur ! Votre niveau de risque est élevé.",
        "Avertissement ! Signes de danger détectés, soyez vigilant.",
        "Alerte sécurité ! Le conducteur est en situation à risque.",
        "Avertissement détectée, soyez vigilant.",
        "Score de risque critique, restez concentré sur la route.",
    ]

    # Messages de danger critique (rotation cyclique)
    _DANGER_MESSAGES = [
        "DANGER CRITIQUE ! Arrêtez le véhicule immédiatement !",
        "ALERTE MAXIMALE ! stoppez d'urgence !",
        "DANGER EXTRÊME ! Le conducteur ne contrôle plus la situation !",
        "URGENCE ! Risque d'accident imminent, arrêtez-vous !",
    ]

    def __init__(self, config=None, log_dir="logs"):
        """
        Initialise le moteur d'alertes.
        
        Args:
            config: Configuration personnalisée (AlertConfig)
            log_dir: Dossier pour les fichiers de logs CSV
        """
        # Configuration
        self.config = config or AlertConfig()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Statistiques de session
        self.yawn_count = 0          # Compteur de bâillements
        self.phone_alerts = 0        # Compteur d'alertes déclenchées
        self.global_score = 0        # Score actuel (0-100)
        
        # Index pour la rotation des messages
        self._alert_msg_idx = 0       # Index pour messages normaux
        self._danger_msg_idx = 0      # Index pour messages danger
        
        # Historique des bâillements (fenêtre glissante)
        self._yawn_timestamps = []     # Timestamps des bâillements récents
        self._yawn_window = 30.0       # Fenêtre de 30 secondes

        # Variables pour la gestion des alertes
        self._last_voice_time = 0.0    # Dernier moment où une alerte a été jouée
        self._last_alert_score = 0     # Dernier score ayant déclenché une alerte
        self._speaking = False         # Flag indiquant si une alerte est en cours
        self._speech_lock = threading.Lock()  # Verrou pour éviter les conflits vocaux
        self._lock = threading.Lock()          # Verrou général pour thread-safety
        
        # État de l'alerte active
        self._alert_active = False     # Alerte active en cours
        self._alert_start_time = 0.0   # Moment du début de l'alerte

        # Détail du score (pour affichage)
        self.score_detail = {"fatigue": 0.0, "yawn": 0.0, "head": 0.0, "phone": 0.0}

        # Initialisation du fichier CSV pour les logs
        self._csv_path = os.path.join(
            log_dir, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        self._init_csv()

        # Moteur TTS (initialisation différée)
        self._tts_engine = None
        self._tts_lock = threading.Lock()
        
        print(" Moteur d'alertes initialisé - Mode alerte multiple actif")

    def _init_csv(self):
        """
        Initialise le fichier CSV avec les en-têtes de colonnes.
        Crée un nouveau fichier à chaque session.
        """
        with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp", "global_score",
                "p_fatigue", "p_yawn", "p_head", "p_phone",
                "phone_conf", "phone_detected",
                "drowsy_conf", "yawn", "head_status", "event"
            ])

    def _log(self, state: DetectionState, event: str = ""):
        """
        Enregistre l'état actuel dans le fichier CSV.
        
        Args:
            state: État de détection actuel
            event: Événement spécial (ex: "VOICE_ALERT")
        """
        try:
            d = self.score_detail
            with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    datetime.now().isoformat(), self.global_score,
                    f"{d['fatigue']:.3f}", f"{d['yawn']:.3f}",
                    f"{d['head']:.3f}", f"{d['phone']:.3f}",
                    f"{state.phone_confidence:.3f}", int(state.phone_detected),
                    f"{state.drowsy_confidence:.3f}", int(state.yawn_detected),
                    state.head_status, event,
                ])
        except Exception as e:
            print(f"⚠️ Erreur log: {e}")

    def compute_score(self, state: DetectionState) -> int:
        """
        Calcule le score composite normalisé sur 100.
        
        Formule: 
            raw = (w_fatigue * p_fatigue + w_yawn * p_yawn + 
                   w_head * p_head + w_phone * p_phone)
            normalized = raw / total_weight
            score = min(100, int(normalized * 100))
        
        Args:
            state: État de détection actuel
            
        Returns:
            Score entre 0 et 100
        """
        cfg = self.config
        now = time.time()

        # 1. Probabilité de fatigue (directe de l'IA)
        p_fatigue = min(1.0, state.drowsy_confidence)

        # 2. Probabilité de bâillement (basée sur la fréquence dans les 30 dernières secondes)
        #    Nettoie les timestamps trop anciens
        self._yawn_timestamps = [t for t in self._yawn_timestamps if now - t < self._yawn_window]
        if state.yawn_detected:
            self._yawn_timestamps.append(now)
        # Maximum 3 bâillements = 100%
        p_yawn = min(1.0, len(self._yawn_timestamps) / 3.0)

        # 3. Probabilité de tête anormale (basée sur les angles)
        if state.head_abnormal:
            max_angle = max(
                abs(state.yaw) / 30.0,    # Yaw max 30° = 100%
                abs(state.pitch) / 25.0,  # Pitch max 25° = 100%
                abs(state.roll) / 20.0    # Roll max 20° = 100%
            )
            p_head = min(1.0, max_angle)
        else:
            p_head = 0.0

        # 4. Probabilité téléphone (confiance YOLO)
        p_phone = state.phone_confidence if state.phone_detected else 0.0

        # Calcul pondéré
        raw = (cfg.weight_fatigue * p_fatigue
               + cfg.weight_yawn   * p_yawn
               + cfg.weight_head   * p_head
               + cfg.weight_phone  * p_phone)

        # Normalisation par la somme des poids
        total_weight = cfg.weight_fatigue + cfg.weight_yawn + cfg.weight_head + cfg.weight_phone
        normalized = raw / total_weight if total_weight > 0 else 0.0

        # Stockage des détails pour affichage
        self.score_detail = {
            "fatigue": p_fatigue,
            "yawn":    p_yawn,
            "head":    p_head,
            "phone":   p_phone,
        }

        return min(100, int(normalized * 100))

    def _speak_robust(self, text: str):
        """
        Version robuste de la synthèse vocale.
        Crée un nouveau moteur TTS à chaque appel pour éviter les bugs.
        Utilise un fallback avec winsound en cas d'échec.
        
        Args:
            text: Message textuel à prononcer
        """
        # Si pyttsx3 n'est pas disponible, utiliser le fallback
        if not _TTS_AVAILABLE:
            self._speak_fallback(text)
            return
        
        engine = None
        try:
            # CRÉER UN NOUVEAU MOTEUR À CHAQUE ALERTE 
            engine = pyttsx3.init()
            
            # Configuration de la voix
            engine.setProperty('rate', 155)   # Vitesse de parole (normal)
            engine.setProperty('volume', 1.0) # Volume maximum
            
            # Sélectionner une voix française 
            voices = engine.getProperty('voices')
            french_voice_found = False
            
            for voice in voices:
                voice_name = voice.name.lower()
                voice_id = voice.id.lower()
                if 'french' in voice_name or 'fr' in voice_id:
                    engine.setProperty('voice', voice.id)
                    french_voice_found = True
                    print(f"Voix française sélectionnée: {voice.name}")
                    break
            
            if not french_voice_found:
                print("Voix française non trouvée, utilisation voix par défaut")
            
            # Prononcer le message
            print(f" Alerte vocale: {text}")
            engine.say(text)
            engine.runAndWait()
            
            # Petite pause pour permettre la réinitialisation
            time.sleep(0.3)
            
        except Exception as e:
            print(f" Erreur synthèse vocale: {e}")
            # Fallback en cas d'erreur
            self._speak_fallback(text)
            
        finally:
            # Nettoyage systématique du moteur
            if engine:
                try:
                    engine.stop()
                except:
                    pass

    def _speak_fallback(self, text: str):
        """
        Méthode de fallback quand la synthèse vocale échoue.
        Utilise des bips sonores et affiche le message dans la console.
        
        Args:
            text: Message textuel à afficher
        """
        print(f"\n{'='*60}")
        print(f" ALERTE: {text}")
        print(f"{'='*60}\n")
        
        # Bips sonores selon le niveau de danger
        if _WINSOUND_AVAILABLE:
            try:
                if self.global_score >= 75:
                    # Danger critique : 3 bips longs
                    for _ in range(3):
                        winsound.Beep(1000, 400)
                        time.sleep(0.2)
                elif self.global_score >= 60:
                    # Alerte : 2 bips
                    for _ in range(2):
                        winsound.Beep(800, 300)
                        time.sleep(0.15)
                else:
                    # Attention : 1 bip
                    winsound.Beep(600, 200)
            except:
                pass

    def _speak(self, text: str):
        """
        Thread wrapper pour la synthèse vocale.
        Utilise un verrou pour éviter les conflits entre alertes simultanées.
        
        Args:
            text: Message textuel à prononcer
        """
        def speak_thread():
            with self._speech_lock:  # Verrou pour éviter les conflits
                try:
                    self._speaking = True
                    self._speak_robust(text)
                except Exception as e:
                    print(f" Erreur dans le thread vocal: {e}")
                finally:
                    self._speaking = False
        
        # Démarrer dans un thread séparé pour ne pas bloquer l'interface
        thread = threading.Thread(target=speak_thread, daemon=True)
        thread.start()
        
        # Petite pause pour éviter l'accumulation de threads
        time.sleep(0.1)

    def update(self, state: DetectionState):
        """
        Met à jour le score et déclenche les alertes.
        C'est la fonction principale appelée à chaque frame.
        
        Args:
            state: État de détection actuel
            
        Returns:
            Tuple (score, events) où events est une liste des événements survenus
        """
        with self._lock:
            events = []
            now = time.time()

            # Si aucun visage n'est détecté, score à 0 et pas d'alerte
            if not state.face_detected:
                self.global_score = 0
                self._alert_active = False
                return 0, []

            # Calcul du nouveau score
            new_score = self.compute_score(state)
            self.global_score = new_score

            # Comptage des bâillements
            if state.yawn_detected:
                self.yawn_count += 1
                events.append("yawn")

            # Vérification du dépassement de seuil
            is_alert_triggered = (self.global_score > self.config.alert_threshold)
            
            # Vérifier si on peut déclencher une nouvelle alerte
            cooldown_ok = now - self._last_voice_time > self.config.voice_cooldown
            
            # Déclencher l'alerte si toutes les conditions sont remplies
            if is_alert_triggered and cooldown_ok and not self._speaking:
                
                # Marquer l'alerte
                self._last_voice_time = now
                self._alert_active = True
                self._alert_start_time = now
                self.phone_alerts += 1
                
                # Choisir le message selon le niveau de danger
                if self.global_score >= self.config.danger_score:
                    msg = self._DANGER_MESSAGES[self._danger_msg_idx % len(self._DANGER_MESSAGES)]
                    self._danger_msg_idx += 1  # Rotation du message
                else:
                    msg = self._ALERT_MESSAGES[self._alert_msg_idx % len(self._ALERT_MESSAGES)]
                    self._alert_msg_idx += 1   # Rotation du message
                
                # Journaliser l'alerte dans le CSV
                self._log(state, f"VOICE_ALERT_{self.global_score}")
                
                # Déclencher l'alerte vocale
                self._speak(msg)
                events.append("voice_alert")
                
                # Affichage console pour débogage
                print(f"\n ALERTE DÉCLENCHÉE - Score: {self.global_score}%")
                print(f"Message: {msg}")
                print(f"Prochaine alerte possible dans {self.config.voice_cooldown} secondes\n")
                
            elif not is_alert_triggered:
                # Si le score redevient normal, désactiver l'alerte
                if self._alert_active:
                    self._alert_active = False
                    events.append("alert_ended")
                    print(" Alerte terminée - Score revenu à la normale")
            
            # Journalisation normale (pas d'alerte)
            if not events or "voice_alert" not in events:
                self._log(state)

            return self.global_score, events

    def get_stats(self) -> dict:
        """
        Retourne les statistiques actuelles de la session.
        
        Returns:
            Dictionnaire contenant toutes les statistiques
        """
        with self._lock:
            return {
                "yawn_count":   self.yawn_count,           # Nombre total de bâillements
                "phone_alerts": self.phone_alerts,         # Nombre d'alertes déclenchées
                "global_score": self.global_score,         # Score actuel
                "score_detail": dict(self.score_detail),   # Détail des sous-scores
                "log_path":     self._csv_path,            # Chemin du fichier CSV
                "alert_active": self._alert_active,        # Alerte en cours?
                "last_alert":   self._last_voice_time,     # Dernier moment d'alerte
                "cooldown":     self.config.voice_cooldown, # Cooldown configuré
                "threshold":    self.config.alert_threshold, # Seuil d'alerte
            }

    def update_config(self, **kwargs):
        """
        Met à jour la configuration dynamiquement (sans redémarrer).
        
        Args:
            **kwargs: Paramètres à modifier (alert_threshold, voice_cooldown, etc.)
        """
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
                    print(f" Configuration mise à jour: {k} = {v}")
    
    def reset_stats(self):
        """
        Réinitialise toutes les statistiques de la session.
        Utile pour recommencer une nouvelle session sans redémarrer l'application.
        """
        with self._lock:
            self.yawn_count = 0
            self.phone_alerts = 0
            self.global_score = 0
            self._alert_msg_idx = 0
            self._danger_msg_idx = 0
            self._yawn_timestamps = []
            self._alert_active = False
            self._last_voice_time = 0.0
            self._speaking = False
            print("🔄 Statistiques réinitialisées")
    
    def test_voice(self):
        """
        Teste la synthèse vocale.
        Utile pour vérifier que le système audio fonctionne correctement.
        
        Returns:
            Message de confirmation
        """
        print(" Test de la synthèse vocale...")
        self._speak("Test du système d'alerte Safe Drive Pro. La voix fonctionne correctement.")
        return "Test vocal envoyé"


# ── Test complet ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Bloc de test exécuté uniquement si le fichier est lancé directement.
    Simule différents scénarios pour vérifier le bon fonctionnement.
    """
    print("=" * 60)
    print("TEST DU MOTEUR D'ALERTES ")
    print("=" * 60)
    
    # Configuration avec cooldown court pour le test
    config = AlertConfig(
        alert_threshold=50,
        danger_score=75,
        voice_cooldown=3.0  # 3 secondes entre les alertes
    )
    
    # Création du moteur d'alertes
    engine = AlertEngine(config=config)
    
    # Test de la voix
    print("\n1. Test de la synthèse vocale...")
    engine.test_voice()
    time.sleep(2)
    
    # Simulation d'états de détection
    test_states = [
        ("État normal", DetectionState(face_detected=True, drowsy_confidence=0.2, phone_detected=False)),
        ("Alerte niveau 1", DetectionState(face_detected=True, drowsy_confidence=0.6, phone_detected=True, phone_confidence=0.7)),
        ("Alerte niveau 2", DetectionState(face_detected=True, drowsy_confidence=0.7, phone_detected=True, phone_confidence=0.8)),
        ("Retour normal", DetectionState(face_detected=True, drowsy_confidence=0.3, phone_detected=False)),
        ("Nouvelle alerte", DetectionState(face_detected=True, drowsy_confidence=0.65, phone_detected=True, phone_confidence=0.75)),
        ("Danger critique", DetectionState(face_detected=True, drowsy_confidence=0.9, phone_detected=True, phone_confidence=0.95, head_abnormal=True, yaw=25)),
    ]
    
    print("\n2. Simulation des alertes...")
    for name, state in test_states:
        print(f"\n--- {name} ---")
        score, events = engine.update(state)
        print(f"Score: {score}%")
        print(f"Événements: {events}")
        
        # Afficher les détails du score
        stats = engine.get_stats()
        print(f"Détails: Fatigue={stats['score_detail']['fatigue']:.0%}, "
              f"Bâillement={stats['score_detail']['yawn']:.0%}, "
              f"Téléphone={stats['score_detail']['phone']:.0%}")
        
        time.sleep(4)  # Attendre pour voir l'alerte suivante
    
    print("\n" + "=" * 60)
    print("STATISTIQUES FINALES")
    print("=" * 60)
    final_stats = engine.get_stats()
    for key, value in final_stats.items():
        print(f"{key}: {value}")
    
    print("\n✅ Test terminé avec succès!")