"""
main.py — Safe Drive Pro · Design Navy Compact
Version optimisée pour petits écrans avec caméra responsive
"""

import os
import torch
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import time
from datetime import datetime
import json
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from video_capture import ThreadedCapture
from detection import FaceDetector
from alert_engine import AlertEngine, AlertConfig, DetectionState

try:
    from yolo_detector import YOLODetector
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

try:
    from drowsiness_model_loader import load_drowsiness_model
    _LOADER_AVAILABLE = True
except ImportError:
    _LOADER_AVAILABLE = False


# ═══════════════════════════════════════════
# PALETTE — Navy Dashboard
# ═══════════════════════════════════════════
class C:
    BG         = "#0B0F1E"
    BG_PANEL   = "#111628"
    BG_CARD    = "#161C30"
    BG_CARD2   = "#1C2340"
    SIDEBAR    = "#0D1120"
    CYAN       = "#00C2E0"
    PINK       = "#E040A0"
    BLUE_LT    = "#3D7AFF"
    SUCCESS    = "#00C2A0"
    WARNING    = "#E0A040"
    DANGER     = "#E04060"
    TXT        = "#E8EAF6"
    TXT2       = "#7B8DB0"
    TXT3       = "#3D4E70"
    BORDER     = "#1E2A45"


class SafeDrivePro:
    def __init__(self, win: tk.Tk):
        self.win = win
        win.title("Safe Drive Pro")
        win.geometry("1200x750")
        win.minsize(1000, 650)
        win.configure(bg=C.BG)
        
        self.running = False
        self._cap = None
        self._t0 = None
        self._last_alert_time = 0.0
        self.session_stats = self._load_stats()
        
        # IA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, loaded = None, False
        if _LOADER_AVAILABLE:
            try:
                model, loaded = load_drowsiness_model("drowsiness_model.pth", self.device)
                if loaded:
                    model.eval()
            except:
                pass
        
        self.detector = FaceDetector(drowsiness_model=model, device=self.device, model_loaded=loaded)
        self.yolo = YOLODetector() if _YOLO_AVAILABLE else None
        self.engine = AlertEngine(config=AlertConfig())
        
        self._build_ui()
        self._update_clock()
        win.protocol("WM_DELETE_WINDOW", self._close)
    
    def _load_stats(self):
        f = Path("session_stats.json")
        if f.exists():
            try:
                with open(f) as fp:
                    return json.load(fp)
            except:
                pass
        return {"total_sessions": 0, "total_alerts": 0, "total_yawns": 0,
                "avg_score": 0, "last_session": None}
    
    def _save_stats(self):
        with open("session_stats.json", "w") as fp:
            json.dump(self.session_stats, fp)
    
    def _build_ui(self):
        # Panneau principal divisé en 2
        main_panel = tk.Frame(self.win, bg=C.BG)
        main_panel.pack(fill="both", expand=True, padx=10, pady=10)
        
        # COLONNE GAUCHE - Vidéo (prend 55% de l'espace)
        left_panel = tk.Frame(main_panel, bg=C.BG)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 8))
        
        # Carte vidéo
        video_card = tk.Frame(left_panel, bg=C.BG_CARD, bd=1, highlightthickness=1, highlightbackground=C.BORDER)
        video_card.pack(fill="both", expand=True)
        
        # Header vidéo
        video_header = tk.Frame(video_card, bg=C.BG_CARD2, height=35)
        video_header.pack(fill="x")
        video_header.pack_propagate(False)
        
        tk.Label(video_header, text="📹 FLUX CAMÉRA", font=("Segoe UI", 9, "bold"),
                fg=C.TXT, bg=C.BG_CARD2).pack(side="left", padx=12)
        
        self.cam_status = tk.Label(video_header, text="● HORS LIGNE", font=("Segoe UI", 8),
                                   fg=C.TXT3, bg=C.BG_CARD2)
        self.cam_status.pack(side="right", padx=12)
        
        # Zone vidéo avec taille fixe proportionnelle
        self.video_container = tk.Frame(video_card, bg="#07091A", height=400)
        self.video_container.pack(fill="both", expand=True)
        self.video_container.pack_propagate(False)
        
        self.video_label = tk.Label(self.video_container, bg="#07091A")
        self.video_label.pack(fill="both", expand=True)
        
        # Barre de score sous la vidéo
        score_bar_frame = tk.Frame(video_card, bg=C.BG_CARD, height=40)
        score_bar_frame.pack(fill="x")
        score_bar_frame.pack_propagate(False)
        
        tk.Label(score_bar_frame, text="NIVEAU DE RISQUE", font=("Segoe UI", 8),
                fg=C.TXT2, bg=C.BG_CARD).pack(anchor="w", padx=12, pady=(5, 0))
        
        self.score_canvas = tk.Canvas(score_bar_frame, height=8, bg=C.BORDER, highlightthickness=0)
        self.score_canvas.pack(fill="x", padx=12, pady=(3, 5))
        self.score_bar = self.score_canvas.create_rectangle(0, 0, 0, 8, fill=C.CYAN, width=0)
        
        # Bande d'alerte
        self.alert_band = tk.Frame(video_card, bg=C.BG_CARD, height=32)
        self.alert_band.pack(fill="x")
        self.alert_band.pack_propagate(False)
        self.alert_label = tk.Label(self.alert_band, text="", font=("Segoe UI", 9, "bold"),
                                    fg=C.PINK, bg=C.BG_CARD)
        self.alert_label.pack(expand=True)
        
        # Contrôles
        controls = tk.Frame(video_card, bg=C.BG_CARD2, height=45)
        controls.pack(fill="x")
        controls.pack_propagate(False)
        
        btn_frame = tk.Frame(controls, bg=C.BG_CARD2)
        btn_frame.pack(expand=True)
        
        self.start_btn = self._make_btn(btn_frame, "▶ DÉMARRER", self.start, C.SUCCESS)
        self.start_btn.pack(side="left", padx=4)
        
        self.stop_btn = self._make_btn(btn_frame, "■ ARRÊTER", self.stop, C.DANGER)
        self.stop_btn.pack(side="left", padx=4)
        self.stop_btn.config(state="disabled")
        
        reset_btn = self._make_btn(btn_frame, "🔄 RESET", self._reset_session, C.WARNING)
        reset_btn.pack(side="left", padx=4)
        
        settings_btn = self._make_btn(btn_frame, "⚙ CONFIG", self._open_settings, C.BLUE_LT)
        settings_btn.pack(side="left", padx=4)
        
        # COLONNE DROITE - Dashboard (45% de l'espace)
        right_panel = tk.Frame(main_panel, bg=C.BG, width=350)
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)
        
        # Scroll pour le dashboard
        canvas = tk.Canvas(right_panel, bg=C.BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(right_panel, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=C.BG)
        
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw", width=350)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # --- Contenu dashboard compact ---
        
        # 1. Score principal
        score_card = tk.Frame(scrollable, bg=C.BG_CARD, bd=1, highlightthickness=1, highlightbackground=C.BORDER)
        score_card.pack(fill="x", pady=(0, 8))
        
        tk.Label(score_card, text="SCORE GLOBAL", font=("Segoe UI", 8),
                fg=C.TXT2, bg=C.BG_CARD).pack(pady=(10, 2))
        
        self.score_label = tk.Label(score_card, text="0", font=("Segoe UI", 42, "bold"),
                                   fg=C.CYAN, bg=C.BG_CARD)
        self.score_label.pack()
        
        self.score_status = tk.Label(score_card, text="SÉCURISÉ", font=("Segoe UI", 9, "bold"),
                                    fg=C.CYAN, bg=C.BG_CARD)
        self.score_status.pack(pady=(2, 10))
        
        # 2. Métriques en grille 2x2
        metrics_card = tk.Frame(scrollable, bg=C.BG_CARD, bd=1, highlightthickness=1, highlightbackground=C.BORDER)
        metrics_card.pack(fill="x", pady=(0, 8))
        
        tk.Label(metrics_card, text="MÉTRIQUES", font=("Segoe UI", 8),
                fg=C.TXT2, bg=C.BG_CARD).pack(anchor="w", padx=12, pady=(8, 4))
        
        grid = tk.Frame(metrics_card, bg=C.BG_CARD)
        grid.pack(fill="x", padx=10, pady=(0, 10))
        
        for i in range(2):
            grid.columnconfigure(i, weight=1)
        
        # Téléphone
        phone_frame = tk.Frame(grid, bg=C.BG_CARD2, bd=1, highlightthickness=1, highlightbackground=C.BORDER)
        phone_frame.grid(row=0, column=0, padx=3, pady=3, sticky="nsew")
        tk.Label(phone_frame, text="📱 TÉLÉPHONE", font=("Segoe UI", 7),
                fg=C.TXT2, bg=C.BG_CARD2).pack(anchor="w", padx=8, pady=(6, 0))
        self.phone_val = tk.Label(phone_frame, text="Non", font=("Segoe UI", 12, "bold"),
                                  fg=C.TXT, bg=C.BG_CARD2)
        self.phone_val.pack(anchor="w", padx=8, pady=(0, 6))
        
        # Fatigue
        fatigue_frame = tk.Frame(grid, bg=C.BG_CARD2, bd=1, highlightthickness=1, highlightbackground=C.BORDER)
        fatigue_frame.grid(row=0, column=1, padx=3, pady=3, sticky="nsew")
        tk.Label(fatigue_frame, text="😴 FATIGUE", font=("Segoe UI", 7),
                fg=C.TXT2, bg=C.BG_CARD2).pack(anchor="w", padx=8, pady=(6, 0))
        self.fatigue_val = tk.Label(fatigue_frame, text="0%", font=("Segoe UI", 12, "bold"),
                                    fg=C.TXT, bg=C.BG_CARD2)
        self.fatigue_val.pack(anchor="w", padx=8, pady=(0, 6))
        
        # Bâillements
        yawn_frame = tk.Frame(grid, bg=C.BG_CARD2, bd=1, highlightthickness=1, highlightbackground=C.BORDER)
        yawn_frame.grid(row=1, column=0, padx=3, pady=3, sticky="nsew")
        tk.Label(yawn_frame, text="🥱 BÂILLEMENTS", font=("Segoe UI", 7),
                fg=C.TXT2, bg=C.BG_CARD2).pack(anchor="w", padx=8, pady=(6, 0))
        self.yawn_val = tk.Label(yawn_frame, text="0", font=("Segoe UI", 12, "bold"),
                                 fg=C.TXT, bg=C.BG_CARD2)
        self.yawn_val.pack(anchor="w", padx=8, pady=(0, 6))
        
        # Position tête
        head_frame = tk.Frame(grid, bg=C.BG_CARD2, bd=1, highlightthickness=1, highlightbackground=C.BORDER)
        head_frame.grid(row=1, column=1, padx=3, pady=3, sticky="nsew")
        tk.Label(head_frame, text="🔄 POSITION TÊTE", font=("Segoe UI", 7),
                fg=C.TXT2, bg=C.BG_CARD2).pack(anchor="w", padx=8, pady=(6, 0))
        self.head_val = tk.Label(head_frame, text="Normal", font=("Segoe UI", 12, "bold"),
                                 fg=C.TXT, bg=C.BG_CARD2)
        self.head_val.pack(anchor="w", padx=8, pady=(0, 6))
        
        # 3. Indicateurs avec barres
        indicators_card = tk.Frame(scrollable, bg=C.BG_CARD, bd=1, highlightthickness=1, highlightbackground=C.BORDER)
        indicators_card.pack(fill="x", pady=(0, 8))
        
        tk.Label(indicators_card, text="INDICATEURS", font=("Segoe UI", 8),
                fg=C.TXT2, bg=C.BG_CARD).pack(anchor="w", padx=12, pady=(8, 4))
        
        self._add_indicator(indicators_card, "Fatigue IA", 0)
        self._add_indicator(indicators_card, "Bâillement", 1)
        self._add_indicator(indicators_card, "Position tête", 2)
        self._add_indicator(indicators_card, "Téléphone", 3)
        
        # 4. Alertes
        alerts_card = tk.Frame(scrollable, bg=C.BG_CARD, bd=1, highlightthickness=1, highlightbackground=C.BORDER)
        alerts_card.pack(fill="x")
        
        tk.Label(alerts_card, text="ALERTES", font=("Segoe UI", 8),
                fg=C.TXT2, bg=C.BG_CARD).pack(anchor="w", padx=12, pady=(8, 4))
        
        self.alerts_msg = tk.Label(alerts_card, text="Aucune alerte", font=("Segoe UI", 9),
                                   fg=C.TXT3, bg=C.BG_CARD)
        self.alerts_msg.pack(anchor="w", padx=12, pady=(0, 4))
        
        self.alerts_count = tk.Label(alerts_card, text="Alertes: 0", font=("Segoe UI", 8),
                                     fg=C.TXT2, bg=C.BG_CARD)
        self.alerts_count.pack(anchor="e", padx=12, pady=(0, 10))
        
        # Placeholder
        self._show_placeholder()
    
    def _add_indicator(self, parent, label, row):
        """Ajoute un indicateur avec barre"""
        frame = tk.Frame(parent, bg=C.BG_CARD)
        frame.pack(fill="x", padx=12, pady=3)
        
        tk.Label(frame, text=label, font=("Segoe UI", 8),
                fg=C.TXT2, bg=C.BG_CARD).pack(side="left")
        
        val_label = tk.Label(frame, text="0%", font=("Segoe UI", 9, "bold"),
                            fg=C.TXT, bg=C.BG_CARD)
        val_label.pack(side="right")
        
        canvas = tk.Canvas(parent, height=3, bg=C.BORDER, highlightthickness=0)
        canvas.pack(fill="x", padx=12, pady=(0, 5))
        bar = canvas.create_rectangle(0, 0, 0, 3, fill=C.CYAN, width=0)
        
        setattr(self, f"ind_{label.replace(' ', '_')}", val_label)
        setattr(self, f"ind_{label.replace(' ', '_')}_bar", (canvas, bar))
    
    def _make_btn(self, parent, text, cmd, color):
        btn = tk.Button(parent, text=text, command=cmd,
                       font=("Segoe UI", 9, "bold"),
                       bg=color, fg="white",
                       activebackground=color, activeforeground="white",
                       relief="flat", cursor="hand2", padx=12, pady=5)
        return btn
    
    def _show_placeholder(self):
        """Affiche un placeholder avec les stats"""
        for w in self.video_container.winfo_children():
            w.destroy()
        
        placeholder = tk.Frame(self.video_container, bg="#07091A")
        placeholder.pack(fill="both", expand=True)
        
        # Logo
        tk.Label(placeholder, text="🚗", font=("Segoe UI", 48),
                bg="#07091A").pack(pady=(40, 10))
        tk.Label(placeholder, text="SAFE DRIVE PRO", font=("Segoe UI", 16, "bold"),
                fg=C.CYAN, bg="#07091A").pack()
        tk.Label(placeholder, text="Système de surveillance conducteur IA",
                font=("Segoe UI", 9), fg=C.TXT2, bg="#07091A").pack()
        
        # Stats rapides
        stats_frame = tk.Frame(placeholder, bg="#07091A")
        stats_frame.pack(pady=20)
        
        stats = [
            ("📊", str(self.session_stats["total_sessions"]), "Sessions"),
            ("🔔", str(self.session_stats["total_alerts"]), "Alertes"),
            ("🥱", str(self.session_stats["total_yawns"]), "Bâillements"),
            ("🎯", f"{self.session_stats['avg_score']}%", "Score moy."),
        ]
        
        for i, (icon, val, lbl) in enumerate(stats):
            card = tk.Frame(stats_frame, bg=C.BG_CARD, bd=1, highlightthickness=1,
                           highlightbackground=C.BORDER)
            card.grid(row=0, column=i, padx=4, pady=4, sticky="nsew")
            tk.Label(card, text=icon, font=("Segoe UI", 20),
                    bg=C.BG_CARD).pack(pady=(8, 0))
            tk.Label(card, text=val, font=("Segoe UI", 16, "bold"),
                    fg=C.CYAN, bg=C.BG_CARD).pack()
            tk.Label(card, text=lbl, font=("Segoe UI", 7),
                    fg=C.TXT2, bg=C.BG_CARD).pack(pady=(0, 8))
        
        # Message
        last = self.session_stats.get("last_session", "Jamais")
        if last and last != "Jamais":
            last = last[:16]
        tk.Label(placeholder, text=f"Dernière session: {last}",
                font=("Segoe UI", 8), fg=C.TXT3, bg="#07091A").pack(pady=10)
        
        self.start_hint = tk.Label(placeholder, text="▶ Cliquez sur DÉMARRER pour commencer",
                                   font=("Segoe UI", 9, "bold"),
                                   fg=C.CYAN, bg="#07091A")
        self.start_hint.pack()
        self._pulse_hint()
    
    def _pulse_hint(self):
        if hasattr(self, 'start_hint') and self.start_hint.winfo_exists():
            cur = self.start_hint.cget("fg")
            self.start_hint.config(fg=C.CYAN if cur == C.TXT3 else C.TXT3)
            self.win.after(800, self._pulse_hint)
    
    def _clear_placeholder(self):
        for w in self.video_container.winfo_children():
            w.destroy()
        self.video_label = tk.Label(self.video_container, bg="#07091A")
        self.video_label.pack(fill="both", expand=True)
    
    def _update_clock(self):
        self.win.after(1000, self._update_clock)
    
    def _open_settings(self):
        w = tk.Toplevel(self.win)
        w.title("Paramètres")
        w.geometry("400x400")
        w.configure(bg=C.BG)
        w.resizable(False, False)
        
        tk.Label(w, text="PARAMÈTRES", font=("Segoe UI", 13, "bold"),
                fg=C.CYAN, bg=C.BG).pack(pady=15)
        
        body = tk.Frame(w, bg=C.BG)
        body.pack(fill="both", expand=True, padx=20)
        
        def add_slider(parent, label, from_, to_, default, key):
            frame = tk.Frame(parent, bg=C.BG)
            frame.pack(fill="x", pady=8)
            
            tk.Label(frame, text=label, font=("Segoe UI", 9),
                    fg=C.TXT2, bg=C.BG).pack(anchor="w")
            
            var = tk.DoubleVar(value=default)
            slider = ttk.Scale(frame, from_=from_, to=to_, variable=var, orient="horizontal")
            slider.pack(fill="x", pady=4)
            
            val_label = tk.Label(frame, text=f"{default:.0f}", font=("Segoe UI", 9),
                                 fg=C.CYAN, bg=C.BG)
            val_label.pack(anchor="e")
            
            var.trace('w', lambda *a: val_label.config(text=f"{var.get():.0f}"))
            return var
        
        alert_var = add_slider(body, "Seuil d'alerte (%)", 30, 80, 50, "alert")
        cooldown_var = add_slider(body, "Cooldown (s)", 2, 10, 3, "cooldown")
        
        def save():
            if self.engine:
                self.engine.update_config(
                    alert_threshold=int(alert_var.get()),
                    voice_cooldown=float(cooldown_var.get())
                )
            w.destroy()
        
        tk.Button(body, text="Appliquer", command=save,
                 bg=C.CYAN, fg=C.BG, font=("Segoe UI", 10, "bold"),
                 relief="flat", pady=8).pack(fill="x", pady=15)
    
    def _reset_session(self):
        if self.engine:
            self.engine.reset_stats()
        self.alerts_msg.config(text="Session réinitialisée", fg=C.SUCCESS)
        self.alerts_count.config(text="Alertes: 0")
        self.yawn_val.config(text="0")
        self.score_label.config(text="0", fg=C.CYAN)
        self.score_status.config(text="SÉCURISÉ", fg=C.CYAN)
        self.score_canvas.coords(self.score_bar, 0, 0, 0, 8)
    
    def start(self):
        if self.running:
            return
        self._cap = ThreadedCapture(src=0).start()
        time.sleep(0.4)
        if not self._cap.is_opened():
            messagebox.showerror("Erreur", "Impossible d'accéder à la caméra.")
            return
        self.running = True
        self._t0 = time.time()
        self.cam_status.config(text="● EN LIGNE", fg=C.SUCCESS)
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._clear_placeholder()
        self._loop()
    
    def stop(self):
        self.running = False
        if self._cap:
            self._cap.stop()
            self._cap = None
        
        if self.engine:
            stats = self.engine.get_stats()
            self.session_stats["total_sessions"] += 1
            self.session_stats["total_alerts"] += stats.get("phone_alerts", 0)
            self.session_stats["total_yawns"] += stats.get("yawn_count", 0)
            total_s = self.session_stats.get("total_score", 0) + stats.get("global_score", 0)
            self.session_stats["total_score"] = total_s
            self.session_stats["avg_score"] = total_s // max(1, self.session_stats["total_sessions"])
            self.session_stats["last_session"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            self._save_stats()
        
        self.cam_status.config(text="● HORS LIGNE", fg=C.TXT3)
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._show_placeholder()
        self.alerts_msg.config(text="Surveillance inactive", fg=C.TXT3)
    
    def _close(self):
        self.stop()
        self.win.destroy()
    
    def _loop(self):
        if not self.running:
            return
        
        ret, frame = self._cap.read()
        if not ret or frame is None:
            self.win.after(15, self._loop)
            return
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Détections
        phone_on, phone_cf = False, 0.0
        if self.yolo:
            try:
                distractions, frame = self.yolo.detect_distractions(frame)
                ph_labels = ("cell phone", "phone", "mobile")
                ph_items = [d for d in distractions if d['label'].lower() in ph_labels]
                phone_on = len(ph_items) > 0
                phone_cf = max((d['confidence'] for d in ph_items), default=0.0)
            except:
                pass
        
        frame, fs = self.detector.process(frame)
        
        state = DetectionState(
            drowsy_confidence=fs.drowsy_confidence,
            yawn_detected=fs.yawn_detected,
            phone_detected=phone_on,
            phone_confidence=phone_cf,
            face_detected=fs.face_detected,
            head_status=getattr(fs, 'head_status', 'Normal'),
            head_abnormal=getattr(fs, 'head_abnormal', False),
            yaw=getattr(fs, 'yaw', 0.0),
            pitch=getattr(fs, 'pitch', 0.0),
            roll=getattr(fs, 'roll', 0.0),
            is_drowsy_ai=getattr(fs, 'is_drowsy_ai', False),
        )
        
        score, events = self.engine.update(state)
        self._update_ui(state, score, events)
        self._draw_overlay(frame, score, w)
        self._show_frame(frame)
        
        self.win.after(10, self._loop)
    
    def _draw_overlay(self, frame, score, w):
        h = frame.shape[0]
        thr = self.engine.config.alert_threshold if self.engine else 50
        
        if score < thr:
            color = (0, 194, 224)
        else:
            color = (224, 64, 160)
        
        # Barre top
        cv2.rectangle(frame, (0, 0), (w, 32), (7, 9, 26), -1)
        cv2.putText(frame, "SAFE DRIVE", (8, 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 90, 110), 1)
        cv2.putText(frame, f"SCORE: {score}%", (w-100, 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        
        # Barre bottom
        bar_y = h - 6
        cv2.rectangle(frame, (0, bar_y), (w, h), (10, 13, 30), -1)
        bw = int(w * score / 100)
        if bw > 0:
            cv2.rectangle(frame, (0, bar_y), (bw, h), color, -1)
        
        # Bordure alerte
        if score >= thr and int(time.time() * 2) % 2:
            cv2.rectangle(frame, (0, 0), (w, h), (224, 64, 160), 2)
    
    def _update_ui(self, state, score, events):
        thr = self.engine.config.alert_threshold if self.engine else 50
        alert = score >= thr
        
        # Couleur
        col = C.PINK if alert else C.CYAN
        
        # Score
        if score < 40:
            status = "SÉCURISÉ"
        elif score < thr:
            status = "VIGILANCE"
        else:
            status = "ALERTE"
        
        self.score_label.config(text=str(score), fg=col)
        self.score_status.config(text=status, fg=col)
        
        # Barre
        width = self.score_canvas.winfo_width()
        if width > 0:
            bar_width = (score / 100) * width
            self.score_canvas.coords(self.score_bar, 0, 0, bar_width, 8)
            self.score_canvas.itemconfig(self.score_bar, fill=col)
        
        # Métriques
        self.phone_val.config(text="Oui" if state.phone_detected else "Non",
                             fg=C.PINK if state.phone_detected else C.SUCCESS)
        
        fc = state.drowsy_confidence
        fc_col = C.PINK if fc > 0.7 else C.WARNING if fc > 0.4 else C.SUCCESS
        self.fatigue_val.config(text=f"{fc:.0%}", fg=fc_col)
        
        stats = self.engine.get_stats()
        self.yawn_val.config(text=str(stats.get("yawn_count", 0)))
        self.head_val.config(text=state.head_status,
                            fg=C.PINK if state.head_abnormal else C.SUCCESS)
        
        # Indicateurs
        detail = stats.get("score_detail", {})
        indicators = [
            ("Fatigue_IA", detail.get("fatigue", 0)),
            ("Bâillement", detail.get("yawn", 0)),
            ("Position_tête", detail.get("head", 0)),
            ("Téléphone", detail.get("phone", 0)),
        ]
        
        for name, val in indicators:
            pct = int(val * 100)
            label = getattr(self, f"ind_{name}", None)
            if label:
                if pct < 40:
                    label_c = C.SUCCESS
                elif pct < 70:
                    label_c = C.WARNING
                else:
                    label_c = C.PINK
                label.config(text=f"{pct}%", fg=label_c)
            
            bar_data = getattr(self, f"ind_{name}_bar", None)
            if bar_data:
                canvas, bar = bar_data
                width = canvas.winfo_width()
                if width > 0:
                    bar_width = (pct / 100) * width
                    canvas.coords(bar, 0, 0, bar_width, 3)
                    if pct < 40:
                        canvas.itemconfig(bar, fill=C.SUCCESS)
                    elif pct < 70:
                        canvas.itemconfig(bar, fill=C.WARNING)
                    else:
                        canvas.itemconfig(bar, fill=C.PINK)
        
        # Alertes
        alert_count = stats.get("phone_alerts", 0)
        self.alerts_count.config(text=f"Alertes: {alert_count}",
                                 fg=C.PINK if alert_count > 0 else C.TXT2)
        
        if "voice_alert" in events:
            self._last_alert_time = time.time()
            self.alerts_msg.config(text=f"🔔 ALERTE - {datetime.now().strftime('%H:%M:%S')}",
                                   fg=C.PINK)
            self.alert_label.config(text="⚠ ALERTE VOCALE DÉCLENCHÉE")
            self.alert_band.config(bg="#1A0818")
        elif time.time() - self._last_alert_time > 4:
            self.alerts_msg.config(text="✅ Surveillance active", fg=C.SUCCESS)
            self.alert_label.config(text="")
            self.alert_band.config(bg=C.BG_CARD)
    
    def _show_frame(self, frame):
        try:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Redimensionnement proportionnel
            container_w = self.video_container.winfo_width()
            container_h = self.video_container.winfo_height()
            if container_w > 10 and container_h > 10:
                img = img.resize((container_w, container_h), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        except Exception as e:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = SafeDrivePro(root)
    root.mainloop()