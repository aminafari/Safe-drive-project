"""
video_capture.py — Capture vidéo dans un thread dédié
Évite les blocages du thread principal Tkinter
"""
import cv2
import threading
import time


class ThreadedCapture:
    """
    Capture vidéo non-bloquante.
    Lit les frames en arrière-plan et les met à disposition
    via self.frame sans jamais bloquer l'interface.
    """
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # buffer minimal = frames fraîches

        self.frame = None
        self.running = False
        self._lock = threading.Lock()
        self._thread = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return self
        self.running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read(self):
        with self._lock:
            return self.frame is not None, (
                self.frame.copy() if self.frame is not None else None
            )

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
        self.cap.release()

    def is_opened(self):
        return self.cap.isOpened()
