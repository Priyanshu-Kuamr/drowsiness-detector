import os
import time
import threading
import cv2
import numpy as np
import config


# ── Sound Alert ───────────────────────────────────────────────────────────────

class SoundAlert:
    """
    Plays an audio alert without blocking the main loop.
    Falls back gracefully if pygame is unavailable or sound file is missing.
    """

    def __init__(self, sound_file: str = config.ALERT_SOUND_FILE,
                 cooldown: float = config.ALERT_COOLDOWN_SEC):
        self.sound_file = sound_file
        self.cooldown   = cooldown
        self._last_played = 0.0
        self._available   = False

        try:
            import pygame
            pygame.mixer.init()
            if os.path.exists(sound_file):
                pygame.mixer.music.load(sound_file)
                self._available = True
                print(f"[SoundAlert] Loaded: {sound_file}")
            else:
                print(f"[SoundAlert] File not found: {sound_file} — using beep fallback")
        except Exception as e:
            print(f"[SoundAlert] pygame unavailable ({e}) — sound disabled")

    def play(self, force: bool = False):
        now = time.time()
        if not force and (now - self._last_played) < self.cooldown:
            return
        self._last_played = now

        if self._available:
            try:
                import pygame
                pygame.mixer.music.play(0)
            except Exception:
                pass
        else:
            # System beep fallback
            threading.Thread(target=self._beep, daemon=True).start()

    @staticmethod
    def _beep():
        print("\a", end="", flush=True)   # Terminal bell

    def stop(self):
        if self._available:
            try:
                import pygame
                pygame.mixer.music.stop()
            except Exception:
                pass


# ── On-Screen Overlay ─────────────────────────────────────────────────────────

def draw_hud(frame: np.ndarray,
             ear: float,
             mar: float,
             pitch: float,
             yaw: float,
             score: float,
             band: str,
             band_color: tuple,
             blinks_per_min: float,
             total_yawns: int,
             alert_msg: str = "") -> np.ndarray:
    """
    Draws a semi-transparent HUD overlay on the frame.
    All metric values are rendered in a dark pill in the top-left corner.
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Background panel
    panel_w, panel_h = 280, 200
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    def put(text, row, color=(220, 220, 220)):
        cv2.putText(frame, text, (18, 30 + row * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    put(f"EAR  : {ear:.3f}",            0)
    put(f"MAR  : {mar:.3f}",            1)
    put(f"Pitch: {pitch:+.1f}  Yaw: {yaw:+.1f}", 2)
    put(f"Blinks/min : {blinks_per_min:.1f}",     3)
    put(f"Yawns      : {total_yawns}",             4)
    put(f"Score: {score:.1f}  [{band}]",           5, band_color)

    # Alert banner
    if alert_msg:
        text_size = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]
        bx = (w - text_size[0]) // 2 - 10
        by = h - 70
        cv2.rectangle(frame, (bx, by - 40), (bx + text_size[0] + 20, by + 10),
                      (0, 0, 180), -1)
        cv2.putText(frame, alert_msg,
                    (bx + 10, by),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def draw_landmarks(frame: np.ndarray,
                   eye_left: np.ndarray,
                   eye_right: np.ndarray,
                   mouth: np.ndarray) -> np.ndarray:
    """Draw coloured dots on detected feature points."""
    for pt in eye_left:
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
    for pt in eye_right:
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
    for pt in mouth:
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (255, 165, 0), -1)
    return frame


# ── Danger Clip Recorder ──────────────────────────────────────────────────────

class DangerClipRecorder:
    """
    Keeps a rolling buffer of the last N seconds and saves to disk
    when a danger event is triggered.
    """

    def __init__(self, fps: float = 30,
                 buffer_seconds: int = config.DANGER_CLIP_SECONDS,
                 output_dir: str = config.OUTPUT_DIR):
        self.fps          = fps
        self.max_frames   = int(fps * buffer_seconds)
        self.output_dir   = output_dir
        self._buffer: list[np.ndarray] = []
        os.makedirs(output_dir, exist_ok=True)

    def push(self, frame: np.ndarray):
        """Add frame to rolling buffer."""
        self._buffer.append(frame.copy())
        if len(self._buffer) > self.max_frames:
            self._buffer.pop(0)

    def save(self, reason: str = "danger", score: float = 0.0):
        """Write buffer to a timestamped MP4 file in a background thread."""
        if not self._buffer:
            return
        frames = list(self._buffer)
        threading.Thread(
            target=self._write, args=(frames, reason, score), daemon=True
        ).start()

    def _write(self, frames: list[np.ndarray], reason: str, score: float = 0.0):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"{reason}_score{int(score)}_{timestamp}.mp4")
        h, w      = frames[0].shape[:2]
        fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
        writer    = cv2.VideoWriter(filename, fourcc, self.fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        print(f"[DangerClip] Saved: {filename}")
