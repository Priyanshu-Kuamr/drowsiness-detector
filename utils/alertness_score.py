import time
import csv
import os
from collections import deque


class AlertnessScore:
    """
    Maintains a 0–100 alertness score that decays on fatigue signals
    and slowly recovers during normal driving.

    Score bands:
        80–100  → Alert  (green)
        50–79   → Mild fatigue (yellow)
        25–49   → Moderate fatigue (orange)
        0–24    → Severe fatigue (red) — hard alert
    """

    BAND_ALERT    = (80, 100)
    BAND_MILD     = (50, 79)
    BAND_MODERATE = (25, 49)
    BAND_SEVERE   = (0,  24)

    def __init__(self,
                 decay_rate:      float = 0.5,
                 recover_rate:    float = 0.1,
                 yawn_penalty:    float = 5.0,
                 head_penalty:    float = 0.3,
                 history_seconds: int   = 300):   # 5-min rolling chart
        self.decay_rate   = decay_rate
        self.recover_rate = recover_rate
        self.yawn_penalty = yawn_penalty
        self.head_penalty = head_penalty

        self._score = 100.0
        self._history: deque[tuple[float, float]] = deque()  # (timestamp, score)
        self._history_window = history_seconds

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, ear_low: bool = False,
               is_yawning: bool = False,
               head_off: bool = False) -> float:
        """
        Call once per frame. Applies penalties/recovery and records history.
        Returns current score.
        """
        if ear_low:
            self._score -= self.decay_rate
        elif head_off:
            self._score -= self.head_penalty
        else:
            self._score += self.recover_rate

        if is_yawning:
            self._score -= self.yawn_penalty

        self._score = max(0.0, min(100.0, self._score))

        now = time.time()
        self._history.append((now, round(self._score, 2)))

        # Prune old entries
        cutoff = now - self._history_window
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        return round(self._score, 2)

    @property
    def score(self) -> float:
        return round(self._score, 2)

    @property
    def band(self) -> str:
        s = self._score
        if s >= 80: return "ALERT"
        if s >= 50: return "MILD"
        if s >= 25: return "MODERATE"
        return "SEVERE"

    @property
    def band_color(self) -> tuple[int, int, int]:
        """BGR colour for OpenCV overlays."""
        return {
            "ALERT":    (50,  205, 50),
            "MILD":     (0,   200, 255),
            "MODERATE": (0,   140, 255),
            "SEVERE":   (0,   0,   255),
        }[self.band]

    @property
    def history(self) -> list[tuple[float, float]]:
        return list(self._history)

    def history_for_chart(self) -> dict:
        """Returns {labels: [...], values: [...]} for the dashboard chart."""
        items = list(self._history)
        if not items:
            return {"labels": [], "values": []}
        t0 = items[0][0]
        return {
            "labels": [round(t - t0, 1) for t, _ in items],
            "values": [v for _, v in items],
        }

    def reset(self):
        self._score = 100.0
        self._history.clear()


# ── Session Logger ─────────────────────────────────────────────────────────────

class SessionLogger:
    """Appends per-second snapshots to a CSV for post-session analysis."""

    COLUMNS = ["timestamp", "score", "band", "ear", "mar",
               "pitch", "yaw", "blinks_per_min", "total_yawns"]

    def __init__(self, filepath: str = "data/session_log.csv"):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._last_log = 0.0

        if not os.path.exists(filepath):
            with open(filepath, "w", newline="") as f:
                csv.writer(f).writerow(self.COLUMNS)

    def log(self, score: float, band: str, ear: float, mar: float,
            pitch: float, yaw: float, bpm: float, yawns: int,
            interval_seconds: float = 1.0):
        """Throttled logging — writes at most once every `interval_seconds`."""
        now = time.time()
        if now - self._last_log < interval_seconds:
            return
        self._last_log = now

        row = [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            round(score, 2), band,
            round(ear, 4), round(mar, 4),
            round(pitch, 2), round(yaw, 2),
            round(bpm, 1), yawns,
        ]
        with open(self.filepath, "a", newline="") as f:
            csv.writer(f).writerow(row)
