import numpy as np
from scipy.spatial.distance import euclidean


def compute_ear(eye_points: np.ndarray) -> float:
    """
    Eye Aspect Ratio (Soukupová & Čech, 2016).

    Eye landmark order (MediaPipe, 6 points):
        p1 = outer corner   (index 0)
        p2 = upper-outer    (index 1)
        p3 = upper-inner    (index 2)
        p4 = inner corner   (index 3)
        p5 = lower-inner    (index 4)
        p6 = lower-outer    (index 5)

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Open eye  → EAR ≈ 0.30–0.35
    Closed eye → EAR < 0.20
    """
    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return round(ear, 4)


def compute_avg_ear(left_eye: np.ndarray, right_eye: np.ndarray) -> float:
    """Average EAR across both eyes — more robust than single-eye."""
    return (compute_ear(left_eye) + compute_ear(right_eye)) / 2.0


class BlinkTracker:
    """
    Tracks blink count and rate over a rolling time window.

    Usage:
        tracker = BlinkTracker(fps=30, window_seconds=60)
        for each frame:
            tracker.update(ear)
        print(tracker.blinks_per_minute)
    """

    def __init__(self, ear_threshold: float = 0.25,
                 consec_frames: int = 3,
                 fps: float = 30,
                 window_seconds: int = 60):

        self.ear_threshold  = ear_threshold
        self.consec_frames  = consec_frames
        self.fps            = fps
        self.window_frames  = int(fps * window_seconds)

        self._closed_counter = 0          
        self._blink_timestamps: list[int] = []   
        self._frame_count    = 0
        self.total_blinks    = 0

    def update(self, ear: float) -> bool:
        """
        Call once per frame with the current EAR value.
        Returns True if a new blink was just detected.
        """
        self._frame_count += 1
        blink_detected = False

        if ear < self.ear_threshold:
            self._closed_counter += 1
        else:
            # Eye just opened — was it a blink?
            if self._closed_counter >= self.consec_frames:
                self.total_blinks += 1
                self._blink_timestamps.append(self._frame_count)
                blink_detected = True
            self._closed_counter = 0

        # Prune old timestamps outside the rolling window
        cutoff = self._frame_count - self.window_frames
        self._blink_timestamps = [t for t in self._blink_timestamps if t > cutoff]

        return blink_detected

    @property
    def blinks_per_minute(self) -> float:
        """Blink rate in the rolling window, normalised to blinks/minute."""
        if self._frame_count == 0:
            return 0.0
        elapsed_minutes = min(self._frame_count, self.window_frames) / (self.fps * 60)
        if elapsed_minutes == 0:
            return 0.0
        return round(len(self._blink_timestamps) / elapsed_minutes, 1)

    @property
    def is_eye_closed(self) -> bool:
        return self._closed_counter > 0

    @property
    def closed_frame_count(self) -> int:
        return self._closed_counter

    def reset(self):
        self._closed_counter     = 0
        self._blink_timestamps   = []
        self._frame_count        = 0
        self.total_blinks        = 0
