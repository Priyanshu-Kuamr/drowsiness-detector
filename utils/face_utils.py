import cv2
import numpy as np
from scipy.spatial.distance import euclidean


# ─── Yawn Detection ───────────────────────────────────────────────────────────

def compute_mar(mouth_points: np.ndarray) -> float:
    """
    Mouth Aspect Ratio — analogous to EAR.

    MediaPipe outer-lip landmark order (8 points):
        0  = left corner
        1  = right corner
        2  = upper-left
        3  = upper-mid-left
        4  = upper-mid-right  (top centre)
        5  = lower-centre
        6  = lower-mid-right
        7  = lower-right

    We use a simplified 3-vertical / 1-horizontal ratio:
        MAR = (||p3-p7|| + ||p4-p6|| + ||p5-p8||) / (3 * ||p1-p2||)
    """
    p1 = mouth_points[0]   # left corner
    p2 = mouth_points[1]   # right corner
    p3 = mouth_points[2]
    p4 = mouth_points[3]
    p5 = mouth_points[4]
    p6 = mouth_points[5]
    p7 = mouth_points[6]
    p8 = mouth_points[7]

    v1 = euclidean(p3, p7)
    v2 = euclidean(p4, p6)
    v3 = euclidean(p5, p8)
    h  = euclidean(p1, p2)

    if h == 0:
        return 0.0
    return round((v1 + v2 + v3) / (3.0 * h), 4)


class YawnTracker:
    """Counts yawns and tracks consecutive open-mouth frames."""

    def __init__(self, mar_threshold: float = 0.75,
                 consec_frames: int = 15):
        self.mar_threshold  = mar_threshold
        self.consec_frames  = consec_frames
        self._open_counter  = 0
        self.total_yawns    = 0

    def update(self, mar: float) -> bool:
        """Returns True when a new yawn is registered."""
        yawned = False
        if mar > self.mar_threshold:
            self._open_counter += 1
        else:
            if self._open_counter >= self.consec_frames:
                self.total_yawns += 1
                yawned = True
            self._open_counter = 0
        return yawned

    @property
    def is_yawning(self) -> bool:
        return self._open_counter >= self.consec_frames

    @property
    def open_frame_count(self) -> int:
        return self._open_counter

    def reset(self):
        self._open_counter = 0
        self.total_yawns   = 0


# ─── Head Pose ────────────────────────────────────────────────────────────────

def rotation_vector_to_euler(rotation_vector) -> tuple[float, float, float]:
    """
    Convert a Rodrigues rotation vector to Euler angles (degrees).
    Returns (pitch, yaw, roll).
    """
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat([rotation_matrix, np.zeros((3, 1))])
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch = euler_angles[0, 0]
    yaw   = euler_angles[1, 0]
    roll  = euler_angles[2, 0]
    return float(pitch), float(yaw), float(roll)


class HeadPoseTracker:
    """
    Classifies head orientation and counts distracted frames.
    Distracted = yaw > threshold (looking sideways)
                 or pitch > threshold (looking down at phone)
    """

    def __init__(self, pitch_threshold: float = 20.0,
                 yaw_threshold: float = 30.0,
                 consec_frames: int = 25):
        self.pitch_threshold = pitch_threshold
        self.yaw_threshold   = yaw_threshold
        self.consec_frames   = consec_frames
        self._off_counter    = 0
        self.total_distracted_events = 0

    def update(self, pitch: float, yaw: float) -> bool:
        """Returns True when a new distraction event fires."""
        is_off = (abs(pitch) > self.pitch_threshold or
                  abs(yaw)   > self.yaw_threshold)
        event = False
        if is_off:
            self._off_counter += 1
            if self._off_counter == self.consec_frames:   # Rising edge only
                self.total_distracted_events += 1
                event = True
        else:
            self._off_counter = 0
        return event

    @property
    def is_distracted(self) -> bool:
        return self._off_counter >= self.consec_frames

    @property
    def off_frame_count(self) -> int:
        return self._off_counter

    def reset(self):
        self._off_counter = 0
        self.total_distracted_events = 0
