import mediapipe as mp
import numpy as np


class FaceDetector:
    """
    Uses MediaPipe Face Mesh to detect 468 facial landmarks per frame.
    Returns normalised (x, y, z) coordinates.
    """

    # MediaPipe landmark indices for key regions
    # Left eye  (from MediaPipe canonical face map)
    LEFT_EYE  = [362, 385, 387, 263, 373, 380]
    # Right eye
    RIGHT_EYE = [33,  160, 158, 133, 153, 144]
    # Mouth (outer lip)
    MOUTH     = [61,  291, 39,  181, 0,   17,  269, 405]
    # Nose tip + chin + left/right temple (for head pose)
    HEAD_POSE_POINTS = [1, 33, 263, 61, 291, 199]

    def __init__(self, static_mode=False, max_faces=1,
                 refine_landmarks=True, min_detection_confidence=0.6,
                 min_tracking_confidence=0.6):

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def find_landmarks(self, frame_rgb):
        """
        Process an RGB frame.
        Returns list of (x_px, y_px, z) tuples for all 468 landmarks,
        or None if no face found.
        """
        results = self.face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None

        h, w = frame_rgb.shape[:2]
        face = results.multi_face_landmarks[0]
        landmarks = [
            (int(lm.x * w), int(lm.y * h), lm.z)
            for lm in face.landmark
        ]
        return landmarks

    def get_eye_points(self, landmarks, eye="left"):
        """Return the 6 landmark (x,y) tuples for the chosen eye."""
        indices = self.LEFT_EYE if eye == "left" else self.RIGHT_EYE
        return np.array([(landmarks[i][0], landmarks[i][1]) for i in indices],
                        dtype=np.float64)

    def get_mouth_points(self, landmarks):
        """Return 8 outer lip landmark (x,y) tuples."""
        return np.array([(landmarks[i][0], landmarks[i][1]) for i in self.MOUTH],
                        dtype=np.float64)

    def get_head_pose_points(self, landmarks, frame_shape):
        """
        Return 2-D image points and matching 3-D model points
        used for solvePnP head-pose estimation.
        """
        import cv2

        h, w = frame_shape[:2]

        # 3-D model points (generic head model, mm scale)
        model_3d = np.array([
            (0.0,    0.0,    0.0),    # Nose tip
            (-30.0, -30.0, -30.0),   # Left eye corner
            (30.0,  -30.0, -30.0),   # Right eye corner
            (-25.0,  17.0, -13.0),   # Left mouth corner
            (25.0,   17.0, -13.0),   # Right mouth corner
            (0.0,    75.0, -15.0),   # Chin
        ], dtype=np.float64)

        image_2d = np.array(
            [(landmarks[i][0], landmarks[i][1]) for i in self.HEAD_POSE_POINTS],
            dtype=np.float64
        )

        focal   = w                          # Approximate focal length
        center  = (w / 2, h / 2)
        cam_mat = np.array(
            [[focal, 0,      center[0]],
             [0,     focal,  center[1]],
             [0,     0,      1        ]],
            dtype=np.float64
        )
        dist_coeffs = np.zeros((4, 1))

        success, rot_vec, trans_vec = cv2.solvePnP(
            model_3d, image_2d, cam_mat, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        return success, rot_vec, trans_vec, cam_mat, dist_coeffs

    def close(self):
        self.face_mesh.close()
