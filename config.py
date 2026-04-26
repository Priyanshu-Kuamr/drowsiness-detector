# ─── EAR (Eye Aspect Ratio) ───────────────────────────────────────────────────
EAR_THRESHOLD       = 0.15   # Below this → eyes considered closed
EAR_CONSEC_FRAMES   = 12     # Frames eyes must stay closed to trigger alert
BLINK_CONSEC_FRAMES = 3      # Short blink window (< this = just a blink)

# ─── MAR (Mouth Aspect Ratio — Yawn) ─────────────────────────────────────────
MAR_THRESHOLD       = 0.75   # Above this → yawn detected
YAWN_CONSEC_FRAMES  = 15     # Frames mouth must stay open to count as yawn

# ─── Head Pose ────────────────────────────────────────────────────────────────
HEAD_PITCH_THRESHOLD = 20    # Degrees — looking down
HEAD_YAW_THRESHOLD   = 30    # Degrees — looking sideways
HEAD_CONSEC_FRAMES   = 25    # Frames of distracted pose before alert

# ─── Alertness Score ──────────────────────────────────────────────────────────
SCORE_DECAY_RATE     = 2.0   # Points lost per drowsy frame
SCORE_RECOVER_RATE   = 0.1   # Points gained per normal frame
SCORE_YAWN_PENALTY   = 5     # Deducted per yawn
SCORE_HEAD_PENALTY   = 0.3   # Per off-axis frame

# ─── Alert ────────────────────────────────────────────────────────────────────
ALERT_SOUND_FILE     = "assets/alert.wav"
ALERT_COOLDOWN_SEC   = 5     # Seconds between repeated alerts

# ─── Recording ────────────────────────────────────────────────────────────────
DANGER_CLIP_SECONDS  = 5     # Seconds to save around danger event
OUTPUT_DIR           = "recordings"

# ─── Dashboard / Flask ────────────────────────────────────────────────────────
FLASK_HOST           = "127.0.0.1"
FLASK_PORT           = 5000
HISTORY_LOG_FILE     = "data/session_log.csv"

# ─── Camera ───────────────────────────────────────────────────────────────────
CAMERA_INDEX         = 0
FRAME_WIDTH          = 640
FRAME_HEIGHT         = 480


# Confidence thresholds
CONFIDENCE_SEVERE  = 0.80
CONFIDENCE_WARNING = 0.60