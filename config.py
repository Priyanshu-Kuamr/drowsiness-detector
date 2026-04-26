# ─── EAR (Eye Aspect Ratio) 
EAR_THRESHOLD       = 0.15   
EAR_CONSEC_FRAMES   = 12     
BLINK_CONSEC_FRAMES = 3      

# ─── MAR (Mouth Aspect Ratio — Yawn)
MAR_THRESHOLD       = 0.75   
YAWN_CONSEC_FRAMES  = 15    

# ─── Head Pose 
HEAD_PITCH_THRESHOLD = 20    
HEAD_YAW_THRESHOLD   = 30    
HEAD_CONSEC_FRAMES   = 25    

# ─── Alertness Score
SCORE_DECAY_RATE     = 2.0   
SCORE_RECOVER_RATE   = 0.1   
SCORE_YAWN_PENALTY   = 5     
SCORE_HEAD_PENALTY   = 0.3   

# ─── Alert 
ALERT_SOUND_FILE     = "assets/alert.wav"
ALERT_COOLDOWN_SEC   = 5    

# ─── Recording 
DANGER_CLIP_SECONDS  = 5     
OUTPUT_DIR           = "recordings"

# ─── Dashboard 
FLASK_HOST           = "127.0.0.1"
FLASK_PORT           = 5000
HISTORY_LOG_FILE     = "data/session_log.csv"

# ─── Camera 
CAMERA_INDEX         = 0
FRAME_WIDTH          = 640
FRAME_HEIGHT         = 480


# Confidence thresholds
CONFIDENCE_SEVERE  = 0.80
CONFIDENCE_WARNING = 0.60
