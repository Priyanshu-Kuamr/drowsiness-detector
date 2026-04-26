import argparse
import threading
import time
import cv2

import config
from alert import (
    SoundAlert, DangerClipRecorder,
    draw_hud, draw_landmarks
)
from utils import (
    FaceDetector,
    compute_avg_ear, BlinkTracker,
    compute_mar, YawnTracker, HeadPoseTracker,
    rotation_vector_to_euler,
    AlertnessScore, SessionLogger,
)

_shared_state: dict = {}

def run_detection(share_state: bool = False):

    # FLAGS
    drowsy_active = False
    yawn_active = False
    head_active = False
    eye_closed_start = None

    # SESSION TRACKING
    min_score = 100
    last_session = {}

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    detector       = FaceDetector()
    blink_tracker  = BlinkTracker(config.EAR_THRESHOLD, config.BLINK_CONSEC_FRAMES, fps=30)
    yawn_tracker   = YawnTracker(config.MAR_THRESHOLD, config.YAWN_CONSEC_FRAMES)
    head_tracker   = HeadPoseTracker(config.HEAD_PITCH_THRESHOLD, config.HEAD_YAW_THRESHOLD, config.HEAD_CONSEC_FRAMES)

    score_engine   = AlertnessScore(
        decay_rate=config.SCORE_DECAY_RATE,
        recover_rate=config.SCORE_RECOVER_RATE,
        yawn_penalty=config.SCORE_YAWN_PENALTY,
        head_penalty=config.SCORE_HEAD_PENALTY
    )

    logger         = SessionLogger(config.HISTORY_LOG_FILE)
    sound_alert    = SoundAlert(config.ALERT_SOUND_FILE, config.ALERT_COOLDOWN_SEC)
    clip_recorder  = DangerClipRecorder(fps=30)

    no_face_counter = 0
    frame_count = 0
    landmarks = None

    alert_msg = ""
    alert_until = 0.0

    session_stats = {
        "drowsy_events": 0,
        "yawn_events": 0,
        "distraction_events": 0,
    }

    print("[INFO] Detection started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 2 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = detector.find_landmarks(frame_rgb)

        ear, mar = 0.0, 0.0
        pitch, yaw = 0.0, 0.0

        if landmarks:

            no_face_counter = 0

            # ── EAR ──
            left_pts  = detector.get_eye_points(landmarks, "left")
            right_pts = detector.get_eye_points(landmarks, "right")
            ear = compute_avg_ear(left_pts, right_pts)

            # DROWSINESS (TIME-BASED)
            if ear < config.EAR_THRESHOLD:
                if eye_closed_start is None:
                    eye_closed_start = time.time()

                closed_duration = time.time() - eye_closed_start

                if closed_duration > 2.0 and not drowsy_active:
                    alert_msg = "🚨 SEVERE DROWSINESS!"
                    alert_until = time.time() + 3

                    sound_alert.play()

                    clip_recorder.save(reason="drowsy", score=score_engine.score)

                    session_stats["drowsy_events"] += 1
                    score_engine._score -= 15

                    drowsy_active = True

            else:
                eye_closed_start = None
                drowsy_active = False

            blink_tracker.update(ear)

            # ── YAWN ──
            mouth_pts = detector.get_mouth_points(landmarks)
            mar = compute_mar(mouth_pts)

            new_yawn = yawn_tracker.update(mar)

            if new_yawn and not yawn_active:
                alert_msg = "😮 YAWN DETECTED"
                alert_until = time.time() + 2

                session_stats["yawn_events"] += 1
                score_engine._score -= 5

                yawn_active = True

            if not yawn_tracker.is_yawning:
                yawn_active = False

            # ── HEAD ──
            success, rot_vec, *_ = detector.get_head_pose_points(landmarks, frame.shape)

            if success:
                pitch, yaw, _ = rotation_vector_to_euler(rot_vec)
                new_event = head_tracker.update(pitch, yaw)

                if new_event and not head_active:
                    alert_msg = "👀 DISTRACTION ALERT!"
                    alert_until = time.time() + 2

                    sound_alert.play()
                    session_stats["distraction_events"] += 1
                    score_engine._score -= 3

                    head_active = True

            if not head_tracker.is_distracted:
                head_active = False

            frame = draw_landmarks(frame, left_pts, right_pts, mouth_pts)

        else:
            no_face_counter += 1

            if no_face_counter > 30:
                alert_msg = "⚠ Face not detected!"
                alert_until = time.time() + 2

        # SCORE UPDATE (RECOVERY ONLY)
        score = score_engine.update(
            ear_low=False,
            is_yawning=False,
            head_off=False
        )

        min_score = min(min_score, score)

        logger.log(
            score=score,
            band=score_engine.band,
            ear=ear, mar=mar,
            pitch=pitch, yaw=yaw,
            bpm=blink_tracker.blinks_per_minute,
            yawns=yawn_tracker.total_yawns
        )

        if share_state:
            _shared_state.update({
                "score": score,
                "band": score_engine.band,
                "ear": round(ear, 4),
                "mar": round(mar, 4),
                "pitch": round(pitch, 2),
                "yaw": round(yaw, 2),
                "bpm": blink_tracker.blinks_per_minute,
                "yawns": yawn_tracker.total_yawns,
                "blinks": blink_tracker.total_blinks,
                "history": score_engine.history_for_chart(),
                "alert": alert_msg if time.time() < alert_until else "",
            })

        clip_recorder.push(frame)

        frame = draw_hud(
            frame,
            ear=ear, mar=mar,
            pitch=pitch, yaw=yaw,
            score=score,
            band=score_engine.band,
            band_color=score_engine.band_color,
            blinks_per_min=blink_tracker.blinks_per_minute,
            total_yawns=yawn_tracker.total_yawns,
            alert_msg=alert_msg if time.time() < alert_until else ""
        )

        cv2.imshow("Driver Drowsiness Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    # FATIGUE SCORE
    fatigue_score = max(0, 100 - (
        session_stats["drowsy_events"] * 15 +
        session_stats["yawn_events"] * 5 +
        session_stats["distraction_events"] * 3
    ))

    # SAVE FINAL SESSION
    if share_state:
        _shared_state.update({
            "final": True,
            "final_score": score_engine.score,
            "min_score": min_score,
            "fatigue_score": fatigue_score,
            "session_stats": session_stats
        })

    print("\n─── Session Summary ───")
    print(session_stats)
    print(f"Final Score (Instant): {score_engine.score:.1f}")
    print(f"Final Score (Session Worst): {min_score:.1f}")
    print(f"Final Score (Fatigue Index): {fatigue_score:.1f}")
    print("[INFO] Detection stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-dashboard", action="store_true")
    args = parser.parse_args()

    if args.no_dashboard:
        run_detection(False)
    else:
        from dashboard import create_app, socketio

        app = create_app(_shared_state)

        flask_thread = threading.Thread(
            target=lambda: socketio.run(
                app,
                host=config.FLASK_HOST,
                port=config.FLASK_PORT,
                allow_unsafe_werkzeug=True,
                use_reloader=False
            ),
            daemon=True
        )
        flask_thread.start()

        run_detection(True)

        while True:
            time.sleep(1)