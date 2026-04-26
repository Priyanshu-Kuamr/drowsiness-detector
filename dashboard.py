import csv
import os
import time
import threading

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS

import config

socketio = SocketIO(cors_allowed_origins="*", async_mode="eventlet")


def create_app(shared_state: dict) -> Flask:
    app = Flask(__name__, template_folder="templates",
                static_folder="static")
    app.config["SECRET_KEY"] = "drowsiness-detector-secret"
    CORS(app)
    socketio.init_app(app)

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/state")
    def api_state():
        return jsonify(shared_state)

    @app.route("/api/history")
    def api_history():
        """Return last 100 rows from the session CSV for the analytics page."""
        rows = []
        if os.path.exists(config.HISTORY_LOG_FILE):
            with open(config.HISTORY_LOG_FILE, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)[-100:]
        return jsonify(rows)

    # ── SocketIO push loop ────────────────────────────────────────────────────

    def push_loop():
        while True:
            if shared_state:
                socketio.emit("state_update", shared_state)
            time.sleep(0.1)   # ~10 Hz push rate

    t = threading.Thread(target=push_loop, daemon=True)
    t.start()

    return app
