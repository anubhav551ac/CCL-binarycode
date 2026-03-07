"""
camera.py — GhostGrid processing core
Importable module. Run standalone with:  python camera.py
Imported by demo.py for direct in-process use (no file I/O middleman).
"""

import cv2
import numpy as np
from ultralytics import YOLOWorld
import threading
import time
import requests
import sqlite3
import os
from datetime import datetime, date
from flask import Flask, Response

# ── CONFIG ───────────────────────────────────────────────────────────────────
CAMERA_IP   = os.environ.get("GHOSTGRID_CAM_IP",        "192.168.137.205")
CAMERA_PORT = os.environ.get("GHOSTGRID_CAM_PORT",       "8080")
CAMERA_URL  = f"http://{CAMERA_IP}:{CAMERA_PORT}/video"
PTZ_URL     = f"http://{CAMERA_IP}:{CAMERA_PORT}/ptz"
DB_PATH     = os.environ.get("GHOSTGRID_DB",             "energy_data.db")
MJPEG_PORT  = int(os.environ.get("GHOSTGRID_MJPEG_PORT", "5050"))
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CLASSES = [
    "person", "laptop", "monitor", "television", "desk lamp",
    "computer mouse", "keyboard", "power strip", "electric fan",
    "air conditioner", "cell phone", "printer", "open window",
    "wall light", "flat rectangular tablet with white edges", "hand",
]
NONELECTRONIC = {"person", "open window", "hand"}
GRACE_PERIOD       = 5.0
# Minimum consecutive frames a detection must persist before it's treated as real.
# At ~10 FPS this is ~0.5s. Eliminates single-frame YOLO misfire ghosts.
MIN_CONFIRM_FRAMES = 5
# Frames a detection can be missing before it's considered gone (avoids flickering out)
MAX_MISSING_FRAMES = 3


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot connect to camera at {src}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                print("[camera] Lost connection to camera")
                self.stopped = True
            else:
                self.frame = frame

    def read(self):
        return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()


def set_zoom(val):
    def _req():
        try:
            requests.get(f"{PTZ_URL}?zoom={val}", timeout=0.2)
        except Exception:
            pass
    threading.Thread(target=_req, daemon=True).start()


def get_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1    = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2    = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter)


def is_overlapping(box1, box2):
    return (max(box1[0], box2[0]) < min(box1[2], box2[2]) and
            max(box1[1], box2[1]) < min(box1[3], box2[3]))


def determine_state(roi, ambient_brightness):
    if roi is None or roi.size == 0:
        return False, 0, 0
    gray           = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    contrast       = np.std(gray)
    hsv            = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    avg_saturation = np.mean(hsv[:, :, 1])

    is_bright        = avg_brightness > (ambient_brightness * 1.25)
    is_vibrant       = avg_saturation > 45
    is_high_contrast = contrast > 40

    score = sum([is_bright, is_vibrant, is_high_contrast])
    return score >= 2, avg_brightness, contrast


def _get_power(cursor, dev_id):
    prefix = dev_id.rsplit('_', 1)[0] if '_' in dev_id else dev_id
    cursor.execute('SELECT power_watts FROM devices WHERE device_type = ?', (prefix,))
    row = cursor.fetchone()
    return row[0] if row else None


# ═══════════════════════════════════════════════════════════════════════════
#  DATABASE SETUP
# ═══════════════════════════════════════════════════════════════════════════

def init_db(path=DB_PATH):
    conn   = sqlite3.connect(path, check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS devices (
        device_type TEXT PRIMARY KEY,
        power_watts REAL
    )''')

    cursor.executemany(
        'INSERT OR IGNORE INTO devices (device_type, power_watts) VALUES (?, ?)',
        [
            ('laptop', 100), ('monitor', 50), ('television', 150), ('desk lamp', 20),
            ('computer mouse', 5), ('keyboard', 10), ('power strip', 5),
            ('electric fan', 50), ('air conditioner', 1000), ('cell phone', 10),
            ('printer', 100), ('wall light', 20), ('screen with white border', 50),
            ('flat rectangular tablet with white edges', 50),
        ]
    )

    # waste_logs: one row per device per "waste episode"
    # (written every time a person re-enters, resetting the waste counter)
    cursor.execute('''CREATE TABLE IF NOT EXISTS waste_logs (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        device_id           TEXT,
        total_time_wasted   REAL,
        total_time_used     REAL,
        date                TEXT,
        total_energy_wasted REAL,
        carbon_footprint    REAL
    )''')

    # daily_summary: one row per device per calendar day
    cursor.execute('''CREATE TABLE IF NOT EXISTS daily_summary (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        date                TEXT,
        device_id           TEXT,
        total_time_used     REAL,
        total_time_wasted   REAL,
        total_energy_used   REAL,
        total_energy_wasted REAL,
        carbon_footprint    REAL,
        UNIQUE(date, device_id)
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS DeviceReferenceTable (
        raw_prefix   TEXT PRIMARY KEY,
        display_name TEXT
    )''')

    cursor.executemany(
        'INSERT OR IGNORE INTO DeviceReferenceTable (raw_prefix, display_name) VALUES (?, ?)',
        [
            ('laptop',                                   'Laptop'),
            ('monitor',                                  'Monitor'),
            ('television',                               'Television'),
            ('desk lamp',                                'Desk Lamp'),
            ('computer mouse',                           'Computer Mouse'),
            ('keyboard',                                 'Keyboard'),
            ('power strip',                              'Power Strip'),
            ('electric fan',                             'Electric Fan'),
            ('air conditioner',                          'Air Conditioner'),
            ('cell phone',                               'Cell Phone'),
            ('printer',                                  'Printer'),
            ('wall light',                               'Wall Light'),
            ('flat rectangular tablet with white edges', 'Tablet / Screen'),
            ('screen with white border',                 'Screen'),
        ]
    )
    conn.commit()
    return conn


# ═══════════════════════════════════════════════════════════════════════════
#  CAMERA PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════

class CameraProcessor:
    """
    Spin up once via start().
    Read .latest_frame (BGR ndarray) and .state (dict) from any thread.
    """

    def __init__(self, db_conn, camera_url=CAMERA_URL):
        self.db_conn    = db_conn
        self.camera_url = camera_url
        self._lock      = threading.Lock()

        # ── live state (read by demo.py) ──────────────────────────────────
        self.latest_frame = None
        self.state = {
            "running":        False,
            "person_present": False,
            "waste_active":   False,
            # keyed by device prefix (e.g. "laptop"), value is latest dict
            "detections":     {},
            # cumulative waste per unique_key since last person-reset
            "current_waste":  {},
            # cumulative totals across the whole session (for display)
            "total_waste":    {},
            "total_usage":    {},
            "fps":            0.0,
        }

        self._stopped        = False
        self._vs             = None
        self._model          = None
        self._device_timers  = {}
        self._last_loop_t    = time.time()
        self._fps_counter    = 0
        self._fps_timer      = time.time()

        # Flicker filter: track consecutive-seen and consecutive-missing frame counts
        # per unique_key.  Format: {unique_key: {"seen": int, "missing": int, "confirmed": bool}}
        self._presence        = {}

        # tracks whether person was present last frame (edge detection)
        self._person_was_present = False

        # daily summary flush — track last flush date
        self._last_daily_date = date.today()
        threading.Thread(target=self._daily_flush_watcher, daemon=True).start()

    # ── public ───────────────────────────────────────────────────────────

    def start(self):
        threading.Thread(target=self._init_and_run,      daemon=True).start()
        threading.Thread(target=self._start_mjpeg_server, daemon=True).start()
        return self

    def stop(self):
        self._stopped = True

    def flush_final_to_db(self):
        """Call on shutdown — writes any remaining waste and daily summary."""
        self._flush_waste_to_db("shutdown")
        self._flush_daily_summary()

    # ── MJPEG server ──────────────────────────────────────────────────────

    def _start_mjpeg_server(self):
        app = Flask(__name__)

        @app.route("/feed")
        def feed():
            def generate():
                while True:
                    with self._lock:
                        frame = self.latest_frame
                    if frame is None:
                        time.sleep(0.05)
                        continue
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if not ok:
                        continue
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" +
                           buf.tobytes() + b"\r\n")
            return Response(generate(),
                            mimetype="multipart/x-mixed-replace; boundary=frame")

        import logging
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        app.run(host="0.0.0.0", port=MJPEG_PORT, threaded=True)

    # ── DB flush helpers ──────────────────────────────────────────────────

    def _flush_waste_to_db(self, reason="person_returned"):
        """
        Write current waste window per device to waste_logs, then reset
        current_waste counters. Called when a person is detected (re-entry)
        or on shutdown.
        """
        cursor       = self.db_conn.cursor()
        session_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self._lock:
            cw = dict(self.state["current_waste"])
            tu = dict(self.state["total_usage"])

        for dev_id, t_waste in cw.items():
            if t_waste < 1.0:
                continue
            power = _get_power(cursor, dev_id)
            if power is None:
                continue

            t_usage             = tu.get(dev_id, 0)
            total_energy_wasted = power * (t_waste / 3600)
            carbon              = total_energy_wasted * 0.0004

            cursor.execute(
                '''INSERT INTO waste_logs
                   (device_id, total_time_wasted, total_time_used, date,
                    total_energy_wasted, carbon_footprint)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (dev_id, t_waste, t_usage, session_date,
                 total_energy_wasted, carbon)
            )
            print(f"[camera] waste_log ← {dev_id}: {t_waste:.1f}s wasted ({reason})")

        self.db_conn.commit()

        # reset the current-waste window; keep total_waste intact for display
        with self._lock:
            self.state["current_waste"] = {}

    def _flush_daily_summary(self):
        """
        Upsert into daily_summary for today using session totals.
        Uses INSERT OR REPLACE with accumulated values so multiple flushes
        within a day keep adding up.
        """
        cursor    = self.db_conn.cursor()
        today_str = date.today().isoformat()

        with self._lock:
            tw = dict(self.state["total_waste"])
            tu = dict(self.state["total_usage"])

        all_keys = set(list(tw.keys()) + list(tu.keys()))
        for dev_id in all_keys:
            t_waste = tw.get(dev_id, 0)
            t_usage = tu.get(dev_id, 0)
            if (t_waste + t_usage) < 1.0:
                continue

            power = _get_power(cursor, dev_id)
            if power is None:
                continue

            energy_used   = power * (t_usage  / 3600)
            energy_wasted = power * (t_waste  / 3600)
            carbon        = (energy_used + energy_wasted) * 0.0004

            # read existing row to accumulate correctly across multiple flushes
            cursor.execute(
                'SELECT total_time_used, total_time_wasted, total_energy_used, '
                'total_energy_wasted, carbon_footprint '
                'FROM daily_summary WHERE date=? AND device_id=?',
                (today_str, dev_id)
            )
            existing = cursor.fetchone()
            if existing:
                # replace with fresh session totals (they already accumulate
                # from t=0, so no need to add — just overwrite)
                cursor.execute(
                    '''INSERT OR REPLACE INTO daily_summary
                       (date, device_id, total_time_used, total_time_wasted,
                        total_energy_used, total_energy_wasted, carbon_footprint)
                       VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (today_str, dev_id, t_usage, t_waste,
                     energy_used, energy_wasted, carbon)
                )
            else:
                cursor.execute(
                    '''INSERT INTO daily_summary
                       (date, device_id, total_time_used, total_time_wasted,
                        total_energy_used, total_energy_wasted, carbon_footprint)
                       VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (today_str, dev_id, t_usage, t_waste,
                     energy_used, energy_wasted, carbon)
                )

        self.db_conn.commit()
        print(f"[camera] daily_summary flushed for {today_str}")

    def _daily_flush_watcher(self):
        """Background thread — flushes daily summary at midnight."""
        while not self._stopped:
            time.sleep(60)
            today = date.today()
            if today != self._last_daily_date:
                print("[camera] New day — flushing daily summary")
                self._flush_daily_summary()
                self._last_daily_date = today

    # ── init + run loop ───────────────────────────────────────────────────

    def _init_and_run(self):
        print("[camera] Loading YOLO model...")
        self._model = YOLOWorld('yolov8s-world.pt').to('cuda')
        self._model.set_classes(CUSTOM_CLASSES)
        dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self._model.predict(dummy, device='cuda', verbose=False)
        print("[camera] Model ready")

        try:
            self._vs = VideoStream(self.camera_url).start()
        except RuntimeError as e:
            print(f"[camera] {e}")
            return

        with self._lock:
            self.state["running"] = True

        self._last_loop_t = time.time()
        self._run_loop()

    def _run_loop(self):
        while not self._stopped and not self._vs.stopped:
            frame = self._vs.read()
            if frame is None:
                continue

            current_time = time.time()
            delta_time   = current_time - self._last_loop_t
            self._last_loop_t = current_time

            frame = cv2.resize(frame, (1920, 1080))
            ambient_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            results = self._model.track(
                frame, conf=0.3, iou=0.5, device='cuda', persist=True, verbose=False
            )

            detected_people       = []
            all_detections        = []
            energy_waste_detected = False
            live_detections       = {}

            if results and len(results[0].boxes) > 0:
                ids = (results[0].boxes.id.cpu().numpy().astype(int)
                       if results[0].boxes.id is not None
                       else [-1] * len(results[0].boxes))

                for box_data, track_id in zip(results[0].boxes, ids):
                    cls_id = int(box_data.cls[0])
                    label  = CUSTOM_CLASSES[cls_id]
                    coords = box_data.xyxy[0].cpu().numpy().astype(int)

                    if label == "person":
                        detected_people.append(coords)
                        cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
                        cv2.putText(frame, f"PERSON ID:{track_id}",
                                    (coords[0], coords[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        unique_key = f"{label}_{track_id}" if track_id != -1 else label
                        all_detections.append((unique_key, label, coords))

                # dedup
                filtered = []
                for unique_key, label, coords in all_detections:
                    is_dup = any(label == fl and get_iou(coords, fc) > 0.7
                                 for _, fl, fc in filtered)
                    if not is_dup:
                        filtered.append((unique_key, label, coords))

                # ── detection per-device prefix (for "This Session" display) ──
                # We collapse track IDs here so we show ONE entry per device type
                live_detections = {}   # keyed by label prefix

                # ── presence / flicker filter ──────────────────────────────
                seen_keys = {uk for uk, _, _ in filtered}
                for uk in list(self._presence.keys()):
                    if uk not in seen_keys:
                        self._presence[uk]["missing"] += 1
                        self._presence[uk]["seen"]     = 0
                        if self._presence[uk]["missing"] > MAX_MISSING_FRAMES:
                            del self._presence[uk]   # truly gone

                for unique_key, label, coords in filtered:
                    p = self._presence.setdefault(unique_key,
                                                  {"seen": 0, "missing": 0, "confirmed": False})
                    p["seen"]    += 1
                    p["missing"]  = 0
                    if p["seen"] >= MIN_CONFIRM_FRAMES:
                        p["confirmed"] = True
                # ───────────────────────────────────────────────────────────────

                for unique_key, label, coords in filtered:
                    if label in NONELECTRONIC:
                        cv2.rectangle(frame,
                                      (coords[0], coords[1]), (coords[2], coords[3]),
                                      (255, 255, 255), 1)
                        cv2.putText(frame, label, (coords[0], coords[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        continue

                    # skip ghost detections that haven't confirmed yet
                    if not self._presence.get(unique_key, {}).get("confirmed", False):
                        continue

                    roi    = frame[coords[1]:coords[3], coords[0]:coords[2]]
                    is_on, _, _ = determine_state(roi, ambient_brightness)
                    being_used  = any(is_overlapping(p, coords) for p in detected_people)

                    if is_on:
                        if being_used:
                            self.state["total_usage"][unique_key] = \
                                self.state["total_usage"].get(unique_key, 0) + delta_time
                            self._device_timers[unique_key] = current_time
                            color, status_tag = (0, 255, 0), "IN USE"
                        else:
                            self.state["total_waste"][unique_key] = \
                                self.state["total_waste"].get(unique_key, 0) + delta_time
                            self.state["current_waste"][unique_key] = \
                                self.state["current_waste"].get(unique_key, 0) + delta_time
                            last_active     = self._device_timers.get(unique_key, current_time)
                            time_unattended = current_time - last_active
                            if time_unattended < GRACE_PERIOD:
                                color, status_tag = (0, 255, 0), "IN USE (STABILIZING)"
                            else:
                                color, status_tag = (0, 0, 255), \
                                    f"WASTED: {self.state['total_waste'][unique_key]:.1f}s"
                                energy_waste_detected = True
                                url = "https://api.pushover.net/1/messages.json"
                                data = {
                                    "token": 'ai62ez85469ho238mv1rf5mgg662pn',
                                    "user": 'ug4i6e1oki9o4wrr4iq1nd7hhuxeye',
                                    "message": f"Device is unattended",
                                    "title": "GhostGrid Sentinel",
                                    "priority": 1,   
                                    "sound": "siren" 
                                }
                                response = requests.post(url, data=data)
                    else:
                        color, status_tag = (100, 100, 100), "OFF"

                    cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)
                    cv2.putText(frame, f"{unique_key} {status_tag}",
                                (coords[0], coords[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # collapse to one entry per label for the UI
                    # prefer the "worse" state: wasting > in use > off
                    prev = live_detections.get(label)
                    rank = {"IN USE": 1, "OFF": 0}
                    def _rank(d):
                        if d["is_on"] and not d["being_used"]: return 2
                        if d["is_on"] and d["being_used"]:     return 1
                        return 0
                    entry = {
                        "key":        unique_key,
                        "label":      label,
                        "status":     status_tag,
                        "is_on":      is_on,
                        "being_used": being_used,
                    }
                    if prev is None or _rank(entry) > _rank(prev):
                        live_detections[label] = entry

            # ── person re-entry edge: flush waste window to DB ──────────────
            person_now = len(detected_people) > 0
            if person_now and not self._person_was_present:
                # rising edge — person just walked in
                threading.Thread(
                    target=self._flush_waste_to_db,
                    args=("person_returned",),
                    daemon=True
                ).start()
            self._person_was_present = person_now

            # FPS
            self._fps_counter += 1
            if current_time - self._fps_timer >= 1.0:
                fps = self._fps_counter / (current_time - self._fps_timer)
                self._fps_timer   = current_time
                self._fps_counter = 0
            else:
                fps = self.state.get("fps", 0.0)

            with self._lock:
                self.latest_frame            = frame.copy()
                self.state["person_present"] = person_now
                self.state["waste_active"]   = energy_waste_detected
                self.state["detections"]     = live_detections   # dict keyed by label
                self.state["fps"]            = round(fps, 1)

        with self._lock:
            self.state["running"] = False
        print("[camera] Processing loop ended")


# ═══════════════════════════════════════════════════════════════════════════
#  STANDALONE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    db_conn   = init_db()
    processor = CameraProcessor(db_conn)
    processor.start()

    cv2.namedWindow("Vampire Power", cv2.WINDOW_NORMAL)
    current_zoom = 0

    print("[camera] Press Q to quit, R/= to zoom in, E to zoom out")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('='), ord('r')]:
            current_zoom = min(current_zoom + 10, 100); set_zoom(current_zoom)
        elif key == ord('e'):
            current_zoom = max(current_zoom - 10, 0);   set_zoom(current_zoom)

        with processor._lock:
            frame = processor.latest_frame
        if frame is not None:
            cv2.imshow("Vampire Power", frame)

    processor.stop()
    processor.flush_final_to_db()
    db_conn.close()
    cv2.destroyAllWindows()