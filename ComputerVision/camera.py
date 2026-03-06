import cv2
import numpy as np
from ultralytics import YOLOWorld
import threading
import time
import requests
import sqlite3
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
CAMERA_IP   = os.environ.get("GHOSTGRID_CAM_IP", "192.168.1.72")
CAMERA_PORT = os.environ.get("GHOSTGRID_CAM_PORT", "8080")
CAMERA_URL  = f"http://{CAMERA_IP}:{CAMERA_PORT}/video"
PTZ_URL     = f"http://{CAMERA_IP}:{CAMERA_PORT}/ptz"
DB_PATH     = os.environ.get("GHOSTGRID_DB", "energy_data.db")
# ────────────────────────────────────────────────────────────────────────────

class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            print("Cannot connect to camera")
            exit()

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                print("Lost connection to camera")
                self.stopped = True
            else:
                self.frame = frame

    def read(self):
        return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()

def set_zoom(val):
    def target():
        try:
            requests.get(f"{PTZ_URL}?zoom={val}", timeout=0.2)
        except:
            pass
    threading.Thread(target=target).start()


def get_iou(box1, box2):
    x1 = max(box1[0], box2[0]);  y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]);  y2 = min(box1[3], box2[3])
    if x1 >= x2 or y1 >= y2: return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / (area1 + area2 - intersection)

def is_overlapping(box1, box2):
    return (max(box1[0], box2[0]) < min(box1[2], box2[2]) and
            max(box1[1], box2[1]) < min(box1[3], box2[3]))

def determine_state(roi, ambient_brightness):
    if roi is None or roi.size == 0: return False, 0, 0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    contrast       = np.std(gray)
    hsv            = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    avg_saturation = np.mean(hsv[:, :, 1])

    is_bright       = avg_brightness > (ambient_brightness * 1.25)
    is_vibrant      = avg_saturation > 45
    is_high_contrast = contrast > 40

    score = sum([is_bright, is_vibrant, is_high_contrast])
    return score >= 2, avg_brightness, contrast


# ── MODEL ───────────────────────────────────────────────────────────────────
model = YOLOWorld('yolov8s-world.pt').to('cuda')
custom_classes = ["person", "laptop", "monitor", "television", "desk lamp",
                  "computer mouse", "keyboard", "power strip", "electric fan",
                  "air conditioner", "cell phone", "printer", "open window",
                  "wall light", "flat rectangular tablet with white edges", "hand"]
nonelectronic_class = ["person", "open window", "hand"]
model.set_classes(custom_classes)

# ── DATABASE ─────────────────────────────────────────────────────────────────
conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS devices (
    device_type TEXT PRIMARY KEY,
    power_watts REAL
)''')

devices_data = [
    ('laptop', 100), ('monitor', 50), ('television', 150), ('desk lamp', 20),
    ('computer mouse', 5), ('keyboard', 10), ('power strip', 5),
    ('electric fan', 50), ('air conditioner', 1000), ('cell phone', 10),
    ('printer', 100), ('wall light', 20), ('screen with white border', 50),
]
cursor.executemany('INSERT OR IGNORE INTO devices (device_type, power_watts) VALUES (?, ?)', devices_data)

cursor.execute('''CREATE TABLE IF NOT EXISTS waste_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id     TEXT,
    total_time_wasted  REAL,
    total_time_used    REAL,
    date          TEXT,
    total_energy_wasted REAL,
    carbon_footprint   REAL
)''')

# Human-readable name lookup table
cursor.execute('''CREATE TABLE IF NOT EXISTS DeviceReferenceTable (
    raw_prefix   TEXT PRIMARY KEY,
    display_name TEXT
)''')

ref_data = [
    ('laptop',                                  'Laptop'),
    ('monitor',                                 'Monitor'),
    ('television',                              'Television'),
    ('desk lamp',                               'Desk Lamp'),
    ('computer mouse',                          'Computer Mouse'),
    ('keyboard',                                'Keyboard'),
    ('power strip',                             'Power Strip'),
    ('electric fan',                            'Electric Fan'),
    ('air conditioner',                         'Air Conditioner'),
    ('cell phone',                              'Cell Phone'),
    ('printer',                                 'Printer'),
    ('wall light',                              'Wall Light'),
    ('flat rectangular tablet with white edges','Tablet / Screen'),
    ('screen with white border',                'Screen'),
]
cursor.executemany('INSERT OR IGNORE INTO DeviceReferenceTable (raw_prefix, display_name) VALUES (?, ?)', ref_data)
conn.commit()

# ── RUNTIME STATE ─────────────────────────────────────────────────────────
dummy_frame  = np.zeros((1920, 1080, 3), dtype=np.uint8)
model.predict(dummy_frame, device='cuda', verbose=False)

last_seen      = time.time()
current_zoom   = 0
device_timers  = {}
grace_period   = 5.0
total_waste    = {}
total_usage    = {}
last_loop_time = time.time()

vs = VideoStream(CAMERA_URL).start()
cv2.namedWindow("Vampire Power", cv2.WINDOW_NORMAL)

while True:
    frame = vs.read()
    if frame is None: continue

    current_time = time.time()

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord('='), ord('r')]:
        current_zoom = min(current_zoom + 10, 100);  set_zoom(current_zoom)
    elif key == ord('e'):
        current_zoom = max(current_zoom - 10, 0);    set_zoom(current_zoom)

    frame      = cv2.resize(frame, (1920, 1080))
    delta_time = current_time - last_loop_time
    last_loop_time = current_time

    ambient_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    results = model.track(frame, conf=0.3, iou=0.5, device='cuda', persist=True, verbose=False)

    energy_waste_detected = False
    detected_people  = []
    all_detections   = []

    if results and len(results[0].boxes) > 0:
        ids = (results[0].boxes.id.cpu().numpy().astype(int)
               if results[0].boxes.id is not None
               else [-1] * len(results[0].boxes))

        for box_data, track_id in zip(results[0].boxes, ids):
            cls_id = int(box_data.cls[0])
            label  = custom_classes[cls_id]
            coords = box_data.xyxy[0].cpu().numpy().astype(int)

            if label == "person":
                detected_people.append(coords)
                cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"PERSON ID:{track_id}", (coords[0], coords[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                unique_key = f"{label}_{track_id}" if track_id != -1 else label
                all_detections.append((unique_key, label, coords))

        filtered_detections = []
        for unique_key, label, coords in all_detections:
            is_dup = any(label == fl and get_iou(coords, fc) > 0.7 for _, fl, fc in filtered_detections)
            if not is_dup:
                filtered_detections.append((unique_key, label, coords))

        for unique_key, label, coords in filtered_detections:
            if label in nonelectronic_class:
                cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (255, 255, 255), 1)
                cv2.putText(frame, label, (coords[0], coords[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                continue

            roi   = frame[coords[1]:coords[3], coords[0]:coords[2]]
            is_on, _, _ = determine_state(roi, ambient_brightness)
            being_used  = any(is_overlapping(p, coords) for p in detected_people)

            if is_on:
                if being_used:
                    total_usage[unique_key]  = total_usage.get(unique_key, 0) + delta_time
                    device_timers[unique_key] = current_time
                    color, status_tag = (0, 255, 0), "IN USE"
                else:
                    total_waste[unique_key]  = total_waste.get(unique_key, 0) + delta_time
                    last_active      = device_timers.get(unique_key, current_time)
                    time_unattended  = current_time - last_active
                    if time_unattended < grace_period:
                        color, status_tag = (0, 255, 0), "IN USE (STABILIZING)"
                    else:
                        color, status_tag = (0, 0, 255), f"WASTED: {total_waste[unique_key]:.1f}s"
                        energy_waste_detected = True
            else:
                color, status_tag = (100, 100, 100), "OFF"

            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)
            cv2.putText(frame, f"{unique_key} {status_tag}", (coords[0], coords[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Vampire Power", frame)

    # Write latest annotated frame for the dashboard to read
    frame_out = os.environ.get("GHOSTGRID_FRAME", "ghostgrid_frame.jpg")
    cv2.imwrite(frame_out, frame)

# ── FINAL REPORT + DB WRITE ───────────────────────────────────────────────
print("\n--- FINAL ENERGY & EFFICIENCY REPORT ---")
session_date = time.ctime()
for dev_id in set(list(total_waste.keys()) + list(total_usage.keys())):
    t_waste = total_waste.get(dev_id, 0)
    t_usage = total_usage.get(dev_id, 0)

    if (t_waste + t_usage) >= 2.0:
        label = dev_id.split('_')[0] if '_' in dev_id else dev_id
        cursor.execute('SELECT power_watts FROM devices WHERE device_type = ?', (label,))
        row = cursor.fetchone()

        if row:
            power               = row[0]
            total_energy_wasted = power * (round(t_waste) / 3600)
            carbon              = total_energy_wasted * 0.0004

            cursor.execute('''INSERT INTO waste_logs
                              (device_id, total_time_wasted, total_time_used, date, total_energy_wasted, carbon_footprint)
                              VALUES (?, ?, ?, ?, ?, ?)''',
                           (dev_id, t_waste, t_usage, session_date, total_energy_wasted, carbon))

            efficiency = (t_usage / (t_usage + t_waste)) * 100 if (t_usage + t_waste) > 0 else 0
            print(f"Device: {dev_id:15} | Used: {t_usage:5.1f}s | Wasted: {t_waste:5.1f}s | Efficiency: {efficiency:.1f}%")

conn.commit()
conn.close()
vs.stop()
cv2.destroyAllWindows()