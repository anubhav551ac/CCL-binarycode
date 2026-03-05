import cv2
import numpy as np
from ultralytics import YOLOWorld
import threading

class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) #cap = capture(frame)
        
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
        self.cap.release() #let the frame fly into the vast void of the world of network

def determine_state(roi, ambient_brightness): #roi = region of interest
    if roi is None or roi.size == 0: return False, 0, 0
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    contrast = np.std(gray)
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) #hsv = hue saturation value
    avg_saturation = np.mean(hsv[:,:,1])

    #calibration
    is_bright = avg_brightness > (ambient_brightness * 1.25) #increase if in brighter
    is_vibrant = avg_saturation > 45 #increase if others colorful
    is_high_contrast = contrast > 40 #increase if screen has reflection

    score = sum([is_bright, is_vibrant, is_high_contrast])
    print(f"Obj: {label} | Bright: {int(avg_brightness)} | Contrast: {int(contrast)}") #remove later
    return score >= 2, avg_brightness, contrast

model = YOLOWorld('yolov8s-world.pt').to('cuda')
custom_classes = ["person", "laptop", "monitor", "television", "desk lamp",
                "computer mouse", "keyboard", "power strip", "electric fan",
                "air conditioner", "cell phone", "printer", "open window",
                "wall light", "tablet"] #the objects
model.set_classes(custom_classes)

#initialise
dummy_frame = np.zeros((1280, 720, 3), dtype=np.uint8)
model.predict(dummy_frame, device='cuda', verbose=False)

#ip camera stream
url = 'http://192.168.137.250:8080/video'
vs = VideoStream(url).start()

cv2.namedWindow("Vampire Power", cv2.WINDOW_NORMAL)

while True:
    frame = vs.read()
    if frame is None: continue

    frame = cv2.resize(frame, (1280, 720))
    temp_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ambient_brightness = np.mean(temp_gray)

    # Use .predict instead of .track for more immediate detections
    results = model.predict(frame, conf=0.3, iou=0.5, device='cuda', verbose=False)

    person_present = False
    energy_waste_detected = False

    if results and len(results[0].boxes) > 0:
        for box_data in results[0].boxes:
            cls_id = int(box_data.cls[0])
            conf = float(box_data.conf[0])

            label = custom_classes[cls_id]
            coords = box_data.xyxy[0].cpu().numpy().astype(int)

            if label == "person":
                person_present = True
                cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
                continue

            roi = frame[coords[1]:coords[3], coords[0]:coords[2]] #for non person objects
            is_on, _, _ = determine_state(roi, ambient_brightness)

            if is_on:
                color = (0, 0, 255)
            else:
                color = (100, 100, 100)
            if is_on:
                status_tag = "ON"
            else:
                status_tag = "OFF"

            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)
            cv2.putText(frame, f"{label} {status_tag}", (coords[0], coords[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if is_on:
                energy_waste_detected = True

    alert_color = (0, 255, 0)
    final_msg = "SYSTEM NORMAL"

    if energy_waste_detected and not person_present:
        alert_color = (0, 0, 255)
        final_msg = "!! ALERT: UNATTENDED WASTE !!"
    elif person_present:
        final_msg = "OCCUPIED - ENERGY AUTHORIZED"

    cv2.rectangle(frame, (0,0), (1280, 60), (0,0,0), -1)
    cv2.putText(frame, final_msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, alert_color, 3)

    cv2.imshow("Vampire Power", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

vs.stop()
cv2.destroyAllWindows()