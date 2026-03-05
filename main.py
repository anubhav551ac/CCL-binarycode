import cv2
import torch
import ultralytics 
import YOLOWorld

def start_world_detection():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device >>> {device}")

    model = YOLOWorld('yolov8m-world.pt')
    model.to(device)

    custom_classes = [
        "monitor with content on screen", "computer monitor which is completely black and off",
        "glowing lamp", "lamp turned off",
        "spinning fan", "still fan",
        "person", "human face", "wall power outlet on", "Person smiling",
        "air conditioner with display on", "air conditioner turned off"
    ]

    model.set_classes(custom_classes)

    #Cold start the cams
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Camera is not accessible. Kaput")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break


        results = model.predict(frame, conf = 0.3, device = device, verbose = False)

        annotated_frame = results[0].plot()
        cv2.imshow('YOLO-World GPU Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_world_detection()