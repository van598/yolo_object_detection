import cv2
import time
from ultralytics import YOLO


KNOWN_WIDTH = 45      
FOCAL_LENGTH = 700    


model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Camera not opened")

def estimate_distance(known_width, focal_length, pixel_width):
    if pixel_width <= 0:
        return None
    return (known_width * focal_length) / pixel_width

prev_time = time.perf_counter()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame, conf=0.35, device="cpu", verbose=False)

    closest_distance = None

    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pixel_width = x2 - x1

                distance = estimate_distance(
                    KNOWN_WIDTH, FOCAL_LENGTH, pixel_width
                )

                if distance:
                    if closest_distance is None or distance < closest_distance:
                        closest_distance = distance

    curr_time = time.perf_counter()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time


    frame = results[0].plot()

    text = f"FPS: {int(fps)}"
    if closest_distance:
        text += f" | Distance: {int(closest_distance)} cm"

    cv2.putText(frame, text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Webcam", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

