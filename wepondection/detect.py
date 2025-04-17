from ultralytics import YOLO
import cv2
import pygame
import time
import numpy as np

# Initialize Pygame mixer
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.wav")

# Load the trained model
model = YOLO("yolov8n.pt")  # Replace with your custom trained model if needed

# Open the webcam
cap = cv2.VideoCapture(0)

# Time tracker to avoid sound spam
last_alarm_time = 0
alarm_interval = 2  # seconds between alarm plays

def is_camera_blocked(frame, threshold=30):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness < threshold  # You can adjust the threshold if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detected = False
    current_time = time.time()

    # Check for camera obstruction
    if is_camera_blocked(frame):
        print("🛑 Camera appears to be blocked!")
        detected = True

    # Run detection only if not blocked
    else:
        results = model(frame, stream=True)

        for r in results:
            boxes = r.boxes
            names = r.names
            for box in boxes:
                cls_id = int(box.cls[0])  # class ID
                class_name = names[cls_id]
                print("Detected:", class_name)

                # Check specifically for 'knife'
                if "knife" in class_name.lower():
                    print("🔪 Knife Detected!")
                    detected = True
                    break
            annotated_frame = r.plot()

    # Play alarm if knife or obstruction is detected
    if detected and (current_time - last_alarm_time > alarm_interval):
        alarm_sound.play()
        last_alarm_time = current_time

    # Show the frame only if not blocked
    if not is_camera_blocked(frame):
        cv2.imshow("Knife Detection", annotated_frame)
    else:
        # Display black screen with warning
        black_frame = np.zeros_like(frame)
        cv2.putText(black_frame, "Camera Blocked!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 0, 255), 3)
        cv2.imshow("Knife Detection", black_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
