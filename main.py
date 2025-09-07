import cv2
import mediapipe as mp
import time
import math
import winsound
import csv
import os
from datetime import datetime

# Initialize face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]  # upper lip, lower lip, left, right
NOSE_TIP = 1
CHIN = 152
LEFT_EAR = 234
RIGHT_EAR = 454

# EAR & MAR thresholds
EAR_THRESHOLD = 0.25
EYE_CLOSED_DURATION = 2
MAR_THRESHOLD = 0.7
YAWN_DURATION = 2

# Logging setup
log_filename = "drowsiness_log.csv"
if not os.path.exists(log_filename):
    with open(log_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Event"])

# Helper functions
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def get_coords(landmarks, indices, w, h):
    return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]

def eye_aspect_ratio(eye_coords):
    return (distance(eye_coords[1], eye_coords[5]) + distance(eye_coords[2], eye_coords[4])) / (2 * distance(eye_coords[0], eye_coords[3]))

def mouth_aspect_ratio(mouth_coords):
    return distance(mouth_coords[0], mouth_coords[1]) / distance(mouth_coords[2], mouth_coords[3])

def log_event(event):
    with open(log_filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event])

# Camera
cap = cv2.VideoCapture(0)

eye_closed_start_time = None
yawn_start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # EAR
            left_eye_coords = get_coords(landmarks, LEFT_EYE, w, h)
            right_eye_coords = get_coords(landmarks, RIGHT_EYE, w, h)
            ear = (eye_aspect_ratio(left_eye_coords) + eye_aspect_ratio(right_eye_coords)) / 2

            # MAR
            mouth_coords = get_coords(landmarks, MOUTH, w, h)
            mar = mouth_aspect_ratio(mouth_coords)

            # Head tilt estimation (simplified using nose and chin vertical distance)
            nose = (int(landmarks[NOSE_TIP].x * w), int(landmarks[NOSE_TIP].y * h))
            chin = (int(landmarks[CHIN].x * w), int(landmarks[CHIN].y * h))
            head_tilt_ratio = distance(nose, chin)

            # Display EAR and MAR
            cv2.putText(frame, f'EAR: {ear:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'MAR: {mar:.2f}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Drowsiness Detection
            if ear < EAR_THRESHOLD:
                if eye_closed_start_time is None:
                    eye_closed_start_time = time.time()
                elif time.time() - eye_closed_start_time > EYE_CLOSED_DURATION:
                    cv2.putText(frame, 'DROWSINESS ALERT!', (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    winsound.Beep(1000, 1000)
                    log_event("Eyes Closed Too Long")
                    cv2.imwrite(f"screenshot_{datetime.now().strftime('%H-%M-%S')}.jpg", frame)
            else:
                eye_closed_start_time = None

            # Yawn Detection
            if mar > MAR_THRESHOLD:
                if yawn_start_time is None:
                    yawn_start_time = time.time()
                elif time.time() - yawn_start_time > YAWN_DURATION:
                    cv2.putText(frame, 'YAWNING DETECTED!', (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
                    winsound.Beep(800, 800)
                    log_event("Yawn Detected")
            else:
                yawn_start_time = None

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
