# Eye-Closure-Detection
Eye Closure Detection System for Driver Safety | Real-time monitoring using computer vision & ML to prevent drowsy driving. | Alerts users when prolonged eye closure is detected.

# Overview
The **Eye Closure Detection System** is a computer vision–based safety project that monitors drivers in real time to prevent accidents caused by drowsiness or fatigue.  
Using image processing and machine learning, the system detects prolonged eye closure and triggers alerts to keep the driver awake and focused.


# Features
- Real-time face and eye detection  
- Eye closure tracking using computer vision  
- Alert mechanism on prolonged drowsiness    
- Can be extended with IoT hardware (buzzers, alarms)


# Technologies Used
- Python – Core programming language
- OpenCV (cv2) – Real-time video capture, image processing, and visual overlays
- MediaPipe – Face mesh landmark detection (eyes, mouth, nose, chin)
- winsound – Generates audible alerts on Windows systems
- CSV – Logs drowsiness/yawn events with timestamps
- OS – File operations and log handling
- Datetime – For timestamping alert logs and screenshots
- Math & Time – For geometric calculations (EAR, MAR) and duration tracking


# Key Concepts
- Eye Aspect Ratio (EAR) – Detects prolonged eye closure
- Mouth Aspect Ratio (MAR) – Detects yawning and fatigue
- Face Mesh Landmarks – Accurate detection of eyes, mouth, and head position
- Real-Time Monitoring – Continuous video analysis via webcam
- Alert System – Beep sound, visual warning, log entry, and screenshots
- Real-Time Monitoring – Continuous video analysis via webcam
- Alert System – Beep sound, visual warning, log entry, and screenshots
