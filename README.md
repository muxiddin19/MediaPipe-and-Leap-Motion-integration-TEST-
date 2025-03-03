# MediaPipe-and-Leap-Motion-integration-TEST-
integrating MediaPipe and Leap Motion for a virtual keyboard with improved fingertip accuracy in a VR context.
- hand_tracker.py: Core hand tracking with MediaPipe, Leap Motion, and data fusion.

- virtual_keyboard.py: Virtual keyboard layout and key press detection.

- utils.py: Helper functions (e.g., Kalman filter initialization).

- requirements.txt: Updated dependencies.

- README.md: Updated documentation reflecting the new structure.

## 1. hand_tracker.py
### This file contains the main logic for hand tracking, Leap Motion integration, and data fusion.
``` bash
# hand_tracker.py
import cv2
import mediapipe as mp
import leap
import numpy as np
from utils import initialize_kalman  # Import Kalman filter utility

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Leap Motion setup
leap_controller = leap.Controller()

def get_vr_camera_feed(index=0):
    """Initialize VR camera feed (replace with VR SDK if needed)."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Failed to open camera at index {index}")
        exit(1)
    return cap

def get_leap_fingertips():
    """Extract fingertip positions from Leap Motion."""
    frame = leap_controller.frame()
    if not frame.hands:
        return None
    fingertips = {}
    for hand in frame.hands:
        hand_type = "Left" if hand.is_left else "Right"
        tips = []
        for finger in hand.fingers:
            tip = finger.bone(leap.Bone.TYPE_DISTAL).next_joint
            tips.append((tip.x, tip.y, tip.z))  # In mm
        fingertips[hand_type] = tips
    return fingertips

def fuse_fingertips(mp_landmarks, leap_tips, image_width, image_height, kfs):
    """Fuse MediaPipe and Leap Motion fingertip data with Kalman filtering."""
    if not leap_tips or not mp_landmarks:
        return mp_landmarks
    fused = mp_landmarks
    leap_fingers = list(leap_tips.values())[0]  # Assume one hand for simplicity
    for i, (kf, leap_tip) in enumerate(zip(kfs, leap_fingers)):
        mp_tip = mp_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP + i * 4]
        # Convert Leap mm to normalized coords (calibrate these offsets)
        leap_x = (leap_tip[0] + 500) / 1000
        leap_y = (leap_tip[1] + 500) / 1000
        leap_z = leap_tip[2] / 1000
        # Fuse with Kalman filter
        z = np.array([(mp_tip.x + leap_x) / 2, (mp_tip.y + leap_y) / 2, leap_z])
        kf.predict()
        kf.update(z)
        fused.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP + i * 4].x, \
        fused.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP + i * 4].y, \
        fused.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP + i * 4].z = kf.x
    return fused

def main():
    """Main hand tracking loop with fusion."""
    cap = get_vr_camera_feed(0)  # Adjust for VR camera
    kfs = [initialize_kalman() for _ in range(5)]  # Kalman filters for 5 fingertips

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                break

            # Process with MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Process with Leap Motion and fuse
            leap_data = get_leap_fingertips()
            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if leap_data:
                        fused_landmarks = fuse_fingertips(hand_landmarks, leap_data, frame.shape[1], frame.shape[0], kfs)
                        mp_drawing.draw_landmarks(
                            image, fused_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                    else:
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )

            # Display (replace with VR rendering later)
            cv2.imshow('Hand Tracking', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

## 2. virtual_keyboard.py
### This file defines the virtual keyboard layout and key press detection logic.
``` bash
# virtual_keyboard.py
import cv2
import mediapipe as mp

# MediaPipe hands for landmark indices
mp_hands = mp.solutions.hands

# Simple QWERTY layout (x, y coordinates in pixels)
keyboard_layout = {
    'q': (100, 50), 'w': (150, 50), 'e': (200, 50), 'r': (250, 50), 't': (300, 50),
    'y': (350, 50), 'u': (400, 50), 'i': (450, 50), 'o': (500, 50), 'p': (550, 50),
    'a': (125, 100), 's': (175, 100), 'd': (225, 100), 'f': (275, 100), 'g': (325, 100),
    'h': (375, 100), 'j': (425, 100), 'k': (475, 100), 'l': (525, 100),
    'z': (150, 150), 'x': (200, 150), 'c': (250, 150), 'v': (300, 150), 'b': (350, 150),
    'n': (400, 150), 'm': (450, 150)
}

def draw_keyboard(image):
    """Draw the virtual keyboard on the image."""
    for key, (x, y) in keyboard_layout.items():
        cv2.rectangle(image, (x-15, y-15), (x+15, y+15), (255, 255, 255), 1)
        cv2.putText(image, key, (x-5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image

def detect_key_press(fused_landmarks, image, layout):
    """Detect and highlight key presses based on fingertip positions."""
    pressed_keys = []
    for idx in [4, 8, 12, 16, 20]:  # Thumb, Index, Middle, Ring, Pinky tips
        x = int(fused_landmarks.landmark[idx].x * image.shape[1])
        y = int(fused_landmarks.landmark[idx].y * image.shape[0])
        for key, (kx, ky) in layout.items():
            if abs(x - kx) < 20 and abs(y - ky) < 20:  # 20px threshold
                pressed_keys.append(key)
                cv2.circle(image, (kx, ky), 10, (0, 255, 0), -1)  # Highlight pressed key
    return pressed_keys

def main():
    """Test the virtual keyboard standalone."""
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = draw_keyboard(image)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    pressed = detect_key_press(hand_landmarks, image, keyboard_layout)
                    if pressed:
                        print(f"Pressed: {pressed}")
            cv2.imshow('Virtual Keyboard', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```
