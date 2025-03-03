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
