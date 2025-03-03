# Virtual Keyboard Hand Tracking

This project enhances fingertip tracking accuracy for a virtual keyboard using Google MediaPipe and Leap Motion, designed for VR environments. It’s part of an ITRC research task to improve hand tracking precision beyond built-in VR or Leap Motion capabilities.

## Files
- **`hand_tracker.py`**: Core hand tracking with MediaPipe and Leap Motion fusion.
- **`virtual_keyboard.py`**: Virtual QWERTY keyboard layout and key press detection.
- **`utils.py`**: Helper functions (e.g., Kalman filter).
- **`requirements.txt`**: Project dependencies.

## Features
- Real-time hand tracking with MediaPipe and Leap Motion.
- Kalman filter-based fusion for precise fingertip tracking.
- Virtual keyboard overlay with key press detection.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/virtual-keyboard-hand-tracking.git
   cd virtual-keyboard-hand-tracking
   ```
## Set Up Conda Environment:
```bash
conda create -n mp python=3.9
conda activate mp
pip install -r requirements.txt
```
## Install Leap Motion SDK:
- ### Download from Ultraleap.

- ### Add Python bindings to your environment.

# Usage
## Run Hand Tracker:
```bash
python hand_tracker.py
```
- Displays fused hand tracking (replace cv2.imshow with VR rendering).

## Run Virtual Keyboard:
```bash
python virtual_keyboard.py
```
- Tests the keyboard standalone with MediaPipe.

## Combine (Edit hand_tracker.py to import virtual_keyboard):
```bash
#python
from virtual_keyboard import draw_keyboard, detect_key_press
# In main loop:
image = draw_keyboard(image)
if results.multi_hand_landmarks:
    pressed = detect_key_press(fused_landmarks, image, keyboard_layout)
    if pressed:
        print(f"Pressed: {pressed}")
```
# Customization
- Keyboard Layout: Edit keyboard_layout in virtual_keyboard.py.

- Fusion Weights: Adjust Kalman parameters in utils.py.

- VR Integration: Replace cv2 display with your VR SDK.

# Troubleshooting
- Camera Issues: Check /dev/video* permissions (sudo chmod 666 /dev/video0).

- Leap Motion: Ensure the service is running (leapd on Linux).

- Accuracy: Calibrate fusion offsets in fuse_fingertips.

# License
MIT License. See LICENSE for details.

---
```bash
### How to Use
1. **Save Files**: Place each file in your project directory (e.g., `~/muhiddin/virtual-keyboard/`).
2. **Test Individually**:
   - `python hand_tracker.py`: Verify tracking and fusion.
   - `python virtual_keyboard.py`: Test the keyboard standalone.
3. **Integrate**: Merge keyboard functionality into `hand_tracker.py` as shown in the README.
4. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial virtual keyboard hand tracking project"
   git remote add origin https://github.com/yourusername/virtual-keyboard-hand-tracking.git
   git push -u origin main
```
# Credits
- Online free open sourse data
- Generative AI
- Group of developers
