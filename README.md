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
