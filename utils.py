# utils.py
from filterpy.kalman import KalmanFilter
import numpy as np

def initialize_kalman():
    """Initialize a Kalman filter for 3D coordinates."""
    kf = KalmanFilter(dim_x=3, dim_z=3)
    kf.x = np.array([0., 0., 0.])  # Initial state
    kf.F = np.eye(3)  # State transition matrix
    kf.H = np.eye(3)  # Measurement function
    kf.P *= 1000.  # Covariance
    kf.R = 5  # Measurement noise
    kf.Q = 0.1  # Process noise
    return kf
