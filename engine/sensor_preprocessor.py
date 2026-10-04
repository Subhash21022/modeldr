import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

@dataclass
class ProcessedSample:
    timestamp_s: float
    dt: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    lat: Optional[float] = None
    lon: Optional[float] = None
    gnss_speed_mps: Optional[float] = None
    gnss_accuracy_m: Optional[float] = None
    gnss_bearing_deg: Optional[float] = None

class SensorPreprocessor:
    """
    Standardizes streaming multi-modal telematics:
    - Unit normalization (g -> m/s^2, deg/s -> rad/s)
    - Physical spike / sensor clipping rejection
    - Ring-buffer management for 1.0-second (100 samples) neural network windows
    """
    def __init__(
        self,
        target_freq_hz: float = 100.0,
        window_size: int = 100,
        accel_in_g: bool = False,
        gyro_in_degs: bool = False,
        max_accel_mps2: float = 55.0,
        max_gyro_rads: float = 25.0,
    ):
        self.target_freq = target_freq_hz
        self.nominal_dt = 1.0 / target_freq_hz
        self.window_size = window_size
        self.accel_in_g = accel_in_g
        self.gyro_in_degs = gyro_in_degs
        self.max_accel = max_accel_mps2
        self.max_gyro = max_gyro_rads
        
        self.last_timestamp: Optional[float] = None
        self.last_valid_accel = np.array([0.0, 0.0, 9.81], dtype=np.float32)
        self.last_valid_gyro = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Ring buffer for 100-sample sequence [ax, ay, az, gx, gy, gz]
        self.buffer = deque(maxlen=window_size)
        
    def add_sample(
        self,
        timestamp_s: float,
        ax: float,
        ay: float,
        az: float,
        gx: float,
        gy: float,
        gz: float,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        gnss_speed_mps: Optional[float] = None,
        gnss_accuracy_m: Optional[float] = None,
        gnss_bearing_deg: Optional[float] = None,
    ) -> ProcessedSample:
        """
        Process a single streaming raw sample and append to ring buffer.
        """
        # 1. Delta time calculation with jitter clamping
        if self.last_timestamp is None:
            dt = self.nominal_dt
        else:
            raw_dt = timestamp_s - self.last_timestamp
            if 0.001 <= raw_dt <= 0.2:
                dt = raw_dt
            else:
                dt = self.nominal_dt
        self.last_timestamp = timestamp_s
        
        # 2. Unit conversion
        if self.accel_in_g:
            ax, ay, az = ax * 9.80665, ay * 9.80665, az * 9.80665
        if self.gyro_in_degs:
            gx, gy, gz = np.radians(gx), np.radians(gy), np.radians(gz)
            
        # 3. Spike and NaN rejection
        acc = np.array([ax, ay, az], dtype=np.float32)
        if np.any(np.isnan(acc)) or np.linalg.norm(acc) > self.max_accel:
            acc = self.last_valid_accel
        else:
            self.last_valid_accel = acc
            
        gyro = np.array([gx, gy, gz], dtype=np.float32)
        if np.any(np.isnan(gyro)) or np.linalg.norm(gyro) > self.max_gyro:
            gyro = self.last_valid_gyro
        else:
            self.last_valid_gyro = gyro
            
        feat = np.concatenate([acc, gyro])
        self.buffer.append(feat)
        
        return ProcessedSample(
            timestamp_s=timestamp_s,
            dt=dt,
            ax=float(acc[0]),
            ay=float(acc[1]),
            az=float(acc[2]),
            gx=float(gyro[0]),
            gy=float(gyro[1]),
            gz=float(gyro[2]),
            lat=lat,
            lon=lon,
            gnss_speed_mps=gnss_speed_mps,
            gnss_accuracy_m=gnss_accuracy_m,
            gnss_bearing_deg=gnss_bearing_deg,
        )

    def is_ready(self) -> bool:
        """Returns True if the ring buffer contains a full 100-sample window."""
        return len(self.buffer) == self.window_size

    def get_classifier_window(self) -> Optional[np.ndarray]:
        """
        Returns shape (1, 100, 6) formatted for GRU Motion Classifier.
        """
        if not self.is_ready():
            return None
        arr = np.array(self.buffer, dtype=np.float32)
        return arr[np.newaxis, :, :] # (1, 100, 6)

    def get_tcn_window(self) -> Optional[np.ndarray]:
        """
        Returns shape (1, 6, 100) formatted for 1D-TCN Speed Estimator.
        """
        if not self.is_ready():
            return None
        arr = np.array(self.buffer, dtype=np.float32).T # (6, 100)
        return arr[np.newaxis, :, :] # (1, 6, 100)
