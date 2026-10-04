import numpy as np
from collections import deque
from typing import Tuple, Optional

class ZUPTDetector:
    """
    Robust Multi-Modal Zero-Velocity Update (ZUPT) and Zero Angular Rate Update (ZARU) Detector.
    
    Addresses the real-world 'Engine Idle Problem':
    Idling engines (scooters, ATVs, patrol vehicles) produce high-frequency vibration
    (high raw variance ~2-6 (m/s^2)^2), but zero net displacement.
    
    Dual-Stage Architecture:
    1. Low-Pass Window Filter (0.50s): Rejects engine RPM harmonic vibration while capturing
       true kinematic translational acceleration and body turning rates.
    2. Hybrid Physical + Neural Gating: Combines low-pass variance thresholds with
       the neural motion classifier / speed estimator prior.
    
    When stationary is confirmed:
    - Velocity is clamped to 0.0 m/s (ZUPT).
    - True physical gyro yaw bias is observed from low-pass mean yaw rate (ZARU).
    - Gravity alignment vector g_down is continuously refined without movement corruption.
    """
    def __init__(
        self,
        sample_rate_hz: float = 100.0,
        window_size: int = 50,              # 0.50s window
        lp_acc_std_thresh: float = 0.35,     # m/s^2 on low-pass filtered acceleration
        lp_gyro_norm_thresh: float = 0.045,  # rad/s (~2.5 deg/s) on smoothed gyro
        gravity_margin: float = 1.8,        # m/s^2
        min_consecutive_stationary: int = 15 # 0.15s confirmation hysteresis
    ):
        self.fs = sample_rate_hz
        self.window_size = window_size
        self.lp_acc_std_thresh = lp_acc_std_thresh
        self.lp_gyro_norm_thresh = lp_gyro_norm_thresh
        self.gravity_margin = gravity_margin
        self.min_consecutive = min_consecutive_stationary
        
        self.raw_acc_buf = deque(maxlen=window_size)
        self.raw_gyr_buf = deque(maxlen=window_size)
        
        # Exponential smoothing state for engine vibration decoupling
        self.s_acc = np.array([0.0, 9.81, 0.0], dtype=np.float64)
        self.s_gyr = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.smoothed_acc_buf = deque(maxlen=window_size)
        self.smoothed_gyr_buf = deque(maxlen=window_size)
        
        self.stationary_counter = 0
        self.is_stationary = False
        self.current_gyro_bias_rads = 0.0
        
        # Invariant downward gravity vector (handlebar default)
        self.g_down = np.array([-0.060, 0.910, 0.411], dtype=np.float64)
        self.g_down /= np.linalg.norm(self.g_down)

    def add_sample(
        self,
        ax: float, ay: float, az: float,
        gx: float, gy: float, gz: float,
        neural_speed_mps: Optional[float] = None,
        neural_class: Optional[int] = None
    ) -> Tuple[bool, float, np.ndarray, float]:
        """
        Process single IMU sample at 100 Hz.
        Returns:
            is_stationary: bool
            confidence: float (0.0 to 1.0)
            g_down: np.ndarray (3,) unit vector
            gyro_yaw_bias: float (rad/s)
        """
        a = np.array([ax, ay, az], dtype=np.float64)
        g = np.array([gx, gy, gz], dtype=np.float64)
        
        # Low-pass filter (alpha = 0.15 ~ 15 Hz cutoff to remove 30-100 Hz engine harmonics)
        alpha = 0.15
        self.s_acc = (1.0 - alpha) * self.s_acc + alpha * a
        self.s_gyr = (1.0 - alpha) * self.s_gyr + alpha * g
        
        self.raw_acc_buf.append(a)
        self.smoothed_acc_buf.append(self.s_acc.copy())
        self.smoothed_gyr_buf.append(self.s_gyr.copy())
        
        if len(self.smoothed_acc_buf) < self.window_size:
            return False, 0.0, self.g_down, self.current_gyro_bias_rads

        # 1. Kinematic Low-Pass Statistics
        acc_s_arr = np.array(self.smoothed_acc_buf) # (W, 3)
        gyr_s_arr = np.array(self.smoothed_gyr_buf) # (W, 3)
        
        lp_acc_std = np.std(acc_s_arr, axis=0) # [std_x, std_y, std_z]
        max_lp_acc_std = float(np.max(lp_acc_std))
        
        mean_gyr_s = np.mean(gyr_s_arr, axis=0)
        mean_gyr_norm = float(np.linalg.norm(mean_gyr_s))
        
        mean_acc_norm = float(np.linalg.norm(np.mean(acc_s_arr, axis=0)))
        gravity_ok = abs(mean_acc_norm - 9.80665) < self.gravity_margin
        
        # 2. Physical Kinematic Stationary Check
        physical_still = (
            max_lp_acc_std < self.lp_acc_std_thresh
            and mean_gyr_norm < self.lp_gyro_norm_thresh
            and gravity_ok
        )
        
        # 3. Decision Logic: Physical stillness strictly takes precedence
        # Idling engine produces low LP-std (<0.35) and zero gyro norm (<0.045).
        # Neural speed or classifier cannot override physical stillness.
        if physical_still:
            is_still_instant = True
        elif neural_class is not None and neural_class == 0:
            is_still_instant = (mean_gyr_norm < self.lp_gyro_norm_thresh * 1.5)
        elif neural_speed_mps is not None and neural_speed_mps < 0.50:
            is_still_instant = (mean_gyr_norm < self.lp_gyro_norm_thresh * 1.5) and (max_lp_acc_std < self.lp_acc_std_thresh * 1.5)
        else:
            is_still_instant = False

        if is_still_instant:
            self.stationary_counter += 1
            if self.stationary_counter >= self.min_consecutive:
                self.is_stationary = True
                
                # Update Gravity Vector
                mean_a = np.mean(acc_s_arr, axis=0)
                mean_a_u = mean_a / np.linalg.norm(mean_a)
                g_alpha = 0.02
                self.g_down = (1.0 - g_alpha) * self.g_down + g_alpha * mean_a_u
                self.g_down /= np.linalg.norm(self.g_down)
                
                # Update ZARU Gyro Yaw Bias
                inst_yaw_rate = -float(np.dot(mean_gyr_s, self.g_down))
                b_alpha = 0.01
                self.current_gyro_bias_rads = (1.0 - b_alpha) * self.current_gyro_bias_rads + b_alpha * inst_yaw_rate
        else:
            self.stationary_counter = 0
            self.is_stationary = False

        confidence = 0.95 if self.is_stationary else 0.0
        return self.is_stationary, confidence, self.g_down, self.current_gyro_bias_rads
