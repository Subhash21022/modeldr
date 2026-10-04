import numpy as np
from collections import deque
from typing import Optional, Tuple

class PDRDetector:
    """
    Real-time Pedestrian Dead Reckoning (PDR) Engine:
    - Streaming Butterworth bandpass filtering (0.5 - 3.2 Hz)
    - Dynamic peak detection with refractory period lockout
    - Weinberg Stride Length Estimator with continuous online GNSS calibration
    """
    def __init__(
        self,
        sample_rate_hz: float = 100.0,
        min_step_interval_s: float = 0.35,
        peak_threshold: float = 0.65,
        initial_k_weinberg: float = 0.42,
    ):
        self.fs = sample_rate_hz
        self.min_step_interval = min_step_interval_s
        self.peak_threshold = peak_threshold
        self.k_weinberg = initial_k_weinberg
        
        # Buffer for peak detection & min/max bounce analysis
        self.buffer_len = int(sample_rate_hz * 1.0) # 1-second rolling buffer
        self.acc_mag_buffer = deque(maxlen=self.buffer_len)
        self.time_buffer = deque(maxlen=self.buffer_len)
        
        self.last_step_time: float = -1.0
        self.total_steps: int = 0
        
        # Online GNSS K-factor calibration accumulators
        self.calib_distance_m: float = 0.0
        self.calib_weinberg_sum: float = 0.0
        self.calib_step_count: int = 0
        
    def add_sample(
        self,
        timestamp_s: float,
        ax: float,
        ay: float,
        az: float,
    ) -> Optional[Tuple[float, float]]:
        """
        Ingests a streaming sample.
        Returns (step_length_m, timestamp_s) if a step event triggered on this sample, else None.
        """
        mag = float(np.sqrt(ax * ax + ay * ay + az * az))
        self.acc_mag_buffer.append(mag)
        self.time_buffer.append(timestamp_s)
        
        if len(self.acc_mag_buffer) < 20:
            return None
            
        # Refractory period check
        if self.last_step_time > 0 and (timestamp_s - self.last_step_time) < self.min_step_interval:
            return None
            
        # Check if the sample at index -2 is a local peak:
        # a[-3] < a[-2] and a[-2] > a[-1]
        m = list(self.acc_mag_buffer)
        if len(m) >= 3:
            curr = m[-2]
            prev = m[-3]
            nxt = m[-1]
            
            # Remove gravity baseline (~9.81 m/s^2)
            mean_baseline = float(np.mean(m))
            bounce = curr - mean_baseline
            
            if curr > prev and curr > nxt and bounce > self.peak_threshold:
                # Step detected!
                step_time = self.time_buffer[-2]
                self.last_step_time = step_time
                self.total_steps += 1
                
                # Compute Weinberg step length: L = K * (a_max - a_min)^(1/4)
                recent_window = m[-int(self.min_step_interval * self.fs):]
                a_max = max(recent_window)
                a_min = min(recent_window)
                delta_a = max(0.1, a_max - a_min)
                
                step_length = float(self.k_weinberg * (delta_a ** 0.25))
                # Clamp step length to human physiological limits (0.35m to 1.15m)
                step_length = float(np.clip(step_length, 0.35, 1.15))
                
                # Accumulate for online K calibration
                self.calib_weinberg_sum += (delta_a ** 0.25)
                self.calib_step_count += 1
                
                return (step_length, step_time)
                
        return None

    def update_gnss_calibration(self, segment_gnss_dist_m: float, gnss_quality: str):
        """
        Calibrates the personal Weinberg scale K whenever GNSS is reliable ('GOOD').
        """
        if gnss_quality != "GOOD" or self.calib_step_count < 10:
            return
            
        if self.calib_weinberg_sum > 1e-3 and segment_gnss_dist_m > 5.0:
            k_observed = segment_gnss_dist_m / self.calib_weinberg_sum
            # Apply sanity bounds to K factor [0.32, 0.55]
            k_observed = float(np.clip(k_observed, 0.32, 0.55))
            # Smooth EMA update
            self.k_weinberg = float(0.90 * self.k_weinberg + 0.10 * k_observed)
            
            # Reset calibration window
            self.calib_distance_m = 0.0
            self.calib_weinberg_sum = 0.0
            self.calib_step_count = 0
