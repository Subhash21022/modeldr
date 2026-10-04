from enum import Enum
import numpy as np
from typing import Optional, Tuple

R_EARTH = 6378137.0

def haversine_dist(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return float(2.0 * R_EARTH * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))

class GNSSQuality(str, Enum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    LOST = "LOST"

class GNSSManager:
    """
    Evaluates real-time satellite fix integrity and computes dynamic EKF covariance:
    - Rejects cold-start GPS teleportation jumps (>35 m/s jump)
    - GOOD (<= 10m): High-confidence GNSS correction
    - DEGRADED (10m - 30m): Inflated covariance (EKF trusts dead reckoning more)
    - LOST (> 30m or timeout): Pure autonomous Dead Reckoning
    """
    def __init__(
        self,
        good_threshold_m: float = 10.0,
        degraded_threshold_m: float = 30.0,
        max_fix_age_s: float = 2.5,
        base_speed_sigma_mps: float = 0.5,
    ):
        self.good_thresh = good_threshold_m
        self.degraded_thresh = degraded_threshold_m
        self.max_age = max_fix_age_s
        self.base_speed_sigma = base_speed_sigma_mps
        
        self.last_fix_time: Optional[float] = None
        self.last_lat: Optional[float] = None
        self.last_lon: Optional[float] = None
        self.last_speed: Optional[float] = None
        self.current_quality = GNSSQuality.LOST

    def evaluate_fix(
        self,
        timestamp_s: float,
        lat: Optional[float],
        lon: Optional[float],
        accuracy_m: Optional[float],
        speed_mps: Optional[float] = None,
    ) -> Tuple[GNSSQuality, Optional[np.ndarray]]:
        """
        Evaluates GNSS health.
        Returns:
            quality: GNSSQuality (GOOD, DEGRADED, LOST)
            R_cov: (3, 3) measurement covariance matrix for [p_E, p_N, v], or None if LOST.
        """
        # 1. Null / Missing check
        if lat is None or lon is None or accuracy_m is None or np.isnan(lat) or np.isnan(lon) or np.isnan(accuracy_m):
            self.current_quality = GNSSQuality.LOST
            return GNSSQuality.LOST, None

        # 2. Fix age timeout check
        if self.last_fix_time is not None:
            age = timestamp_s - self.last_fix_time
            if age > self.max_age:
                self.current_quality = GNSSQuality.LOST
                return GNSSQuality.LOST, None
            
            # 3. Position teleportation check (e.g. cold-start cell tower jump)
            if self.last_lat is not None and self.last_lon is not None and age > 0:
                jump_dist = haversine_dist(self.last_lat, self.last_lon, lat, lon)
                implied_speed = jump_dist / max(0.01, age)
                if jump_dist > 50.0 and implied_speed > 35.0: # >126 km/h unphysical jump
                    self.current_quality = GNSSQuality.LOST
                    return GNSSQuality.LOST, None

        # Teleportation speed check (>50 m/s ~ 180 km/h)
        if speed_mps is not None and speed_mps > 50.0:
            self.current_quality = GNSSQuality.LOST
            return GNSSQuality.LOST, None

        self.last_fix_time = timestamp_s
        self.last_lat = lat
        self.last_lon = lon
        self.last_speed = speed_mps

        # 4. Graduated quality categorization & covariance construction
        if accuracy_m <= self.good_thresh:
            self.current_quality = GNSSQuality.GOOD
            pos_sigma = max(1.5, float(accuracy_m))
            spd_sigma = self.base_speed_sigma
            R_cov = np.diag([pos_sigma ** 2, pos_sigma ** 2, spd_sigma ** 2]).astype(np.float32)
            return GNSSQuality.GOOD, R_cov

        elif accuracy_m <= self.degraded_thresh:
            self.current_quality = GNSSQuality.DEGRADED
            pos_sigma = float(accuracy_m) * np.sqrt(8.0)
            spd_sigma = self.base_speed_sigma * 2.0
            R_cov = np.diag([pos_sigma ** 2, pos_sigma ** 2, spd_sigma ** 2]).astype(np.float32)
            return GNSSQuality.DEGRADED, R_cov

        else:
            self.current_quality = GNSSQuality.LOST
            return GNSSQuality.LOST, None
