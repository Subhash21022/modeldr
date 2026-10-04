from enum import Enum
from collections import deque
from typing import Tuple, List

class NavigationMode(str, Enum):
    STATIONARY = "STATIONARY"
    VEHICLE = "VEHICLE"
    PEDESTRIAN = "PEDESTRIAN"

class ModeManager:
    """
    State machine with hysteresis debouncing to prevent navigation mode jitter.
    Translates raw classifier labels:
      0: Stationary -> STATIONARY
      1: Vehicle Cruising -> VEHICLE
      2: Vehicle Turning -> VEHICLE
      3: Walking -> PEDESTRIAN
    """
    def __init__(
        self,
        history_window_size: int = 15, # 15 predictions (~1.5s at 10Hz inference)
        vehicle_threshold: float = 0.70,
        pedestrian_threshold: float = 0.70,
        stationary_threshold: float = 0.65,
    ):
        self.history = deque(maxlen=history_window_size)
        self.current_mode = NavigationMode.STATIONARY
        self.vehicle_threshold = vehicle_threshold
        self.pedestrian_threshold = pedestrian_threshold
        self.stationary_threshold = stationary_threshold
        self.mode_confidence = 1.0

    def update(self, raw_class_id: int) -> Tuple[NavigationMode, float]:
        """
        Receives raw class prediction (0, 1, 2, 3) from the Motion Classifier.
        Returns: (debounced_mode, confidence)
        """
        # Map raw class ID to target navigation mode
        if raw_class_id == 0:
            candidate = NavigationMode.STATIONARY
        elif raw_class_id in (1, 2):
            candidate = NavigationMode.VEHICLE
        elif raw_class_id == 3:
            candidate = NavigationMode.PEDESTRIAN
        else:
            candidate = self.current_mode
            
        self.history.append(candidate)
        
        # Count occurrences in the rolling debounce window
        n = len(self.history)
        n_stat = sum(1 for m in self.history if m == NavigationMode.STATIONARY)
        n_veh = sum(1 for m in self.history if m == NavigationMode.VEHICLE)
        n_ped = sum(1 for m in self.history if m == NavigationMode.PEDESTRIAN)
        
        ratio_stat = n_stat / n
        ratio_veh = n_veh / n
        ratio_ped = n_ped / n
        
        # Hysteresis switching rules
        if self.current_mode == NavigationMode.STATIONARY:
            # Fast breakout if vehicle starts moving
            if ratio_veh >= 0.50 and candidate == NavigationMode.VEHICLE:
                self.current_mode = NavigationMode.VEHICLE
                self.mode_confidence = ratio_veh
            elif ratio_ped >= 0.50 and candidate == NavigationMode.PEDESTRIAN:
                self.current_mode = NavigationMode.PEDESTRIAN
                self.mode_confidence = ratio_ped
            else:
                self.mode_confidence = ratio_stat

        elif self.current_mode == NavigationMode.VEHICLE:
            if ratio_stat >= self.stationary_threshold:
                self.current_mode = NavigationMode.STATIONARY
                self.mode_confidence = ratio_stat
            elif ratio_ped >= self.pedestrian_threshold:
                self.current_mode = NavigationMode.PEDESTRIAN
                self.mode_confidence = ratio_ped
            else:
                self.mode_confidence = ratio_veh

        elif self.current_mode == NavigationMode.PEDESTRIAN:
            if ratio_stat >= self.stationary_threshold:
                self.current_mode = NavigationMode.STATIONARY
                self.mode_confidence = ratio_stat
            elif ratio_veh >= self.vehicle_threshold:
                self.current_mode = NavigationMode.VEHICLE
                self.mode_confidence = ratio_veh
            else:
                self.mode_confidence = ratio_ped
                
        return self.current_mode, self.mode_confidence
