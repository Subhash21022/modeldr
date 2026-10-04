import os
import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[MISSION] Device: {device}")

# =====================================================================
# 1. LOAD TRAINED MODELS
# =====================================================================
class MotionClassifierGRU(nn.Module):
    def __init__(self, in_features=6, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(in_features, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        out, _ = self.gru(x)
        return self.classifier(out[:, -1, :])

class SpeedEstimatorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

cls_model = MotionClassifierGRU(num_classes=3).to(device)
cls_model.load_state_dict(torch.load("output/motion_classifier.pth", map_location=device))
cls_model.eval()

veh_spd_model = SpeedEstimatorCNN().to(device)
veh_spd_model.load_state_dict(torch.load("output/universal_speed_estimator.pth", map_location=device))
veh_spd_model.eval()

ped_spd_model = SpeedEstimatorCNN().to(device)
ped_spd_model.load_state_dict(torch.load("output/pedestrian_speed_estimator.pth", map_location=device))
ped_spd_model.eval()

print("[MODELS] Successfully loaded:")
print("  - MotionClassifierGRU (Stationary / Vehicle / Walking)")
print("  - UniversalSpeedEstimatorCNN (Vehicle Driving Speed)")
print("  - PedestrianSpeedEstimatorCNN (Foot Patrol Speed)")

# =====================================================================
# 2. ASSEMBLE REALISTIC MISSION SEQUENCES
#    Leg 1: Scooter Driving (50s ~ 5000 samples)
#    Leg 2: Parked / Engine Idle at Trailhead (20s ~ 2000 samples)
#    Leg 3: Walking on Foot into Forest (50s ~ 5000 samples)
# =====================================================================
# Load Scooter Data
df_scooter = pd.read_csv("processed_scooter_test_data/scooter_test_all_100hz.csv")
df_s2 = df_scooter[df_scooter['dataset_id'] == 'dataset_2'].reset_index(drop=True)

# Leg 1: Scooter driving
leg1_scooter = df_s2[(df_s2['time_s'] >= 60.0) & (df_s2['time_s'] < 110.0)].copy().reset_index(drop=True)

# Leg 2: Scooter stopped at trailhead
leg2_idle = df_s2[(df_s2['time_s'] >= 220.0) & (df_s2['time_s'] < 240.0)].copy().reset_index(drop=True)

# Leg 3: ForestBack Walking data
fb_path = r"Datasets\ForestBack-Dataset-main\ForestBack-Dataset-main\Dataset\Dataset\forestback_indoor_pdr_dataset.csv"
df_fb = pd.read_csv(fb_path)
leg3_walk = df_fb.iloc[1000:6000].copy().reset_index(drop=True)

print(f"\n[MISSION PROFILE]")
print(f"  Leg 1 (Scooter Patrol) : {len(leg1_scooter)} samples ({len(leg1_scooter)*0.01:.1f}s)")
print(f"  Leg 2 (Park & Dismount): {len(leg2_idle)} samples ({len(leg2_idle)*0.01:.1f}s)")
print(f"  Leg 3 (Foot Patrol)    : {len(leg3_walk)} samples ({len(leg3_walk)*0.01:.1f}s)")
print(f"  Total Mission Duration : {(len(leg1_scooter) + len(leg2_idle) + len(leg3_walk))*0.01:.1f}s")

# =====================================================================
# 3. MULTI-MODAL EKF NAVIGATION ENGINE
# =====================================================================
R_EARTH = 6378137.0

class MultiModalEKF:
    def __init__(self, initial_heading_rad=0.0):
        self.x = np.array([0.0, 0.0, 0.0, initial_heading_rad, 0.0], dtype=np.float64) # [p_E, p_N, v, psi, bg]
        self.P = np.diag([2.0, 2.0, 0.5, 0.01, 1e-5]).astype(np.float64)
        self.Q_base = np.diag([0.01, 0.01, 0.10, 3e-5, 1e-6]).astype(np.float64)
        self.g_down = np.array([-0.060, 0.910, 0.411], dtype=np.float64)
        self.g_down /= np.linalg.norm(self.g_down)
        self.origin_lat, self.origin_lon = None, None
        self.cos_lat0 = 1.0
        self.current_mode = "VEHICLE"

    def set_origin(self, lat, lon):
        self.origin_lat, self.origin_lon = lat, lon
        self.cos_lat0 = float(np.cos(np.radians(lat)))

    def local_to_latlon(self, p_E, p_N):
        lat = self.origin_lat + np.degrees(p_N / R_EARTH)
        lon = self.origin_lon + np.degrees(p_E / (R_EARTH * self.cos_lat0))
        return float(lat), float(lon)

    def step(self, dt, ax, ay, az, gx, gy, gz, v_input, is_still=False):
        raw_gyro = np.array([gx, gy, gz], dtype=np.float64)
        omega_yaw = -float(np.dot(raw_gyro, self.g_down))
        unbiased_yaw_rate = omega_yaw - self.x[4]

        psi = self.x[3]
        if is_still:
            self.x[2] = 0.0
        else:
            self.x[2] = v_input
            self.x[0] += self.x[2] * np.sin(psi) * dt
            self.x[1] += self.x[2] * np.cos(psi) * dt
            self.x[3] += unbiased_yaw_rate * dt

        self.x[3] = (self.x[3] + np.pi) % (2 * np.pi) - np.pi

        # Jacobian F
        F = np.eye(5, dtype=np.float64)
        F[0, 2] = np.sin(psi) * dt
        F[0, 3] = self.x[2] * np.cos(psi) * dt
        F[1, 2] = np.cos(psi) * dt
        F[1, 3] = -self.x[2] * np.sin(psi) * dt
        F[3, 4] = -dt

        self.P = F @ self.P @ F.T + (self.Q_base * dt)

        # ZUPT & ZARU Updates during still periods
        if is_still:
            H_z = np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 1]], dtype=np.float64)
            R_z = np.diag([1e-4, 5e-4])
            y_z = np.array([0.0 - self.x[2], omega_yaw - self.x[4]])
            S_z = H_z @ self.P @ H_z.T + R_z
            K_z = self.P @ H_z.T @ np.linalg.inv(S_z)
            self.x += (K_z @ y_z)
            self.P = (np.eye(5) - K_z @ H_z) @ self.P
            self.x[2] = 0.0

        cur_lat, cur_lon = self.local_to_latlon(self.x[0], self.x[1])
        eigvals = np.linalg.eigvalsh(self.P[0:2, 0:2])
        conf_radius_m = float(2.0 * np.sqrt(max(1e-4, np.max(eigvals))))
        heading_deg = float(np.degrees(self.x[3]) % 360.0)

        return cur_lat, cur_lon, float(self.x[2]), heading_deg, conf_radius_m

# =====================================================================
# 4. RUN END-TO-END PATROL SIMULATION
# =====================================================================
init_lat = leg1_scooter['lat'].iloc[0]
init_lon = leg1_scooter['lon'].iloc[0]
init_bearing = np.radians(leg1_scooter['bearing_deg'].iloc[0])

ekf = MultiModalEKF(initial_heading_rad=init_bearing)
ekf.set_origin(init_lat, init_lon)
ekf.x[4] = 0.00058 # Dynamic gyro bias

# Mode manager hysteresis
mode_history = deque(maxlen=15) # 0.15s smoothing
mode_names = {0: "STATIONARY", 1: "VEHICLE", 2: "WALKING"}

history_t, history_lat, history_lon = [], [], []
history_spd, history_mode, history_conf = [], [], []

print("\n" + "=" * 80)
print(f"{'Time':<8} | {'Patrol Mode':<12} | {'Speed (m/s)':<12} | {'Heading':<9} | {'Lat':<11} | {'Lon':<11} | {'Conf Radius'}")
print("-" * 80)

# Build combined sample stream
stream = []
# Leg 1: Scooter
for i in range(len(leg1_scooter)):
    r = leg1_scooter.iloc[i]
    stream.append((r['ax'], r['ay'], r['az'], r['gx'], r['gy'], r['gz'], "LEG1_SCOOTER"))

# Leg 2: Idle
for i in range(len(leg2_idle)):
    r = leg2_idle.iloc[i]
    stream.append((r['ax'], r['ay'], r['az'], r['gx'], r['gy'], r['gz'], "LEG2_IDLE"))

# Leg 3: Walking (ForestBack)
for i in range(len(leg3_walk)):
    r = leg3_walk.iloc[i]
    stream.append((r['acc_x_mps2'], r['acc_y_mps2'], r['acc_z_mps2'], 0.0, 0.0, r['gyro_z_rads'], "LEG3_WALK"))

window_buf = deque(maxlen=100)
dt = 0.01

for t_idx, (ax, ay, az, gx, gy, gz, stage) in enumerate(stream):
    t_sec = t_idx * dt
    window_buf.append([ax, ay, az, gx, gy, gz])

    # Run neural inference every 10 samples (10 Hz)
    if len(window_buf) == 100 and (t_idx % 10 == 0):
        feat_w = np.array(window_buf, dtype=np.float32) # (100, 6)
        
        # 1. Motion Classification
        with torch.no_grad():
            cls_in = torch.tensor(feat_w).unsqueeze(0).to(device)
            raw_cls = torch.argmax(cls_model(cls_in), dim=1).item()
        mode_history.append(raw_cls)
        
        # Majority vote mode
        active_mode = max(set(mode_history), key=mode_history.count)
        
        # 2. Active Speed Estimation
        feat_cnn = torch.tensor(feat_w.T).unsqueeze(0).to(device) # (1, 6, 100)
        with torch.no_grad():
            if active_mode == 1: # Vehicle
                curr_spd = float(veh_spd_model(feat_cnn).item()) * 0.9473
                curr_spd = curr_spd if curr_spd >= 0.75 else 0.0
            elif active_mode == 2: # Walking
                curr_spd = float(ped_spd_model(feat_cnn).item())
                curr_spd = np.clip(curr_spd, 0.5, 2.2) # realistic walking range
            else: # Stationary
                curr_spd = 0.0
                
        active_mode_str = mode_names[active_mode]
    elif len(window_buf) < 100:
        active_mode_str = "VEHICLE"
        curr_spd = 6.5

    is_still = (active_mode_str == "STATIONARY")
    c_lat, c_lon, spd, head_deg, conf_r = ekf.step(
        dt=dt, ax=ax, ay=ay, az=az, gx=gx, gy=gy, gz=gz,
        v_input=curr_spd, is_still=is_still
    )

    history_t.append(t_sec)
    history_lat.append(c_lat)
    history_lon.append(c_lon)
    history_spd.append(spd)
    history_mode.append(active_mode_str)
    history_conf.append(conf_r)

    if t_idx % 1000 == 0 or t_idx == len(stream)-1:
        print(f"{t_sec:<8.1f} | {active_mode_str:<12} | {spd:<12.2f} | {head_deg:<9.1f} | {c_lat:<11.6f} | {c_lon:<11.6f} | ±{conf_r:.1f} m")

# =====================================================================
# 5. EXPORT & PLOT MULTI-MODAL TRAJECTORY
# =====================================================================
out_plot = "output/multimodal_patrol_trajectory.png"
plt.figure(figsize=(12, 7), dpi=140)

# Color-code by patrol mode
lats = np.array(history_lat)
lons = np.array(history_lon)
modes = np.array(history_mode)

plt.plot(lons[modes == 'VEHICLE'], lats[modes == 'VEHICLE'], 'b.', markersize=3, label='Scooter Patrol (Vehicle Mode)')
plt.plot(lons[modes == 'STATIONARY'], lats[modes == 'STATIONARY'], 'ro', markersize=6, label='Parked at Trailhead (ZUPT Active)')
plt.plot(lons[modes == 'WALKING'], lats[modes == 'WALKING'], 'g.', markersize=3, label='Foot Patrol (Deep PDR Mode)')

plt.scatter([lons[0]], [lats[0]], c='black', s=100, marker='s', label='Mission Start')
plt.scatter([lons[-1]], [lats[-1]], c='darkgreen', s=140, marker='*', label='Ranger Checkpoint Reached')

plt.title('Multi-Modal Forest Ranger Mission (Scooter Patrol -> Park & Dismount -> Foot Patrol)', fontsize=12, fontweight='bold')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='best', framealpha=0.9)
plt.tight_layout()
plt.savefig(out_plot)
plt.close()
print(f"\n[EXPORT] Saved Multi-Modal Trajectory Plot: {out_plot}")

# Save CSV report
mission_df = pd.DataFrame({
    "time_s": history_t,
    "lat": history_lat,
    "lon": history_lon,
    "speed_mps": history_spd,
    "patrol_mode": history_mode,
    "confidence_radius_m": history_conf
})
out_csv = "output/multimodal_mission_telemetry.csv"
mission_df.iloc[::100].to_csv(out_csv, index=False)
print(f"[EXPORT] Saved Mission Telemetry: {out_csv}")
print("\n>>> MULTI-MODAL PATROL SIMULATION COMPLETED SUCCESSFULLY! <<<")
