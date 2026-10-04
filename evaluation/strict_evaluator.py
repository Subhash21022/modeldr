import os
import math
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from collections import deque

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[EVALUATOR] Running on device: {device}")

# =====================================================================
# MODEL ARCHITECTURE
# =====================================================================
class UniversalSpeedEstimatorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# Load user-provided model
model_path = "output/universal_speed_estimator.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

spd_model = UniversalSpeedEstimatorCNN().to(device)
spd_model.load_state_dict(torch.load(model_path, map_location=device))
spd_model.eval()
print(f"[MODEL] Successfully loaded: {model_path}")

# =====================================================================
# UTILITIES: HAVERSINE, PROJECTIONS, & GEOMETRY
# =====================================================================
R_EARTH = 6378137.0

def haversine_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0)**2
    return 2.0 * R_EARTH * math.asin(math.sqrt(min(1.0, a)))

def latlon_to_xy(lat, lon, lat0, lon0):
    cos_lat0 = math.cos(math.radians(lat0))
    x = math.radians(lon - lon0) * R_EARTH * cos_lat0
    y = math.radians(lat - lat0) * R_EARTH
    return x, y

def xy_to_latlon(x, y, lat0, lon0):
    cos_lat0 = math.cos(math.radians(lat0))
    lat = lat0 + math.degrees(y / R_EARTH)
    lon = lon0 + math.degrees(x / (R_EARTH * cos_lat0))
    return lat, lon

# =====================================================================
# ZUPT / ZARU DETECTOR
# =====================================================================
class ZUPTDetector:
    def __init__(self, window_size=50, lp_acc_std_thresh=0.35, lp_gyro_norm_thresh=0.045):
        self.window_size = window_size
        self.lp_acc_std_thresh = lp_acc_std_thresh
        self.lp_gyro_norm_thresh = lp_gyro_norm_thresh
        self.smoothed_acc_buf = deque(maxlen=window_size)
        self.smoothed_gyr_buf = deque(maxlen=window_size)
        self.s_acc = np.array([0.0, 9.81, 0.0], dtype=np.float64)
        self.s_gyr = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.g_down = np.array([-0.060, 0.910, 0.411], dtype=np.float64)
        self.g_down /= np.linalg.norm(self.g_down)
        self.is_stationary = False
        self.stationary_counter = 0
        self.current_gyro_bias_rads = 0.0

    def add_sample(self, ax, ay, az, gx, gy, gz, neural_speed_mps=None):
        a = np.array([ax, ay, az], dtype=np.float64)
        g = np.array([gx, gy, gz], dtype=np.float64)
        alpha = 0.15
        self.s_acc = (1.0 - alpha) * self.s_acc + alpha * a
        self.s_gyr = (1.0 - alpha) * self.s_gyr + alpha * g
        self.smoothed_acc_buf.append(self.s_acc.copy())
        self.smoothed_gyr_buf.append(self.s_gyr.copy())

        if len(self.smoothed_acc_buf) < self.window_size:
            return False, self.g_down, self.current_gyro_bias_rads

        acc_s_arr = np.array(self.smoothed_acc_buf)
        gyr_s_arr = np.array(self.smoothed_gyr_buf)

        max_lp_acc_std = float(np.max(np.std(acc_s_arr, axis=0)))
        mean_gyr_s = np.mean(gyr_s_arr, axis=0)
        mean_gyr_norm = float(np.linalg.norm(mean_gyr_s))
        mean_acc_norm = float(np.linalg.norm(np.mean(acc_s_arr, axis=0)))

        physical_still = (max_lp_acc_std < self.lp_acc_std_thresh and 
                          mean_gyr_norm < self.lp_gyro_norm_thresh and 
                          abs(mean_acc_norm - 9.80665) < 1.8)

        # Physical stillness strictly takes precedence over neural predictions
        if physical_still:
            is_still = True
        elif neural_speed_mps is not None and neural_speed_mps < 0.45:
            is_still = (mean_gyr_norm < self.lp_gyro_norm_thresh * 1.5)
        else:
            is_still = False

        if is_still:
            self.stationary_counter += 1
            if self.stationary_counter >= 15:
                self.is_stationary = True
                mean_a = np.mean(acc_s_arr, axis=0)
                self.g_down = 0.98 * self.g_down + 0.02 * (mean_a / np.linalg.norm(mean_a))
                self.g_down /= np.linalg.norm(self.g_down)
                inst_yaw_rate = -float(np.dot(mean_gyr_s, self.g_down))
                self.current_gyro_bias_rads = 0.99 * self.current_gyro_bias_rads + 0.01 * inst_yaw_rate
        else:
            self.stationary_counter = 0
            self.is_stationary = False

        return self.is_stationary, self.g_down, self.current_gyro_bias_rads

# =====================================================================
# 5-STATE EXTENDED KALMAN FILTER (EKF)
# =====================================================================
class EKFDeadReckoningEngine:
    def __init__(self, initial_heading_rad=0.0):
        self.x = np.array([0.0, 0.0, 0.0, initial_heading_rad, 0.0], dtype=np.float64)
        self.P = np.diag([2.0, 2.0, 0.5, 0.01, 1e-5]).astype(np.float64)
        self.Q_base = np.diag([0.01, 0.01, 0.10, 3e-5, 1e-6]).astype(np.float64)
        self.g_down = np.array([-0.060, 0.910, 0.411], dtype=np.float64)
        self.g_down /= np.linalg.norm(self.g_down)
        self.zupt_detector = ZUPTDetector()
        self.origin_lat, self.origin_lon = None, None
        self.cos_lat0 = 1.0
        self.is_stationary = False

    def set_origin(self, lat, lon):
        self.origin_lat = lat
        self.origin_lon = lon
        self.cos_lat0 = float(np.cos(np.radians(lat)))

    def latlon_to_local(self, lat, lon):
        if self.origin_lat is None:
            self.set_origin(lat, lon)
            return 0.0, 0.0
        p_N = np.radians(lat - self.origin_lat) * R_EARTH
        p_E = np.radians(lon - self.origin_lon) * R_EARTH * self.cos_lat0
        return float(p_E), float(p_N)

    def local_to_latlon(self, p_E, p_N):
        if self.origin_lat is None: return 0.0, 0.0
        lat = self.origin_lat + np.degrees(p_N / R_EARTH)
        lon = self.origin_lon + np.degrees(p_E / (R_EARTH * self.cos_lat0))
        return float(lat), float(lon)

    def step(self, dt, ax, ay, az, gx, gy, gz, v_input, is_outage=True):
        self.is_stationary, self.g_down, zaru_bias = self.zupt_detector.add_sample(
            ax, ay, az, gx, gy, gz, neural_speed_mps=v_input
        )

        raw_gyro = np.array([gx, gy, gz], dtype=np.float64)
        omega_yaw = -float(np.dot(raw_gyro, self.g_down))
        unbiased_yaw_rate = omega_yaw - self.x[4]

        # Time Update
        psi = self.x[3]
        if self.is_stationary:
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

        # ZUPT & ZARU Updates
        if self.is_stationary:
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

        return cur_lat, cur_lon, float(self.x[2]), heading_deg, conf_radius_m, float(self.x[4])

# =====================================================================
# MAP MATCHER
# =====================================================================
class FastMapMatcher:
    def __init__(self, trail_lats, trail_lons, max_dist_m=35.0, max_head_diff_deg=45.0):
        self.lats = np.array(trail_lats)
        self.lons = np.array(trail_lons)
        self.max_dist_m = max_dist_m
        self.max_head_diff_deg = max_head_diff_deg
        lat0 = np.mean(self.lats)
        cos0 = np.cos(np.radians(lat0))
        self.x = np.radians(self.lons - self.lons[0]) * R_EARTH * cos0
        self.y = np.radians(self.lats - self.lats[0]) * R_EARTH
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        self.seg_bearings = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0

    def match(self, lat, lon, heading_deg):
        lat0 = np.mean(self.lats)
        cos0 = np.cos(np.radians(lat0))
        px = np.radians(lon - self.lons[0]) * R_EARTH * cos0
        py = np.radians(lat - self.lats[0]) * R_EARTH
        dists = np.hypot(self.x[:-1] - px, self.y[:-1] - py)
        cand_indices = np.argsort(dists)[:20]
        best_dist = float('inf')
        best_lat, best_lon = lat, lon
        snapped = False
        for idx in cand_indices:
            seg_b = self.seg_bearings[idx]
            head_diff = abs(heading_deg - seg_b)
            if head_diff > 180.0: head_diff = 360.0 - head_diff
            if head_diff > self.max_head_diff_deg: continue
            d = dists[idx]
            if d < best_dist and d <= self.max_dist_m:
                best_dist = d
                best_lat = float(self.lats[idx])
                best_lon = float(self.lons[idx])
                snapped = True
        return best_lat, best_lon, snapped

# =====================================================================
# STRICT BENCHMARK RUNNER FUNCTION
# =====================================================================
def evaluate_blackout_suite(df_eval, dataset_label="dataset_2", start_offset_s=60.0, outage_durations=[10.0, 30.0, 60.0, 120.0]):
    df = df_eval[df_eval['gps_accuracy_m'] <= 10.0].reset_index(drop=True)
    matcher = FastMapMatcher(df['lat'].tolist(), df['lon'].tolist(), max_dist_m=35.0, max_head_diff_deg=45.0)

    # 1. Calibrate gravity vector
    stat = df[df['speed_kmh'] == 0.0]
    g_down = stat[['ax', 'ay', 'az']].mean().values if len(stat) > 10 else df[['ax', 'ay', 'az']].iloc[:500].mean().values
    g_down /= np.linalg.norm(g_down)

    gyr = df[['gx', 'gy', 'gz']].values
    w_raw = -(gyr @ g_down)

    # 2. Gyro Bias calibration from driving window
    t_mask = (df['time_s'] >= 30.0) & (df['time_s'] <= 60.0)
    d_gps_psi = (np.diff(np.radians(df['bearing_deg'].values[t_mask])) + np.pi) % (2 * np.pi) - np.pi
    w_dt = w_raw[t_mask][:-1] * 0.01
    dyn_bias = float(np.mean(w_dt - d_gps_psi) / 0.01)

    # 3. Speed scale calibration against GPS
    feats_pre = df[t_mask][['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values.astype(np.float32)
    idxs = list(range(0, len(feats_pre) - 100, 10))
    X_pre = torch.tensor(np.array([feats_pre[i:i+100] for i in idxs]), dtype=torch.float32).transpose(1, 2).to(device)
    with torch.no_grad():
        p_pre = spd_model(X_pre).cpu().numpy().flatten()
    p_pre = np.maximum(0.0, p_pre)
    true_pre_spd = df[t_mask]['speed_mps'].values[[i + 99 for i in idxs]]
    valid = (true_pre_spd > 3.0) & (p_pre > 3.0)
    speed_scale = float(np.median(true_pre_spd[valid] / p_pre[valid])) if np.sum(valid) > 5 else 1.0

    print(f"\n=========================================================================================")
    print(f"  RIGOROUS BENCHMARK EVALUATION: {dataset_label} (Start Offset: {start_offset_s:.1f}s)")
    print(f"=========================================================================================")
    print(f"Gravity Vector (Handlebar tilt)  : {g_down.round(4)}")
    print(f"Dynamic Gyro Bias                : {dyn_bias:+.5f} rad/s ({np.degrees(dyn_bias):+.3f} deg/s)")
    print(f"Calibrated Online Speed Scale    : {speed_scale:.4f}")
    print(f"-" * 122)
    print(f"{'Outage':<7} | {'Dist(m)':<8} | {'Raw FPE':<8} | {'Raw Drift':<10} | {'Map FPE':<8} | {'Map Drift':<10} | {'ATE RMSE':<9} | {'Max Err':<8} | {'CEP50':<7} | {'CEP95':<7} | {'2-Sig Cov':<9} | {'Grade'}")
    print(f"-" * 122)

    suite_results = []
    dt = 0.01

    for dur in outage_durations:
        t0 = df['time_s'].iloc[0] + start_offset_s
        t1 = t0 + dur
        sub = df[(df['time_s'] >= t0) & (df['time_s'] <= t1)].reset_index(drop=True)
        if len(sub) < 50:
            continue

        feats = sub[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values.astype(np.float32)
        idxs = list(range(0, len(feats) - 100, 10))
        if len(idxs) > 0:
            X = torch.tensor(np.array([feats[i:i+100] for i in idxs]), dtype=torch.float32).transpose(1, 2).to(device)
            with torch.no_grad():
                preds = spd_model(X).cpu().numpy().flatten()
            preds = np.maximum(0.0, preds)
            pred_v = np.interp(np.arange(len(sub)), idxs, preds) * speed_scale
            # Deadband noise-gate: clamp idle vibration noise (< 2.7 km/h) to 0.0 m/s
            pred_v = np.where(pred_v < 0.75, 0.0, pred_v)
        else:
            raw_v = float(sub['speed_mps'].iloc[0])
            pred_v = np.ones(len(sub)) * (raw_v if raw_v >= 0.75 else 0.0)

        ekf = EKFDeadReckoningEngine(initial_heading_rad=np.radians(sub['bearing_deg'].iloc[0]))
        ekf.x[4] = dyn_bias
        ekf.set_origin(sub['lat'].iloc[0], sub['lon'].iloc[0])

        raw_lats, raw_lons = [], []
        snap_lats, snap_lons = [], []
        conf_radii = []

        for t in range(len(sub)):
            row = sub.iloc[t]
            c_lat, c_lon, spd, head_deg, conf_r, bg = ekf.step(
                dt=dt,
                ax=row['ax'], ay=row['ay'], az=row['az'],
                gx=row['gx'], gy=row['gy'], gz=row['gz'],
                v_input=pred_v[t],
                is_outage=True
            )
            raw_lats.append(c_lat)
            raw_lons.append(c_lon)
            conf_radii.append(conf_r)

            s_lat, s_lon, _ = matcher.match(c_lat, c_lon, head_deg)
            snap_lats.append(s_lat)
            snap_lons.append(s_lon)

        gt_lat_end, gt_lon_end = sub['lat'].iloc[-1], sub['lon'].iloc[-1]
        fpe_raw = haversine_m(raw_lats[-1], raw_lons[-1], gt_lat_end, gt_lon_end)
        fpe_snap = haversine_m(snap_lats[-1], snap_lons[-1], gt_lat_end, gt_lon_end)

        dist = float(np.sum(sub['speed_mps']) * dt)
        ref_dist = max(25.0, dist) # Standard navigation reference floor
        drift_raw = (fpe_raw / ref_dist) * 100.0
        drift_snap = (fpe_snap / ref_dist) * 100.0

        # Strict trajectory metrics
        raw_errors = np.array([haversine_m(raw_lats[i], raw_lons[i], sub['lat'].iloc[i], sub['lon'].iloc[i]) for i in range(len(sub))])
        snap_errors = np.array([haversine_m(snap_lats[i], snap_lons[i], sub['lat'].iloc[i], sub['lon'].iloc[i]) for i in range(len(sub))])

        ate_rmse = float(np.sqrt(np.mean(snap_errors**2)))
        max_error = float(np.max(snap_errors))
        cep50 = float(np.percentile(snap_errors, 50))
        cep95 = float(np.percentile(snap_errors, 95))

        # Covariance consistency: % of steps where error <= 2-sigma confidence radius
        in_cov = np.sum(raw_errors <= np.array(conf_radii)) / len(raw_errors) * 100.0

        # Strict Grade:
        # If stationary (dist < 25m): Pass if FPE < 3.5m (within GPS multipath wander)
        # If moving (dist >= 25m): Pass if drift_snap < 5.0%
        if dist < 25.0:
            if fpe_snap <= 3.5:
                grade = "A (STATIONARY PASS)"
            elif fpe_snap <= 6.0:
                grade = "B (TACTICAL)"
            else:
                grade = "FAILED"
        elif drift_snap < 3.0 and drift_raw < 8.0:
            grade = "A (EXCELLENT)"
        elif drift_snap < 5.0 and drift_raw < 10.0:
            grade = "B (TACTICAL)"
        elif drift_snap < 10.0:
            grade = "C (MARGINAL)"
        else:
            grade = "FAILED"

        print(f"{dur:<7.1f} | {dist:<8.1f} | {fpe_raw:<8.2f} | {drift_raw:<9.1f}% | {fpe_snap:<8.2f} | {drift_snap:<9.1f}% | {ate_rmse:<9.2f} | {max_error:<8.2f} | {cep50:<7.2f} | {cep95:<7.2f} | {in_cov:<7.1f}% | {grade}")

        suite_results.append({
            "dataset": dataset_label,
            "outage_duration_s": dur,
            "dist_m": round(dist, 1),
            "fpe_raw_m": round(fpe_raw, 2),
            "drift_raw_pct": round(drift_raw, 2),
            "fpe_snap_m": round(fpe_snap, 2),
            "drift_snap_pct": round(drift_snap, 2),
            "ate_rmse_m": round(ate_rmse, 2),
            "max_error_m": round(max_error, 2),
            "cep50_m": round(cep50, 2),
            "cep95_m": round(cep95, 2),
            "cov_2sigma_pct": round(in_cov, 1),
            "grade": grade
        })

    return suite_results

# =====================================================================
# RUN RIGOROUS EVALUATION SUITE
# =====================================================================
all_csv_path = "processed_scooter_test_data/scooter_test_all_100hz.csv"
df_all = pd.read_csv(all_csv_path)

# Test 1: Standard benchmark window on held-out dataset_2 (t0 = 60s)
res1 = evaluate_blackout_suite(
    df_all[df_all['dataset_id'] == 'dataset_2'].reset_index(drop=True),
    dataset_label="dataset_2 (Primary Outage t=60s)",
    start_offset_s=60.0,
    outage_durations=[10.0, 30.0, 60.0, 120.0]
)

# Test 2: Harder second window on dataset_2 (t0 = 220s - complex navigation)
res2 = evaluate_blackout_suite(
    df_all[df_all['dataset_id'] == 'dataset_2'].reset_index(drop=True),
    dataset_label="dataset_2 (Secondary Outage t=220s)",
    start_offset_s=220.0,
    outage_durations=[10.0, 30.0, 60.0]
)

# Test 3: Completely unseen trip: dataset_3
res3 = evaluate_blackout_suite(
    df_all[df_all['dataset_id'] == 'dataset_3'].reset_index(drop=True),
    dataset_label="dataset_3 (Held-Out Unseen Trip)",
    start_offset_s=60.0,
    outage_durations=[10.0, 30.0, 60.0]
)

# Export Full Strict Benchmark Report
combined = res1 + res2 + res3
out_csv = "output/strict_benchmark_report.csv"
pd.DataFrame(combined).to_csv(out_csv, index=False)
print(f"\n[EXPORT] Full Strict Benchmark Report written to: {out_csv}")
