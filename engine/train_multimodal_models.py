import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[TRAINER] Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

out_dir = "output"
os.makedirs(out_dir, exist_ok=True)

# =====================================================================
# 1. LOAD DATASETS: SCOOTER, FORESTBACK, & RONIN
# =====================================================================
scooter_path = "processed_scooter_test_data/scooter_test_all_100hz.csv"
fb_path = r"Datasets\ForestBack-Dataset-main\ForestBack-Dataset-main\Dataset\Dataset\forestback_indoor_pdr_dataset.csv"
ronin_path = "Datasets/ronin_walking_sample_100hz.csv"

print(f"\n[DATA] Loading Scooter dataset from: {scooter_path}")
df_scooter = pd.read_csv(scooter_path)
print(f"       Loaded {len(df_scooter):,} scooter telematics rows.")

print(f"[DATA] Loading ForestBack walking dataset from: {fb_path}")
df_fb = pd.read_csv(fb_path)
print(f"       Loaded {len(df_fb):,} ForestBack walking rows.")

print(f"[DATA] Loading RoNiN walking dataset from: {ronin_path}")
df_ronin = pd.read_csv(ronin_path)
print(f"       Loaded {len(df_ronin):,} RoNiN walking rows.")

# =====================================================================
# 2. BUILD 3-CLASS MULTI-MODAL DATASET
#    Class 0: Stationary (Parked / Engine Idle / Stopped)
#    Class 1: Vehicle (Scooter Driving / Cruising / Turns)
#    Class 2: Walking (Foot Patrol)
# =====================================================================
print("\n" + "=" * 70)
print("  STEP 1: PREPARING 3-CLASS MULTI-MODAL MOTION CLASSIFIER DATA")
print("=" * 70)

win = 100
step = 25

# Class 0: Stationary (from scooter speed < 0.5 km/h)
stat_mask = df_scooter['speed_kmh'] < 0.5
stat_df = df_scooter[stat_mask].reset_index(drop=True)
stat_feats = stat_df[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values.astype(np.float32)
stat_windows = [stat_feats[i:i+win] for i in range(0, len(stat_feats)-win, step)]

# Class 1: Vehicle (from scooter speed >= 4.0 km/h)
veh_mask = df_scooter['speed_kmh'] >= 4.0
veh_df = df_scooter[veh_mask].reset_index(drop=True)
veh_feats = veh_df[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values.astype(np.float32)
veh_windows = [veh_feats[i:i+win] for i in range(0, len(veh_feats)-win, step * 4)]

# Class 2: Walking (from ForestBack & RoNiN)
# ForestBack features: acc_x_mps2, acc_y_mps2, acc_z_mps2, gyro_z_rads
fb_acc = df_fb[['acc_x_mps2', 'acc_y_mps2', 'acc_z_mps2']].values.astype(np.float32)
fb_gz = df_fb['gyro_z_rads'].values.astype(np.float32)
fb_feats = np.column_stack([fb_acc, np.zeros_like(fb_gz), np.zeros_like(fb_gz), fb_gz])
fb_windows = [fb_feats[i:i+win] for i in range(0, len(fb_feats)-win, step)]

# RoNiN features: ax, ay, az, gx, gy, gz
ronin_feats = df_ronin[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values.astype(np.float32)
ronin_windows = [ronin_feats[i:i+win] for i in range(0, len(ronin_feats)-win, step)]
walk_windows = fb_windows + ronin_windows

# Balance dataset across classes
n_samples = min(len(stat_windows), len(veh_windows), len(walk_windows), 8000)
print(f"Extracted Windows:")
print(f"  Class 0 (Stationary) : {len(stat_windows):,} -> sampled {n_samples:,}")
print(f"  Class 1 (Vehicle)    : {len(veh_windows):,} -> sampled {n_samples:,}")
print(f"  Class 2 (Walking)    : {len(walk_windows):,} -> sampled {n_samples:,}")

np.random.seed(42)
stat_sel = np.random.choice(len(stat_windows), n_samples, replace=False)
veh_sel = np.random.choice(len(veh_windows), n_samples, replace=False)
walk_sel = np.random.choice(len(walk_windows), n_samples, replace=False)

X_cls = np.array([stat_windows[i] for i in stat_sel] + 
                 [veh_windows[i] for i in veh_sel] + 
                 [walk_windows[i] for i in walk_sel], dtype=np.float32)
y_cls = np.array([0] * n_samples + [1] * n_samples + [2] * n_samples, dtype=np.int64)

# Shuffle
perm = np.random.permutation(len(X_cls))
X_cls = X_cls[perm]
y_cls = y_cls[perm]

# Split 80/20 train/val
split = int(0.8 * len(X_cls))
X_tr_cls, y_tr_cls = torch.tensor(X_cls[:split]).to(device), torch.tensor(y_cls[:split]).to(device)
X_val_cls, y_val_cls = torch.tensor(X_cls[split:]).to(device), torch.tensor(y_cls[split:]).to(device)

# =====================================================================
# 3. TRAIN MOTION CLASSIFIER GRU (3 CLASSES)
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

torch.manual_seed(42)
cls_model = MotionClassifierGRU(num_classes=3).to(device)
cls_crit = nn.CrossEntropyLoss()
cls_opt = torch.optim.AdamW(cls_model.parameters(), lr=1e-3, weight_decay=1e-4)

cls_loader = DataLoader(TensorDataset(X_tr_cls, y_tr_cls), batch_size=128, shuffle=True)
print("\n[TRAIN] Training MotionClassifierGRU (Stationary / Vehicle / Walking)...")

t0 = time.time()
for epoch in range(1, 11):
    cls_model.train()
    total_loss = 0.0
    for bx, by in cls_loader:
        cls_opt.zero_grad()
        out = cls_model(bx)
        loss = cls_crit(out, by)
        loss.backward()
        cls_opt.step()
        total_loss += loss.item() * len(bx)
    
    cls_model.eval()
    with torch.no_grad():
        val_preds = torch.argmax(cls_model(X_val_cls), dim=1)
        val_acc = (val_preds == y_val_cls).float().mean().item() * 100.0
    
    if epoch % 2 == 0 or epoch == 10:
        print(f"  Epoch {epoch:2d}/10 | Train Loss: {total_loss/len(X_tr_cls):.4f} | Val Accuracy: {val_acc:.2f}%")

cls_pth = os.path.join(out_dir, "motion_classifier.pth")
torch.save(cls_model.state_dict(), cls_pth)
print(f"[EXPORT] Saved MotionClassifierGRU to: {cls_pth}")

# =====================================================================
# 4. PREPARE PEDESTRIAN SPEED ESTIMATOR DATA
# =====================================================================
print("\n" + "=" * 70)
print("  STEP 2: PREPARING PEDESTRIAN WALKING SPEED ESTIMATOR DATA")
print("=" * 70)

# ForestBack: Ground truth walking speed from step frequency * step length
fb_speed = (df_fb['adaptive_step_length_m'].fillna(0.70) * df_fb['step_frequency_hz'].fillna(1.8)).values.astype(np.float32)

fb_X, fb_y = [], []
for i in range(0, len(fb_feats) - win, 20):
    fb_X.append(fb_feats[i:i+win])
    fb_y.append(fb_speed[i+win-1])

# RoNiN: Compute adaptive Weinberg step frequency speed with peak detection
from scipy.signal import find_peaks
amag = np.linalg.norm(ronin_feats[:, :3], axis=1)
ronin_X, ronin_y = [], []
for i in range(0, len(ronin_feats) - win, 25):
    w_a = amag[i:i+win]
    peaks, _ = find_peaks(w_a - 9.81, height=0.6, distance=35)
    cadence = len(peaks)
    step_len = 0.42 * (max(0.2, np.max(w_a) - np.min(w_a))**0.25)
    v = float(np.clip(cadence * step_len, 0.0, 2.5))
    ronin_X.append(ronin_feats[i:i+win])
    ronin_y.append(v)

X_ped = np.array(fb_X + ronin_X, dtype=np.float32).transpose(0, 2, 1) # (N, 6, 100)
y_ped = np.array(fb_y + ronin_y, dtype=np.float32).reshape(-1, 1)

print(f"Total Pedestrian Windows: {len(X_ped):,} (ForestBack: {len(fb_X):,}, RoNiN: {len(ronin_X):,})")
print(f"Mean Walking Speed      : {np.mean(y_ped):.2f} m/s ({np.mean(y_ped)*3.6:.1f} km/h)")
print(f"Max Walking Speed       : {np.max(y_ped):.2f} m/s ({np.max(y_ped)*3.6:.1f} km/h)")

perm_ped = np.random.permutation(len(X_ped))
X_ped = X_ped[perm_ped]
y_ped = y_ped[perm_ped]

split_ped = int(0.8 * len(X_ped))
X_tr_ped, y_tr_ped = torch.tensor(X_ped[:split_ped]).to(device), torch.tensor(y_ped[:split_ped]).to(device)
X_val_ped, y_val_ped = torch.tensor(X_ped[split_ped:]).to(device), torch.tensor(y_ped[split_ped:]).to(device)

# =====================================================================
# 5. TRAIN PEDESTRIAN SPEED ESTIMATOR (1D-CNN)
# =====================================================================
class PedestrianSpeedEstimatorCNN(nn.Module):
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

torch.manual_seed(42)
ped_model = PedestrianSpeedEstimatorCNN().to(device)
ped_crit = nn.HuberLoss()
ped_opt = torch.optim.AdamW(ped_model.parameters(), lr=1e-3, weight_decay=1e-4)

ped_loader = DataLoader(TensorDataset(X_tr_ped, y_tr_ped), batch_size=128, shuffle=True)
print("\n[TRAIN] Training PedestrianSpeedEstimatorCNN...")

for epoch in range(1, 11):
    ped_model.train()
    total_loss = 0.0
    for bx, by in ped_loader:
        ped_opt.zero_grad()
        out = ped_model(bx)
        loss = ped_crit(out, by)
        loss.backward()
        ped_opt.step()
        total_loss += loss.item() * len(bx)
        
    ped_model.eval()
    with torch.no_grad():
        val_preds = ped_model(X_val_ped)
        val_mae = torch.mean(torch.abs(val_preds - y_val_ped)).item()
        
    if epoch % 2 == 0 or epoch == 10:
        print(f"  Epoch {epoch:2d}/10 | Train Loss: {total_loss/len(X_tr_ped):.4f} | Val MAE: {val_mae:.3f} m/s ({val_mae*3.6:.2f} km/h)")

ped_pth = os.path.join(out_dir, "pedestrian_speed_estimator.pth")
torch.save(ped_model.state_dict(), ped_pth)
print(f"[EXPORT] Saved PedestrianSpeedEstimatorCNN to: {ped_pth}")

print("\n" + "=" * 70)
print(f"  ALL MULTI-MODAL MODELS SUCCESSFULLY TRAINED IN {time.time()-t0:.1f}s!")
print(f"  1. Motion Classifier   : {cls_pth}")
print(f"  2. Pedestrian Speed    : {ped_pth}")
print(f"  3. Vehicle Speed       : output/universal_speed_estimator.pth")
print("=" * 70)
