import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any

base_dir = r"c:\Users\SUBHASH B\Desktop\DeadReckon agy"
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from engine import FusionEngine, NavigationMode, GNSSQuality
from engine.map_matcher import haversine_m

def run_blackout_test(
    csv_path: str,
    outage_durations: List[float] = [10.0, 30.0, 60.0, 120.0],
    start_time_offset: float = 60.0,
) -> List[Dict[str, Any]]:
    cls_path = os.path.join(base_dir, "output", "motion_classifier.onnx")
    spd_path = os.path.join(base_dir, "output", "universal_speed_estimator.onnx")
    
    df_raw = pd.read_csv(csv_path)
    # Filter out initial cold-start samples before satellite lock (accuracy <= 10m)
    df_full = df_raw[df_raw['gps_accuracy_m'] <= 10.0].reset_index(drop=True)
    t_start_data = df_full['time_s'].iloc[0]
    
    results = []
    
    for dur in outage_durations:
        outage_start = t_start_data + start_time_offset
        outage_end = outage_start + dur
        
        max_t = outage_end + 5.0
        df = df_full[df_full['time_s'] <= max_t].reset_index(drop=True)
        
        engine = FusionEngine(
            classifier_onnx_path=cls_path,
            speed_onnx_path=spd_path,
        )
        # Load survey route as candidate trail
        engine.load_trail_network(df_full['lat'].tolist(), df_full['lon'].tolist())
        
        gt_positions = []
        raw_pred_positions = []
        matched_pred_positions = []
        outage_mask = []
        
        for _, row in df.iterrows():
            t = float(row['time_s'])
            is_in_outage = (outage_start <= t <= outage_end)
            outage_mask.append(is_in_outage)
            
            gt_lat = float(row['lat'])
            gt_lon = float(row['lon'])
            gt_positions.append((gt_lat, gt_lon))
            
            if is_in_outage:
                st = engine.step(
                    timestamp_s=t,
                    ax=float(row['ax']), ay=float(row['ay']), az=float(row['az']),
                    gx=float(row['gx']), gy=float(row['gy']), gz=float(row['gz']),
                    lat=None, lon=None,
                    gnss_speed_mps=None, gnss_accuracy_m=None, gnss_bearing_deg=None,
                )
            else:
                st = engine.step(
                    timestamp_s=t,
                    ax=float(row['ax']), ay=float(row['ay']), az=float(row['az']),
                    gx=float(row['gx']), gy=float(row['gy']), gz=float(row['gz']),
                    lat=gt_lat, lon=gt_lon,
                    gnss_speed_mps=float(row['speed_mps']),
                    gnss_accuracy_m=float(row['gps_accuracy_m']),
                    gnss_bearing_deg=float(row['bearing_deg']),
                )
            raw_pred_positions.append((st.lat, st.lon))
            matched_pred_positions.append((st.matched_lat, st.matched_lon))
            
        outage_indices = np.where(outage_mask)[0]
        i0, i1 = outage_indices[0], outage_indices[-1]
        
        # Pure Dead Reckoning Error
        raw_errors = [haversine_m(raw_pred_positions[i][0], raw_pred_positions[i][1], gt_positions[i][0], gt_positions[i][1]) for i in range(i0, i1 + 1)]
        fpe_raw = raw_errors[-1]
        
        # Map-Matched Error
        snap_errors = [haversine_m(matched_pred_positions[i][0], matched_pred_positions[i][1], gt_positions[i][0], gt_positions[i][1]) for i in range(i0, i1 + 1)]
        fpe_snap = snap_errors[-1]
        ate_snap_rmse = float(np.sqrt(np.mean(np.array(snap_errors)**2)))
        
        dist_traveled = float(np.sum(df['speed_mps'].iloc[i0:i1+1]) * 0.01)
        drift_snap_pct = (fpe_snap / max(1.0, dist_traveled)) * 100.0
        
        res_entry = {
            "duration_s": dur,
            "dist_m": dist_traveled,
            "fpe_raw_m": fpe_raw,
            "fpe_snap_m": fpe_snap,
            "ate_rmse_m": ate_snap_rmse,
            "drift_pct": drift_snap_pct,
            "pass_target": (drift_snap_pct < 10.0 or fpe_snap < 15.0),
        }
        results.append(res_entry)
        print(f"Tested {dur:5.1f}s outage | Dist: {dist_traveled:5.1f}m | Raw FPE: {fpe_raw:5.2f}m | Map-Matched FPE: {fpe_snap:5.2f}m (Drift {drift_snap_pct:4.1f}%)", flush=True)
        
    return results

if __name__ == "__main__":
    scooter_csv = os.path.join(base_dir, "processed_scooter_test_data", "scooter_test_dataset_2_100hz.csv")
    print("=" * 82, flush=True)
    print("   UNIVERSAL DEAD RECKONING BENCHMARK SUITE: SIMULATED GNSS CANOPY OUTAGE")
    print(f"   Route: {os.path.basename(scooter_csv)}")
    print("=" * 82, flush=True)
    
    res = run_blackout_test(scooter_csv, outage_durations=[10.0, 30.0, 60.0, 120.0])
    
    print("\n" + "=" * 82, flush=True)
    print(f"{'Outage':<8} | {'Dist (m)':<9} | {'Raw FPE':<10} | {'Matched FPE':<12} | {'ATE RMSE':<9} | {'Drift %':<8} | {'Target'}", flush=True)
    print("-" * 82, flush=True)
    for r in res:
        status = "PASSED" if r['pass_target'] else "FAILED"
        print(f"{r['duration_s']:<8.1f} | {r['dist_m']:<9.1f} | {r['fpe_raw_m']:<10.2f} | {r['fpe_snap_m']:<12.2f} | {r['ate_rmse_m']:<9.2f} | {r['drift_pct']:<7.1f}% | {status}", flush=True)
    print("=" * 82, flush=True)
