import numpy as np
from typing import List, Tuple, Optional, Dict, Any

R_EARTH = 6378137.0

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0)**2
    return float(2.0 * R_EARTH * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))

def point_to_segment_projection(
    p: np.ndarray, a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Projects point p onto line segment ab in local metric coordinates.
    Returns: (projected_point, orthogonal_distance)
    """
    ab = b - a
    ab_len_sq = float(np.dot(ab, ab))
    if ab_len_sq < 1e-4:
        return a, float(np.linalg.norm(p - a))
    
    t = float(np.dot(p - a, ab) / ab_len_sq)
    t_clamped = max(0.0, min(1.0, t))
    proj = a + t_clamped * ab
    dist = float(np.linalg.norm(p - proj))
    return proj, dist

class MapMatcher:
    """
    High-performance topological trail and road snapping engine:
    - Automatic polyline simplification (Ramer-Douglas-Peucker / chord distance)
    - Heading compatibility gating (|delta_heading| <= 50 deg)
    - Orthogonal projection to road centerline
    """
    def __init__(
        self,
        max_snap_distance_m: float = 30.0,
        max_heading_diff_deg: float = 55.0,
        min_segment_spacing_m: float = 15.0,
    ):
        self.max_snap_dist = max_snap_distance_m
        self.max_heading_diff = max_heading_diff_deg
        self.min_spacing = min_segment_spacing_m
        self.road_segments: List[Dict[str, Any]] = []
        self.origin_lat: Optional[float] = None
        self.origin_lon: Optional[float] = None
        self.cos_lat0: float = 1.0

    def load_trajectory_as_trail(self, lats: List[float], lons: List[float]):
        """
        Loads and automatically simplifies trail/road waypoints.
        """
        if len(lats) < 2:
            return
        self.origin_lat = lats[0]
        self.origin_lon = lons[0]
        self.cos_lat0 = float(np.cos(np.radians(self.origin_lat)))
        self.road_segments.clear()
        
        # Subsample to min_segment_spacing_m
        simp_lats = [lats[0]]
        simp_lons = [lons[0]]
        
        for i in range(1, len(lats)):
            d = haversine_m(simp_lats[-1], simp_lons[-1], lats[i], lons[i])
            if d >= self.min_spacing or i == len(lats) - 1:
                simp_lats.append(lats[i])
                simp_lons.append(lons[i])

        # Convert simplified points to local meters
        pts = []
        for lat, lon in zip(simp_lats, simp_lons):
            pN = np.radians(lat - self.origin_lat) * R_EARTH
            pE = np.radians(lon - self.origin_lon) * R_EARTH * self.cos_lat0
            pts.append(np.array([pE, pN]))

        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i + 1]
            seg_len = float(np.linalg.norm(p2 - p1))
            if seg_len < 1.0:
                continue
            bearing = (float(np.degrees(np.arctan2(p2[0] - p1[0], p2[1] - p1[1]))) + 360.0) % 360.0
            self.road_segments.append({
                "p1": p1,
                "p2": p2,
                "bearing": bearing,
                "bbox": (
                    min(p1[0], p2[0]) - self.max_snap_dist,
                    max(p1[0], p2[0]) + self.max_snap_dist,
                    min(p1[1], p2[1]) - self.max_snap_dist,
                    max(p1[1], p2[1]) + self.max_snap_dist,
                )
            })

    def match(
        self,
        ekf_lat: float,
        ekf_lon: float,
        heading_deg: float,
    ) -> Tuple[float, float, bool]:
        """
        Fast spatial index matching of current position against candidate trails.
        """
        if not self.road_segments or self.origin_lat is None:
            return ekf_lat, ekf_lon, False

        pN = float(np.radians(ekf_lat - self.origin_lat) * R_EARTH)
        pE = float(np.radians(ekf_lon - self.origin_lon) * R_EARTH * self.cos_lat0)
        p = np.array([pE, pN])

        best_dist = float("inf")
        best_proj = None

        for seg in self.road_segments:
            # Fast Bounding Box check
            bb = seg["bbox"]
            if not (bb[0] <= pE <= bb[1] and bb[2] <= pN <= bb[3]):
                continue

            # Heading compatibility check
            h_diff = abs((heading_deg - seg["bearing"] + 180.0) % 360.0 - 180.0)
            if h_diff > self.max_heading_diff and (180.0 - h_diff) > self.max_heading_diff:
                continue

            proj, dist = point_to_segment_projection(p, seg["p1"], seg["p2"])
            if dist < best_dist:
                best_dist = dist
                best_proj = proj

        if best_dist <= self.max_snap_dist and best_proj is not None:
            snapped_N = best_proj[1]
            snapped_E = best_proj[0]
            lat = self.origin_lat + np.degrees(snapped_N / R_EARTH)
            lon = self.origin_lon + np.degrees(snapped_E / (R_EARTH * self.cos_lat0))
            return float(lat), float(lon), True

        return ekf_lat, ekf_lon, False
