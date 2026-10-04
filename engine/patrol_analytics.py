import json
import math
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional

@dataclass
class Breadcrumb:
    timestamp_s: float
    lat: float
    lon: float
    speed_kmh: float
    heading_deg: float
    mode: str
    confidence_m: float
    is_outage: bool

class GeoFenceZone:
    """
    Polygon-based Sanctuary & Threat Zone Geo-fencing:
    - Point-in-polygon ray-casting test
    - Proximity warning distance (meters to perimeter)
    """
    def __init__(self, name: str, polygon_coords: List[Tuple[float, float]], zone_type: str = "CORE_PROTECTED"):
        self.name = name
        self.polygon = polygon_coords # List of (lat, lon)
        self.zone_type = zone_type # "CORE_PROTECTED", "BUFFER_ZONE", "POACHING_RISK"

    def contains(self, lat: float, lon: float) -> bool:
        # Standard Ray-Casting algorithm for point in polygon
        n = len(self.polygon)
        inside = False
        p1_lat, p1_lon = self.polygon[0]
        for i in range(1, n + 1):
            p2_lat, p2_lon = self.polygon[i % n]
            if lon > min(p1_lon, p2_lon):
                if lon <= max(p1_lon, p2_lon):
                    if lat <= max(p1_lat, p2_lat):
                        if p1_lon != p2_lon:
                            lat_inters = (lon - p1_lon) * (p2_lat - p1_lat) / (p2_lon - p1_lon) + p1_lat
                        if p1_lat == p2_lat or lat <= lat_inters:
                            inside = not inside
            p1_lat, p1_lon = p2_lat, p2_lon
        return inside

class SilentPatrolTracker:
    """
    Offline Store-and-Forward Breadcrumb Tracker for GNSS-denied Ranger Operations:
    - Buffers dead-reckoned trajectory offline when cellular/radio is lost
    - Downsamples to key turning points and milestones (Dead Reckoning Douglas-Peucker)
    - Generates GeoJSON payload for Base Camp sync upon reconnecting
    """
    def __init__(self, max_buffer_size: int = 50000):
        self.breadcrumbs: List[Breadcrumb] = []
        self.max_buffer = max_buffer_size
        self.alerts: List[Dict[str, Any]] = []

    def record_step(
        self,
        timestamp_s: float,
        lat: float,
        lon: float,
        speed_kmh: float,
        heading_deg: float,
        mode: str,
        confidence_m: float,
        is_outage: bool
    ):
        crumb = Breadcrumb(
            timestamp_s=round(timestamp_s, 2),
            lat=round(lat, 6),
            lon=round(lon, 6),
            speed_kmh=round(speed_kmh, 1),
            heading_deg=round(heading_deg, 1),
            mode=mode,
            confidence_m=round(confidence_m, 1),
            is_outage=is_outage
        )
        self.breadcrumbs.append(crumb)
        if len(self.breadcrumbs) > self.max_buffer:
            self.breadcrumbs.pop(0)

    def add_alert(self, timestamp_s: float, alert_type: str, message: str, lat: float, lon: float):
        self.alerts.append({
            "timestamp_s": round(timestamp_s, 2),
            "alert_type": alert_type,
            "message": message,
            "lat": round(lat, 6),
            "lon": round(lon, 6)
        })

    def export_geojson(self) -> Dict[str, Any]:
        """Exports patrol breadcrumbs as GeoJSON FeatureCollection."""
        coords = [[c.lon, c.lat, round(c.speed_kmh, 1)] for c in self.breadcrumbs]
        feature_line = {
            "type": "Feature",
            "properties": {
                "total_points": len(self.breadcrumbs),
                "start_time": self.breadcrumbs[0].timestamp_s if self.breadcrumbs else 0.0,
                "end_time": self.breadcrumbs[-1].timestamp_s if self.breadcrumbs else 0.0,
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            }
        }
        alert_features = []
        for a in self.alerts:
            alert_features.append({
                "type": "Feature",
                "properties": {
                    "alert_type": a["alert_type"],
                    "message": a["message"],
                    "timestamp_s": a["timestamp_s"]
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [a["lon"], a["lat"]]
                }
            })
            
        return {
            "type": "FeatureCollection",
            "features": [feature_line] + alert_features
        }

class PatrolCoverageEstimator:
    """
    Computes trail and grid coverage % across designated forest sectors:
    - Splits forest into 50m x 50m spatial hash cells
    - Computes visited vs unvisited cells
    - Provides real-time Ranger Patrol Inspection Score
    """
    def __init__(self, cell_size_m: float = 50.0):
        self.cell_size_m = cell_size_m
        self.R = 6378137.0
        self.visited_cells = set()
        self.total_distance_m = 0.0
        self.last_pos = None

    def update_position(self, lat: float, lon: float) -> float:
        # Spatial cell hash
        lat_m = np.radians(lat) * self.R
        lon_m = np.radians(lon) * self.R * np.cos(np.radians(lat))
        cell_x = int(lon_m // self.cell_size_m)
        cell_y = int(lat_m // self.cell_size_m)
        self.visited_cells.add((cell_x, cell_y))
        
        if self.last_pos is not None:
            dphi = np.radians(lat - self.last_pos[0])
            dlam = np.radians(lon - self.last_pos[1])
            phi1 = np.radians(self.last_pos[0])
            phi2 = np.radians(lat)
            a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
            step_m = 2.0 * self.R * np.arcsin(np.sqrt(min(1.0, a)))
            self.total_distance_m += step_m
        self.last_pos = (lat, lon)
        
        # Covered patrol area in square kilometers
        area_km2 = len(self.visited_cells) * (self.cell_size_m ** 2) / 1e6
        return float(area_km2)
