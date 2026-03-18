"""
╔══════════════════════════════════════════════════════════════════╗
║       AUTONOMOUS DRONE NAVIGATION - DISASTER ZONE RESCUE         ║
║              AirSim Implementation  |  v2.0                      ║
║   Scenarios: Earthquake | Volcano | Flood                        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import airsim
import time
import math
import random
import json
import csv
import os
import sys
import numpy as np
from datetime import datetime
from collections import deque
import threading
import cv2

# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────

class SimConfig:
    # Environment
    AREA_SIZE          = 30.0          # 30×30 m operational zone
    FLIGHT_ALTITUDE    = 4.0           # Default cruise altitude (m)
    MIN_ALT            = 2.0
    MAX_ALT            = 8.0

    # Obstacle avoidance
    SAFETY_DISTANCE    = 3.0           # Repulsion activates within 3 m
    DANGER_DISTANCE    = 2.5           # RED status
    CAUTION_DISTANCE   = 3.5           # YELLOW status
    SCAN_RANGE         = 5.0           # Proximity sensor range (m)

    # Survivor detection
    FOV_ANGLE_DEG      = 60.0          # 60° forward cone
    DETECTION_RANGE    = 8.0           # Max detection distance (m)
    TOTAL_SURVIVORS    = 20
    SURVIVOR_MOVE_M    = 0.03          # Movement per step (m)

    # Navigation
    WAYPOINT_THRESHOLD = 1.5           # Distance to "reach" a waypoint (m)
    SPEED              = 5.0           # m/s cruise speed
    LOOKAHEAD          = 2.0           # Trajectory prediction steps

    # Camera
    CAM_WIDTH          = 320
    CAM_HEIGHT         = 240
    CAM_CAPTURE_EVERY  = 15            # frames

    # Wind
    WIND_MAX_N         = 1.0           # ±1.0 N

    # Experiment
    HEADLESS           = False         # Set True for batch runs
    LOG_DIR            = "logs"
    IMG_DIR            = "images"
    CSV_DIR            = "results"

    # Scenarios
    SCENARIOS          = ["earthquake", "volcano", "flood"]


# ─────────────────────────────────────────────────────────────────
#  WAYPOINTS  (x, y, z  in NED — z is negative = up in AirSim)
# ─────────────────────────────────────────────────────────────────

WAYPOINTS = [
    {"name": "Entry Point Alpha",    "pos": ( 0.0,  0.0, -3.0)},
    {"name": "Corridor Junction",    "pos": ( 4.0,  2.0, -3.5)},
    {"name": "Debris Field Center",  "pos": ( 8.0,  0.0, -4.0)},
    {"name": "Collapsed Sector",     "pos": (10.0,  5.0, -4.5)},
    {"name": "North Ruins",          "pos": ( 6.0,  9.0, -5.0)},
    {"name": "West Tower Base",      "pos": (-2.0,  8.0, -4.0)},
    {"name": "Central Courtyard",    "pos": ( 0.0,  4.0, -3.5)},
    {"name": "South Passage",        "pos": (-5.0,  1.0, -3.0)},
    {"name": "Lower Ruins",          "pos": (-8.0,  5.0, -4.0)},
    {"name": "East Debris",          "pos": ( 3.0, -4.0, -3.5)},
    {"name": "Canyon Exit",          "pos": ( 7.0, -7.0, -4.0)},
    {"name": "Final Approach",       "pos": (-3.0, -5.0, -3.0)},
    {"name": "Command Center",       "pos": ( 0.0,  0.0, -3.0)},  # return
]

# ─────────────────────────────────────────────────────────────────
#  SURVIVORS
# ─────────────────────────────────────────────────────────────────

NATO = [
    "Alpha","Bravo","Charlie","Delta","Echo","Foxtrot","Golf","Hotel",
    "India","Juliet","Kilo","Lima","Mike","November","Oscar","Papa",
    "Quebec","Romeo","Sierra","Tango"
]

def generate_survivors():
    """Place 20 survivors near waypoints with some scatter."""
    survivors = []
    anchor_wps = WAYPOINTS[:-1]  # exclude return waypoint
    for i in range(SimConfig.TOTAL_SURVIVORS):
        wp = anchor_wps[i % len(anchor_wps)]
        x = wp["pos"][0] + random.uniform(-4, 4)
        y = wp["pos"][1] + random.uniform(-4, 4)
        x = max(-14, min(14, x))
        y = max(-14, min(14, y))
        survivors.append({
            "id":       i,
            "name":     f"Survivor {NATO[i]}",
            "pos":      [x, y, 0.0],   # ground level
            "detected": False,
            "active":   True,
        })
    return survivors


# ─────────────────────────────────────────────────────────────────
#  SCENARIO OBSTACLE LAYOUTS  (returned as list of dicts)
# ─────────────────────────────────────────────────────────────────

def generate_obstacles(scenario: str):
    """
    Returns a list of obstacle descriptors for the chosen scenario.
    AirSim obstacle spawning requires Unreal Engine setup.
    This function documents positions used in the UE map, and also
    provides data for software-based avoidance in case dynamic
    obstacles are injected via the API.
    """
    rng = random.Random(42)  # deterministic layout

    buildings = []
    for _ in range(19):
        bx = rng.uniform(-12, 12)
        by = rng.uniform(-12, 12)
        bh = rng.uniform(3.5, 5.2)
        tilt = rng.uniform(-5, 5)
        buildings.append({"type": "building", "x": bx, "y": by,
                           "h": bh, "tilt": tilt})

    debris = []
    for _ in range(23):
        dx = rng.uniform(-13, 13)
        dy = rng.uniform(-13, 13)
        dh = rng.uniform(1.0, 1.5)
        debris.append({"type": "debris", "x": dx, "y": dy, "h": dh})

    verticals = []
    for _ in range(8):
        vx = rng.uniform(-12, 12)
        vy = rng.uniform(-12, 12)
        vh = rng.uniform(2.5, 3.2)
        verticals.append({"type": "pole", "x": vx, "y": vy, "h": vh})

    fire_zones = []
    n_fires = {"earthquake": 4, "volcano": 12, "flood": 6}[scenario]
    for _ in range(n_fires):
        fx = rng.uniform(-10, 10)
        fy = rng.uniform(-10, 10)
        fire_zones.append({"type": "fire", "x": fx, "y": fy, "r": 2.0})

    return buildings + debris + verticals + fire_zones


# ─────────────────────────────────────────────────────────────────
#  METRICS TRACKER
# ─────────────────────────────────────────────────────────────────

class MissionMetrics:
    def __init__(self):
        self.start_time        = time.time()
        self.end_time          = None
        self.waypoints_done    = 0
        self.total_wps         = len(WAYPOINTS)
        self.collisions        = 0
        self.near_misses       = 0
        self.avoidances        = 0
        self.survivors_found   = []
        self.images_captured   = 0
        self.speed_samples     = []
        self.positions         = deque(maxlen=500)
        self.path_length       = 0.0
        self.prev_pos          = None
        self.wp_times          = []
        self.wp_start          = time.time()

    # ── helpers ──────────────────────────────────────────────────

    def update_position(self, x, y, z):
        if self.prev_pos is not None:
            d = math.sqrt((x - self.prev_pos[0])**2 +
                          (y - self.prev_pos[1])**2 +
                          (z - self.prev_pos[2])**2)
            self.path_length += d
        self.prev_pos = (x, y, z)
        self.positions.append((x, y, z))

    def update_speed(self, vx, vy, vz):
        spd = math.sqrt(vx**2 + vy**2 + vz**2)
        self.speed_samples.append(spd)

    def waypoint_reached(self, idx):
        now = time.time()
        elapsed = now - self.wp_start
        self.wp_times.append(elapsed)
        self.wp_start = now
        self.waypoints_done = idx + 1

    def add_survivor(self, survivor, dist):
        self.survivors_found.append({
            "name":      survivor["name"],
            "x":         round(survivor["pos"][0], 2),
            "y":         round(survivor["pos"][1], 2),
            "timestamp": round(time.time() - self.start_time, 2),
            "distance":  round(dist, 2),
        })

    # ── derived ──────────────────────────────────────────────────

    @property
    def elapsed(self):
        t = self.end_time or time.time()
        return t - self.start_time

    @property
    def avg_speed(self):
        return sum(self.speed_samples) / len(self.speed_samples) \
               if self.speed_samples else 0.0

    @property
    def safety_score(self):
        return max(0.0, 100 - self.collisions * 10 - self.near_misses * 0.5)

    @property
    def detection_rate(self):
        return len(self.survivors_found) / SimConfig.TOTAL_SURVIVORS * 100

    @property
    def path_efficiency(self):
        # Optimal = sum of straight-line distances between consecutive WPs
        opt = 0.0
        for i in range(len(WAYPOINTS) - 1):
            p1, p2 = WAYPOINTS[i]["pos"], WAYPOINTS[i + 1]["pos"]
            opt += math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))
        if self.path_length > 0:
            return min(100.0, opt / self.path_length * 100)
        return 0.0

    @property
    def avg_wp_time(self):
        return sum(self.wp_times) / len(self.wp_times) if self.wp_times else 0.0

    # ── report generation ────────────────────────────────────────

    def print_telemetry(self, wp_idx, status):
        bar_len = 20
        pct = self.waypoints_done / self.total_wps
        filled = int(pct * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\n{'─'*60}")
        print(f"  ⏱  Mission time    : {self.elapsed:6.1f}s")
        print(f"  🗺  Waypoint        : {self.waypoints_done}/{self.total_wps}  [{bar}] {pct*100:.0f}%")
        print(f"  💨  Avg speed       : {self.avg_speed:.2f} m/s")
        print(f"  🛑  Status          : {status}")
        print(f"  👥  Survivors found : {len(self.survivors_found)}/{SimConfig.TOTAL_SURVIVORS}")
        print(f"  📸  Images captured : {self.images_captured}")
        print(f"{'─'*60}")

    def print_final_report(self):
        print("\n" + "═"*60)
        print("         🏁  FINAL MISSION REPORT")
        print("═"*60)
        print(f"  Mission Duration    : {self.elapsed:.1f}s")
        print(f"  Total Distance      : {self.path_length:.1f}m")
        print(f"  Average Speed       : {self.avg_speed:.2f} m/s")
        print(f"  Waypoint Completion : {self.waypoints_done}/{self.total_wps}")
        print(f"  Avg Time / Waypoint : {self.avg_wp_time:.1f}s")
        print(f"  ─────────────────────────────────────────────")
        print(f"  Collisions          : {self.collisions}")
        print(f"  Near Misses         : {self.near_misses}")
        print(f"  Successful Avoids   : {self.avoidances}")
        print(f"  Safety Score        : {self.safety_score:.1f}/100")
        print(f"  ─────────────────────────────────────────────")
        print(f"  Survivors Found     : {len(self.survivors_found)}/{SimConfig.TOTAL_SURVIVORS}")
        print(f"  Detection Rate      : {self.detection_rate:.1f}%")
        print(f"  Images Captured     : {self.images_captured}")
        print(f"  Camera Rate         : {self.images_captured/max(1,self.elapsed):.2f} img/s")
        print(f"  ─────────────────────────────────────────────")
        print(f"  Path Efficiency     : {self.path_efficiency:.1f}%")
        print(f"  Optimal Distance    : {self.path_length*self.path_efficiency/100:.1f}m")
        print("═"*60)

    def print_rescue_report(self):
        print("\n" + "═"*60)
        print("         🚁  RESCUE COORDINATE REPORT")
        print("═"*60)
        if not self.survivors_found:
            print("  No survivors detected.")
        for i, s in enumerate(self.survivors_found, 1):
            print(f"  [{i:02d}] {s['name']:<22} "
                  f"({s['x']:+.2f}, {s['y']:+.2f})  "
                  f"@ {s['timestamp']:.1f}s  |  dist={s['distance']:.1f}m")
        print("═"*60)

    def save_csv(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["survivor", "x", "y", "timestamp", "distance"])
            for s in self.survivors_found:
                w.writerow([s["name"], s["x"], s["y"],
                             s["timestamp"], s["distance"]])

    def save_json(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "duration_s":        round(self.elapsed, 2),
            "distance_m":        round(self.path_length, 2),
            "avg_speed_ms":      round(self.avg_speed, 2),
            "waypoints":         f"{self.waypoints_done}/{self.total_wps}",
            "collisions":        self.collisions,
            "near_misses":       self.near_misses,
            "avoidances":        self.avoidances,
            "safety_score":      round(self.safety_score, 2),
            "survivors_found":   len(self.survivors_found),
            "detection_rate_pct":round(self.detection_rate, 1),
            "images_captured":   self.images_captured,
            "path_efficiency_pct":round(self.path_efficiency, 1),
            "survivors":         self.survivors_found,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────────
#  POTENTIAL-FIELD NAVIGATOR
# ─────────────────────────────────────────────────────────────────

class PotentialFieldNav:
    """
    Combines:
      • Attraction toward target waypoint
      • Repulsion from nearby obstacles
      • Wind disturbance
    Returns a velocity setpoint (vx, vy, vz).
    """

    def __init__(self, obstacles: list):
        self.obstacles  = obstacles      # list of obstacle dicts
        self.wind_force = [0.0, 0.0]    # updated each step

    def update_wind(self):
        self.wind_force = [
            random.uniform(-SimConfig.WIND_MAX_N, SimConfig.WIND_MAX_N),
            random.uniform(-SimConfig.WIND_MAX_N, SimConfig.WIND_MAX_N),
        ]

    def compute_velocity(self, drone_pos, target_pos) -> tuple:
        """
        drone_pos, target_pos: (x, y, z) in AirSim NED metres
        Returns (vx, vy, vz) velocity setpoint.
        """
        dx = target_pos[0] - drone_pos[0]
        dy = target_pos[1] - drone_pos[1]
        dz = target_pos[2] - drone_pos[2]
        dist_to_target = math.sqrt(dx**2 + dy**2 + dz**2) + 1e-6

        # Attraction force (proportional to distance, clamped)
        attract_scale = min(SimConfig.SPEED, dist_to_target * 1.5)
        ax = (dx / dist_to_target) * attract_scale
        ay = (dy / dist_to_target) * attract_scale
        az = (dz / dist_to_target) * attract_scale * 0.5

        # Repulsion forces
        rx, ry, rz = 0.0, 0.0, 0.0
        for obs in self.obstacles:
            ox, oy = obs["x"], obs["y"]
            ddx = drone_pos[0] - ox
            ddy = drone_pos[1] - oy
            d   = math.sqrt(ddx**2 + ddy**2) + 1e-6
            if d < SimConfig.SCAN_RANGE:
                strength = (SimConfig.SAFETY_DISTANCE - d) * 6.0
                strength = max(0.0, strength)
                rx += (ddx / d) * strength
                ry += (ddy / d) * strength

        # Wind disturbance (treated as a velocity nudge)
        wx = self.wind_force[0] * 0.1
        wy = self.wind_force[1] * 0.1

        vx = ax + rx + wx
        vy = ay + ry + wy
        vz = az

        # Clamp to max speed
        spd = math.sqrt(vx**2 + vy**2 + vz**2) + 1e-6
        max_spd = SimConfig.SPEED
        if spd > max_spd:
            vx, vy, vz = vx/spd*max_spd, vy/spd*max_spd, vz/spd*max_spd

        return vx, vy, vz

    def obstacle_status(self, drone_pos) -> tuple:
        """
        Returns (status_str, min_dist, repulsion_vector).
        status: 'SAFE' | 'CAUTION' | 'DANGER'
        """
        min_dist = float("inf")
        for obs in self.obstacles:
            d = math.sqrt((drone_pos[0]-obs["x"])**2 +
                          (drone_pos[1]-obs["y"])**2)
            if d < min_dist:
                min_dist = d

        if min_dist <= SimConfig.DANGER_DISTANCE:
            return "DANGER", min_dist
        elif min_dist <= SimConfig.CAUTION_DISTANCE:
            return "CAUTION", min_dist
        else:
            return "SAFE", min_dist


# ─────────────────────────────────────────────────────────────────
#  FOV SURVIVOR DETECTOR
# ─────────────────────────────────────────────────────────────────

class SurvivorDetector:
    """
    Detects survivors inside a forward-facing 60° FOV cone up to 8 m.
    """

    def check(self, drone_pos, drone_yaw_rad, survivors: list) -> list:
        """
        Returns list of newly detected survivors.
        """
        newly_detected = []
        fov_half = math.radians(SimConfig.FOV_ANGLE_DEG / 2)

        for s in survivors:
            if s["detected"]:
                continue

            sx, sy = s["pos"][0], s["pos"][1]
            dx     = sx - drone_pos[0]
            dy     = sy - drone_pos[1]
            dist   = math.sqrt(dx**2 + dy**2)

            if dist > SimConfig.DETECTION_RANGE:
                continue

            # Bearing to survivor
            bearing = math.atan2(dy, dx)
            angle_diff = bearing - drone_yaw_rad
            # Normalise to [-π, π]
            angle_diff = (angle_diff + math.pi) % (2*math.pi) - math.pi

            if abs(angle_diff) <= fov_half:
                s["detected"] = True
                newly_detected.append((s, dist))

        return newly_detected

    def move_survivors(self, survivors: list):
        """Simulate injured survivor micro-movement."""
        for s in survivors:
            if s["detected"]:
                continue
            s["pos"][0] += random.uniform(-SimConfig.SURVIVOR_MOVE_M,
                                           SimConfig.SURVIVOR_MOVE_M)
            s["pos"][1] += random.uniform(-SimConfig.SURVIVOR_MOVE_M,
                                           SimConfig.SURVIVOR_MOVE_M)
            s["pos"][0] = max(-20, min(20, s["pos"][0]))
            s["pos"][1] = max(-20, min(20, s["pos"][1]))


# ─────────────────────────────────────────────────────────────────
#  CAMERA / IMAGE HANDLER
# ─────────────────────────────────────────────────────────────────

class DroneCamera:
    def __init__(self, client: airsim.MultirotorClient, save_dir: str):
        self.client    = client
        self.save_dir  = save_dir
        self.frame_idx = 0
        self.captured  = 0
        os.makedirs(save_dir, exist_ok=True)

    def try_capture(self):
        """Capture image every CAM_CAPTURE_EVERY frames."""
        self.frame_idx += 1
        if self.frame_idx % SimConfig.CAM_CAPTURE_EVERY != 0:
            return None

        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(
                    "front_center",
                    airsim.ImageType.Scene,
                    False,
                    False
                )
            ])
            if responses and responses[0].width > 0:
                img1d = np.frombuffer(responses[0].image_data_uint8,
                                      dtype=np.uint8)
                img   = img1d.reshape(responses[0].height,
                                       responses[0].width, 3)
                # Add Gaussian noise (σ=3) to simulate sensor noise
                noise = np.random.normal(0, 3, img.shape).astype(np.int16)
                img   = np.clip(img.astype(np.int16) + noise, 0, 255)\
                               .astype(np.uint8)
                fname = os.path.join(
                    self.save_dir,
                    f"frame_{self.captured:05d}.png"
                )
                cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                self.captured += 1
                return img
        except Exception as e:
            pass  # Silently skip if camera not available
        return None


# ─────────────────────────────────────────────────────────────────
#  DISTANCE SENSOR (using AirSim distance sensors / raycasts)
# ─────────────────────────────────────────────────────────────────

class ProximitySensor:
    """
    360° proximity detection using AirSim distance sensor API.
    Falls back to software-based obstacle distance if sensor not configured.
    """

    def __init__(self, client: airsim.MultirotorClient, obstacles: list):
        self.client    = client
        self.obstacles = obstacles

    def get_min_distance(self, drone_pos) -> float:
        """Return minimum distance to any obstacle (software fallback)."""
        min_d = float("inf")
        for obs in self.obstacles:
            d = math.sqrt((drone_pos[0]-obs["x"])**2 +
                          (drone_pos[1]-obs["y"])**2)
            if d < min_d:
                min_d = d
        return min_d

    def get_airsim_distance(self) -> float:
        """
        Try reading from AirSim distance sensor (if configured in settings.json).
        """
        try:
            data = self.client.getDistanceSensorData()
            return data.distance
        except Exception:
            return float("inf")


# ─────────────────────────────────────────────────────────────────
#  FLIGHT PATH LOGGER  (stores last N positions for trail)
# ─────────────────────────────────────────────────────────────────

class FlightTrail:
    MAX_LEN = 150

    def __init__(self):
        self.positions = deque(maxlen=self.MAX_LEN)

    def update(self, x, y, z, frame):
        if frame % 3 == 0:
            self.positions.append((x, y, z))

    def as_list(self):
        return list(self.positions)


# ─────────────────────────────────────────────────────────────────
#  MAIN SIMULATION CLASS
# ─────────────────────────────────────────────────────────────────

class DisasterDroneSim:
    """
    Full autonomous drone search-and-rescue simulation.
    Connects to AirSim, runs the mission, collects all metrics.
    """

    def __init__(self, scenario: str = "earthquake"):
        if scenario not in SimConfig.SCENARIOS:
            raise ValueError(f"Scenario must be one of {SimConfig.SCENARIOS}")

        self.scenario  = scenario
        self.obstacles = generate_obstacles(scenario)
        self.survivors = generate_survivors()
        self.metrics   = MissionMetrics()
        self.trail     = FlightTrail()
        self.navigator = PotentialFieldNav(self.obstacles)
        self.detector  = SurvivorDetector()
        self.frame     = 0
        self.running   = False

        # AirSim client
        self.client = airsim.MultirotorClient()
        self._run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sub-systems (initialised after connection)
        self.camera     = None
        self.prox       = None
        self._connected = False   # set True only after confirmConnection()

        self._print_banner()

    # ── banner ───────────────────────────────────────────────────

    def _print_banner(self):
        print("╔══════════════════════════════════════════════════════════╗")
        print("║    AUTONOMOUS DRONE NAVIGATION  —  DISASTER ZONE SAR    ║")
        print("║                   AirSim  |  v2.0                       ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print(f"\n  Scenario    : {self.scenario.upper()}")
        print(f"  Waypoints   : {len(WAYPOINTS)}")
        print(f"  Survivors   : {SimConfig.TOTAL_SURVIVORS}")
        print(f"  Obstacles   : {len(self.obstacles)}")
        print(f"  Scan range  : {SimConfig.SCAN_RANGE}m")
        print(f"  FOV         : {SimConfig.FOV_ANGLE_DEG}° @ {SimConfig.DETECTION_RANGE}m")
        print(f"  Timestamp   : {self._run_ts}")
        print()

    # ── connection ───────────────────────────────────────────────

    def connect(self):
        print("  [INIT] Connecting to AirSim (make sure UE4/AirSim is running)...")
        MAX_RETRIES = 5
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self.client = airsim.MultirotorClient()
                self.client.confirmConnection()
                break
            except Exception as e:
                if attempt == MAX_RETRIES:
                    print("\n" + "!"*60)
                    print("  [ERROR] Cannot connect to AirSim after "
                          f"{MAX_RETRIES} attempts.")
                    print("\n  ► Make sure you have done ALL of the following:")
                    print("    1. Opened your Unreal Engine 4 project with the")
                    print("       AirSim plugin enabled.")
                    print("    2. Pressed  PLAY  inside the UE4 editor (or")
                    print("       launched a packaged binary).")
                    print("    3. Waited until the simulation world has fully")
                    print("       loaded before running this script.")
                    print("    4. Confirmed that settings.json uses:")
                    print('       "SimMode": "Multirotor"')
                    print("\n  ► If AirSim is on a remote machine, set the IP:")
                    print('       client = airsim.MultirotorClient(ip="<HOST>")')
                    print("!"*60 + "\n")
                    raise ConnectionError(
                        "AirSim is not reachable. Start UE4 + AirSim first."
                    ) from e
                print(f"  [INIT] Attempt {attempt}/{MAX_RETRIES} failed — "
                      f"retrying in 3s...")
                time.sleep(3)

        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self._connected = True
        print("  [INIT] ✓ Connected and armed.")

        # Camera & proximity
        img_dir = os.path.join(SimConfig.IMG_DIR, self._run_ts)
        self.camera = DroneCamera(self.client, img_dir)
        self.prox   = ProximitySensor(self.client, self.obstacles)

        # Configure wind (AirSim environment API)
        try:
            wind = airsim.Vector3r(
                random.uniform(-0.5, 0.5),
                random.uniform(-0.5, 0.5),
                0.0
            )
            self.client.simSetWind(wind)
            print(f"  [INIT] ✓ Wind set: ({wind.x_val:.2f}, {wind.y_val:.2f}) m/s")
        except Exception:
            print("  [INIT]   Wind API not available (AirSim version may differ).")

    # ── takeoff ──────────────────────────────────────────────────

    def takeoff(self):
        print("\n  [TAKEOFF] Taking off...")
        self.client.takeoffAsync().join()
        # Ascend to cruise altitude
        self.client.moveToZAsync(-SimConfig.FLIGHT_ALTITUDE, 2.0).join()
        print(f"  [TAKEOFF] ✓ Airborne at {SimConfig.FLIGHT_ALTITUDE}m AGL")
        time.sleep(1.0)

    # ── land ─────────────────────────────────────────────────────

    def land(self):
        if not self._connected:
            return   # never connected — nothing to land
        print("\n  [LAND] Returning to home and landing...")
        try:
            home = WAYPOINTS[-1]["pos"]
            self.client.moveToPositionAsync(
                home[0], home[1], home[2],
                velocity=2.0
            ).join()
            self.client.landAsync().join()
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            print("  [LAND] ✓ Landed safely.")
        except Exception as e:
            print(f"  [LAND] Warning: clean landing failed ({e})")

    # ── get drone state ──────────────────────────────────────────

    def _get_state(self):
        state = self.client.getMultirotorState()
        pos   = state.kinematics_estimated.position
        vel   = state.kinematics_estimated.linear_velocity
        ori   = state.kinematics_estimated.orientation

        x = pos.x_val
        y = pos.y_val
        z = pos.z_val

        # Yaw from quaternion
        q = ori
        siny_cosp = 2 * (q.w_val * q.z_val + q.x_val * q.y_val)
        cosy_cosp = 1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        vx = vel.x_val
        vy = vel.y_val
        vz = vel.z_val

        return x, y, z, yaw, vx, vy, vz

    # ── waypoint navigation ──────────────────────────────────────

    def _navigate_to_waypoint(self, wp_idx: int):
        wp      = WAYPOINTS[wp_idx]
        target  = wp["pos"]
        wp_name = wp["name"]
        print(f"\n  ► WP {wp_idx+1}/{len(WAYPOINTS)}: {wp_name}  "
              f"→ ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})")

        wp_start = time.time()
        dist_wp  = 0.0
        prev_pos = None

        while self.running:
            x, y, z, yaw, vx, vy, vz = self._get_state()

            # Metrics
            self.metrics.update_position(x, y, z)
            self.metrics.update_speed(vx, vy, vz)
            if prev_pos:
                dist_wp += math.sqrt((x-prev_pos[0])**2 +
                                     (y-prev_pos[1])**2 +
                                     (z-prev_pos[2])**2)
            prev_pos = (x, y, z)

            # Trail
            self.trail.update(x, y, z, self.frame)

            # Wind update every 60 frames
            if self.frame % 60 == 0:
                self.navigator.update_wind()

            # Compute velocity setpoint via potential fields
            vel_cmd = self.navigator.compute_velocity(
                (x, y, z), target
            )

            # Obstacle status
            status, min_d = self.navigator.obstacle_status((x, y, z))
            if min_d < 2.0:
                self.metrics.near_misses += 1
                print(f"  ⚠ NEAR-MISS  dist={min_d:.2f}m")
            if min_d < 1.0:
                self.metrics.collisions += 1
                print(f"  💥 COLLISION  dist={min_d:.2f}m")
            if status != "SAFE" and min_d > 2.0:
                self.metrics.avoidances += 1

            # Apply velocity command
            self.client.moveByVelocityAsync(
                vel_cmd[0], vel_cmd[1], vel_cmd[2],
                duration=0.1
            )

            # Camera capture
            self.camera.try_capture()
            self.metrics.images_captured = self.camera.captured

            # Survivor detection
            newly = self.detector.check((x, y), yaw, self.survivors)
            for s, d in newly:
                self.metrics.add_survivor(s, d)
                print(f"\n  🚨 SURVIVOR DETECTED")
                print(f"     Name     : {s['name']}")
                print(f"     Coords   : ({s['pos'][0]:.2f}, {s['pos'][1]:.2f})")
                print(f"     Distance : {d:.1f}m")

            # Move undetected survivors
            if self.frame % 5 == 0:
                self.detector.move_survivors(self.survivors)

            # Telemetry every 120 frames (~12s)
            if self.frame % 120 == 0:
                self.metrics.print_telemetry(wp_idx, status)

            # Check if waypoint reached
            d_to_wp = math.sqrt((x-target[0])**2 +
                                 (y-target[1])**2 +
                                 (z-target[2])**2)
            if d_to_wp < SimConfig.WAYPOINT_THRESHOLD:
                elapsed_wp = time.time() - wp_start
                self.metrics.waypoint_reached(wp_idx)
                print(f"  ✓ WP {wp_idx+1} reached in {elapsed_wp:.1f}s  "
                      f"(dist flown: {dist_wp:.1f}m)")
                break

            self.frame += 1
            time.sleep(0.1)   # ~10 Hz control loop

    # ── main mission loop ────────────────────────────────────────

    def run_mission(self):
        self.running = True
        self.metrics.start_time = time.time()

        for wp_idx in range(len(WAYPOINTS)):
            if not self.running:
                break
            self._navigate_to_waypoint(wp_idx)

        self.metrics.end_time = time.time()
        self.running = False

    # ── full run ─────────────────────────────────────────────────

    def run(self):
        try:
            self.connect()       # raises ConnectionError if AirSim is down
            self.takeoff()
            self.run_mission()
        except ConnectionError as e:
            print(f"\n  [FATAL] {e}")
            return               # exit cleanly — nothing to land or save
        except KeyboardInterrupt:
            print("\n  [ABORT] Mission aborted by user.")
            self.running = False
        finally:
            if self._connected:
                self.land()
                self._save_results()
                self.metrics.print_final_report()
                self.metrics.print_rescue_report()

    # ── save results ─────────────────────────────────────────────

    def _save_results(self):
        os.makedirs(SimConfig.LOG_DIR, exist_ok=True)
        os.makedirs(SimConfig.CSV_DIR, exist_ok=True)

        base = f"{self.scenario}_{self._run_ts}"

        json_path = os.path.join(SimConfig.LOG_DIR, f"{base}.json")
        csv_path  = os.path.join(SimConfig.CSV_DIR,  f"{base}.csv")

        self.metrics.save_json(json_path)
        self.metrics.save_csv(csv_path)

        # Save flight trail
        trail_path = os.path.join(SimConfig.LOG_DIR, f"{base}_trail.json")
        with open(trail_path, "w") as f:
            json.dump(self.trail.as_list(), f)

        print(f"\n  [SAVE] Results  → {json_path}")
        print(f"  [SAVE] CSV      → {csv_path}")
        print(f"  [SAVE] Trail    → {trail_path}")
        print(f"  [SAVE] Images   → {SimConfig.IMG_DIR}/{self._run_ts}/")


# ─────────────────────────────────────────────────────────────────
#  BATCH EXPERIMENT RUNNER  (headless, 150+ trials)
# ─────────────────────────────────────────────────────────────────

class BatchExperiment:
    """
    Runs multiple trials (headless) per scenario, aggregates stats,
    and exports a combined CSV for statistical analysis.
    """

    def __init__(self, trials_per_scenario: int = 50):
        self.trials = trials_per_scenario
        self.results: list = []

    def run(self):
        for scenario in SimConfig.SCENARIOS:
            print(f"\n{'═'*50}")
            print(f"  BATCH: {scenario.upper()}  ×{self.trials} trials")
            print(f"{'═'*50}")
            for t in range(self.trials):
                print(f"  Trial {t+1}/{self.trials} ...", end="\r")
                sim = DisasterDroneSim(scenario)
                try:
                    sim.connect()
                    sim.takeoff()
                    sim.run_mission()
                finally:
                    sim.land()
                    m = sim.metrics
                    self.results.append({
                        "scenario":       scenario,
                        "trial":          t + 1,
                        "duration_s":     round(m.elapsed, 2),
                        "distance_m":     round(m.path_length, 2),
                        "avg_speed":      round(m.avg_speed, 2),
                        "collisions":     m.collisions,
                        "near_misses":    m.near_misses,
                        "avoidances":     m.avoidances,
                        "safety_score":   round(m.safety_score, 2),
                        "survivors":      len(m.survivors_found),
                        "detection_rate": round(m.detection_rate, 1),
                        "images":         m.images_captured,
                        "efficiency":     round(m.path_efficiency, 1),
                    })
            print()
        self._save_batch()
        self._print_summary()

    def _save_batch(self):
        os.makedirs(SimConfig.CSV_DIR, exist_ok=True)
        path = os.path.join(
            SimConfig.CSV_DIR,
            f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        if not self.results:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.results[0].keys())
            w.writeheader()
            w.writerows(self.results)
        print(f"\n  [BATCH] Saved → {path}")

    def _print_summary(self):
        print("\n" + "═"*60)
        print("  BATCH SUMMARY  (mean ± std)")
        print("═"*60)
        for scenario in SimConfig.SCENARIOS:
            rows = [r for r in self.results if r["scenario"] == scenario]
            if not rows:
                continue
            def _stat(key):
                vals = [r[key] for r in rows]
                return np.mean(vals), np.std(vals)
            dur_m,  dur_s  = _stat("duration_s")
            det_m,  det_s  = _stat("detection_rate")
            saf_m,  saf_s  = _stat("safety_score")
            eff_m,  eff_s  = _stat("efficiency")
            print(f"\n  {scenario.upper():<12}")
            print(f"    Duration      : {dur_m:.1f} ± {dur_s:.1f}s")
            print(f"    Detection     : {det_m:.1f} ± {det_s:.1f}%")
            print(f"    Safety Score  : {saf_m:.1f} ± {saf_s:.1f}")
            print(f"    Path Eff.     : {eff_m:.1f} ± {eff_s:.1f}%")
        print("═"*60)


# ─────────────────────────────────────────────────────────────────
#  AIRSIM settings.json helper (prints recommended config)
# ─────────────────────────────────────────────────────────────────

AIRSIM_SETTINGS = {
    "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
    "SettingsVersion": 2.0,
    "SimMode": "Multirotor",
    "ClockSpeed": 1.0,
    "Vehicles": {
        "Drone1": {
            "VehicleType": "SimpleFlight",
            "X": 0, "Y": 0, "Z": 0,
            "Cameras": {
                "front_center": {
                    "CaptureSettings": [
                        {
                            "ImageType": 0,
                            "Width": SimConfig.CAM_WIDTH,
                            "Height": SimConfig.CAM_HEIGHT,
                            "FOV_Degrees": 90
                        }
                    ],
                    "X": 0.30, "Y": 0, "Z": -0.10,
                    "Pitch": 0, "Roll": 0, "Yaw": 0
                }
            },
            "Sensors": {
                "DistanceSensor": {
                    "SensorType": 5,
                    "Enabled": True,
                    "X": 0, "Y": 0, "Z": -0.10,
                    "Yaw": 0, "Pitch": -90, "Roll": 0,
                    "DrawDebugPoints": True,
                    "ReportFrequency": 50
                }
            }
        }
    },
    "Wind": {"X": 0, "Y": 0, "Z": 0}
}


def print_settings():
    print("\n  ─── Recommended ~/Documents/AirSim/settings.json ───")
    print(json.dumps(AIRSIM_SETTINGS, indent=2))
    print("  ────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Autonomous Drone SAR — AirSim"
    )
    parser.add_argument(
        "--scenario",
        choices=SimConfig.SCENARIOS,
        default="earthquake",
        help="Disaster zone scenario (default: earthquake)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch experiments (50 trials × 3 scenarios)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Trials per scenario in batch mode (default: 50)"
    )
    parser.add_argument(
        "--settings",
        action="store_true",
        help="Print recommended AirSim settings.json and exit"
    )
    args = parser.parse_args()

    if args.settings:
        print_settings()
        sys.exit(0)

    if args.batch:
        BatchExperiment(trials_per_scenario=args.trials).run()
    else:
        DisasterDroneSim(scenario=args.scenario).run()