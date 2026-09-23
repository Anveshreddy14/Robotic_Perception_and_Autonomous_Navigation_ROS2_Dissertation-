#!/usr/bin/env python3
"""Headless ROS 2 AMR perception and navigation evaluation for Project 8.

This experiment intentionally avoids claiming Gazebo, RViz, physical robot
deployment, or Nav2 action-server execution. It uses a real ROS 2 Humble
runtime through rclpy and standard ROS 2 messages, then evaluates controlled
occupancy-grid navigation scenarios with deterministic sampling.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


Cell = Tuple[int, int]
Grid = List[List[int]]

SEED = 20260912
GRID_W = 58
GRID_H = 58
TRIALS_PER_CONDITION = 30
PLANNERS = (
    ("dijkstra_navfn_style", 0.0),
    ("a_star", 1.0),
    ("weighted_a_star", 1.35),
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_DIR / "outputs"
TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR / "logs"


@dataclass(frozen=True)
class Condition:
    name: str
    obstacle_p: float
    fp_rate: float
    fn_rate: float
    max_range: int
    drift_cells: int
    dynamic_p: float
    blocked_p: float


CONDITIONS = (
    Condition("baseline", 0.095, 0.003, 0.010, 46, 0, 0.00, 0.00),
    Condition("cluttered_static", 0.155, 0.006, 0.020, 44, 0, 0.04, 0.02),
    Condition("sensor_noise", 0.120, 0.030, 0.070, 42, 0, 0.02, 0.00),
    Condition("range_limited", 0.125, 0.010, 0.045, 24, 0, 0.03, 0.00),
    Condition("localisation_drift", 0.125, 0.012, 0.045, 40, 2, 0.03, 0.00),
    Condition("dynamic_intrusion", 0.120, 0.010, 0.030, 40, 0, 0.38, 0.04),
)


class RosProbe(Node):
    """Minimal ROS 2 pub/sub probe used inside the experiment loop."""

    def __init__(self) -> None:
        super().__init__("amr_eval_probe")
        self.scan_pub = self.create_publisher(LaserScan, "/amr_eval/scan", 10)
        self.map_pub = self.create_publisher(OccupancyGrid, "/amr_eval/occupancy_grid", 10)
        self.scan_count = 0
        self.map_count = 0
        self.create_subscription(LaserScan, "/amr_eval/scan", self._scan_cb, 10)
        self.create_subscription(OccupancyGrid, "/amr_eval/occupancy_grid", self._map_cb, 10)

    def _scan_cb(self, _msg: LaserScan) -> None:
        self.scan_count += 1

    def _map_cb(self, _msg: OccupancyGrid) -> None:
        self.map_count += 1


def ensure_dirs() -> None:
    for path in (TABLE_DIR, FIGURE_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def neighbours(cell: Cell) -> Iterable[Tuple[Cell, float]]:
    x, y = cell
    for dx, dy, cost in (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2)),
    ):
        yield (x + dx, y + dy), cost


def in_bounds(cell: Cell) -> bool:
    x, y = cell
    return 0 <= x < GRID_W and 0 <= y < GRID_H


def is_free(grid: Grid, cell: Cell) -> bool:
    x, y = cell
    return in_bounds(cell) and grid[y][x] == 0


def heuristic(a: Cell, b: Cell) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)


def reconstruct(came_from: Dict[Cell, Cell], current: Cell) -> List[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def plan_path(grid: Grid, start: Cell, goal: Cell, planner_weight: float) -> Tuple[Optional[List[Cell]], int]:
    if not is_free(grid, start) or not is_free(grid, goal):
        return None, 0

    open_set: List[Tuple[float, int, Cell]] = []
    heapq.heappush(open_set, (0.0, 0, start))
    came_from: Dict[Cell, Cell] = {}
    g_score: Dict[Cell, float] = {start: 0.0}
    seen = set()
    counter = 0
    expansions = 0

    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current in seen:
            continue
        seen.add(current)
        expansions += 1
        if current == goal:
            return reconstruct(came_from, current), expansions

        for nxt, step_cost in neighbours(current):
            if not is_free(grid, nxt):
                continue
            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                h = 0.0 if planner_weight == 0.0 else heuristic(nxt, goal) * planner_weight
                counter += 1
                heapq.heappush(open_set, (tentative_g + h, counter, nxt))

    return None, expansions


def path_length(path: Optional[Sequence[Cell]]) -> Optional[float]:
    if not path or len(path) < 2:
        return None
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        total += math.hypot(a[0] - b[0], a[1] - b[1])
    return total


def make_empty_grid() -> Grid:
    return [[0 for _ in range(GRID_W)] for _ in range(GRID_H)]


def copy_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def set_cell(grid: Grid, cell: Cell, value: int) -> None:
    if in_bounds(cell):
        x, y = cell
        grid[y][x] = value


def carve_corridor(grid: Grid, start: Cell, goal: Cell, rng: random.Random) -> None:
    x, y = start
    gx, gy = goal
    while (x, y) != (gx, gy):
        for yy in range(max(1, y - 1), min(GRID_H - 1, y + 2)):
            for xx in range(max(1, x - 1), min(GRID_W - 1, x + 2)):
                grid[yy][xx] = 0
        if rng.random() < 0.52:
            x += 1 if gx > x else -1 if gx < x else 0
        else:
            y += 1 if gy > y else -1 if gy < y else 0


def generate_grid(cond: Condition, rng: random.Random) -> Tuple[Grid, Cell, Cell, Optional[List[Cell]], bool]:
    start = (4, 4)
    goal = (GRID_W - 5, GRID_H - 5)
    force_blocked = rng.random() < cond.blocked_p
    for _ in range(80):
        grid = make_empty_grid()
        for x in range(GRID_W):
            grid[0][x] = grid[GRID_H - 1][x] = 1
        for y in range(GRID_H):
            grid[y][0] = grid[y][GRID_W - 1] = 1

        for y in range(1, GRID_H - 1):
            for x in range(1, GRID_W - 1):
                if rng.random() < cond.obstacle_p:
                    grid[y][x] = 1

        for _ in range(9):
            rw = rng.randint(3, 9)
            rh = rng.randint(2, 7)
            rx = rng.randint(2, GRID_W - rw - 2)
            ry = rng.randint(2, GRID_H - rh - 2)
            if rng.random() < cond.obstacle_p * 3.8:
                for y in range(ry, ry + rh):
                    for x in range(rx, rx + rw):
                        grid[y][x] = 1

        carve_corridor(grid, start, goal, rng)
        if force_blocked:
            bx = rng.randint(18, GRID_W - 18)
            for y in range(1, GRID_H - 1):
                grid[y][bx] = 1

        for cell in (start, goal):
            for yy in range(cell[1] - 2, cell[1] + 3):
                for xx in range(cell[0] - 2, cell[0] + 3):
                    set_cell(grid, (xx, yy), 0)

        truth_path, _ = plan_path(grid, start, goal, 1.0)
        if force_blocked or truth_path:
            return grid, start, goal, truth_path, force_blocked

    return grid, start, goal, truth_path, force_blocked


def apply_perception_model(grid: Grid, cond: Condition, rng: random.Random, origin: Cell) -> Grid:
    perceived = copy_grid(grid)
    ox, oy = origin
    for y in range(1, GRID_H - 1):
        for x in range(1, GRID_W - 1):
            dist = math.hypot(x - ox, y - oy)
            fp = cond.fp_rate
            fn = cond.fn_rate
            if dist > cond.max_range:
                fn += 0.08
                fp += 0.006
            if cond.drift_cells:
                fn += 0.015
                fp += 0.010
            if grid[y][x] and rng.random() < fn:
                perceived[y][x] = 0
            elif not grid[y][x] and rng.random() < fp:
                perceived[y][x] = 1

    for cell in ((4, 4), (GRID_W - 5, GRID_H - 5)):
        for yy in range(cell[1] - 2, cell[1] + 3):
            for xx in range(cell[0] - 2, cell[0] + 3):
                set_cell(perceived, (xx, yy), 0)
    return perceived


def add_dynamic_intrusion(grid: Grid, path: Optional[List[Cell]], cond: Condition, rng: random.Random) -> Tuple[Grid, bool]:
    executed = copy_grid(grid)
    if not path or rng.random() >= cond.dynamic_p or len(path) < 12:
        return executed, False
    idx = rng.randint(max(3, len(path) // 4), min(len(path) - 4, (len(path) * 3) // 4))
    cx, cy = path[idx]
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if rng.random() < 0.8:
                set_cell(executed, (cx + dx, cy + dy), 1)
    return executed, True


def grid_to_occ_msg(grid: Grid) -> OccupancyGrid:
    msg = OccupancyGrid()
    msg.header.frame_id = "map"
    msg.info.width = GRID_W
    msg.info.height = GRID_H
    msg.info.resolution = 0.20
    msg.info.origin.position.x = 0.0
    msg.info.origin.position.y = 0.0
    msg.info.origin.orientation.w = 1.0
    msg.data = [100 if grid[y][x] else 0 for y in range(GRID_H) for x in range(GRID_W)]
    return msg


def raycast(grid: Grid, origin: Cell, angle: float, max_range: int) -> float:
    ox, oy = origin
    step = 0.25
    r = step
    while r <= max_range:
        x = int(round(ox + math.cos(angle) * r))
        y = int(round(oy + math.sin(angle) * r))
        if not in_bounds((x, y)):
            return float(r)
        if grid[y][x] == 1:
            return float(r)
        r += step
    return float(max_range)


def grid_to_scan_msg(grid: Grid, origin: Cell, cond: Condition, rng: random.Random) -> LaserScan:
    msg = LaserScan()
    msg.header.frame_id = "base_laser"
    msg.angle_min = -math.pi
    msg.angle_max = math.pi
    msg.angle_increment = (2 * math.pi) / 72
    msg.range_min = 0.05
    msg.range_max = float(cond.max_range)
    msg.ranges = []
    for i in range(72):
        angle = msg.angle_min + i * msg.angle_increment
        value = raycast(grid, origin, angle, cond.max_range)
        value += rng.gauss(0.0, 0.10 + cond.fp_rate * 5)
        msg.ranges.append(max(msg.range_min, min(msg.range_max, value)))
    return msg


def perception_scores(truth: Grid, perceived: Grid) -> Dict[str, float]:
    tp = fp = fn = tn = 0
    for y in range(1, GRID_H - 1):
        for x in range(1, GRID_W - 1):
            gt = truth[y][x] == 1
            pd = perceived[y][x] == 1
            if gt and pd:
                tp += 1
            elif not gt and pd:
                fp += 1
            elif gt and not pd:
                fn += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def collision_count(path: Optional[List[Cell]], grid: Grid) -> int:
    if not path:
        return 0
    return sum(1 for x, y in path if grid[y][x] == 1)


def distance_transform(grid: Grid) -> List[List[float]]:
    obstacle_cells = [(x, y) for y in range(GRID_H) for x in range(GRID_W) if grid[y][x] == 1]
    dist = [[float("inf") for _ in range(GRID_W)] for _ in range(GRID_H)]
    for y in range(GRID_H):
        for x in range(GRID_W):
            if grid[y][x] == 1:
                dist[y][x] = 0.0
                continue
            best = float("inf")
            for ox, oy in obstacle_cells:
                d = abs(ox - x) + abs(oy - y)
                if d < best:
                    best = d
                    if best <= 1:
                        break
            dist[y][x] = best
    return dist


def path_clearance(path: Optional[List[Cell]], dist: List[List[float]]) -> Optional[float]:
    if not path:
        return None
    values = [dist[y][x] for x, y in path]
    return percentile(values, 10)


def decide_state(
    path: Optional[List[Cell]],
    collisions: int,
    length_ratio: Optional[float],
    clearance: Optional[float],
    dynamic_inserted: bool,
    cond: Condition,
) -> str:
    if not path or collisions > 0:
        return "STOP"
    if dynamic_inserted or cond.drift_cells > 0:
        return "RECOVERY"
    if (clearance is not None and clearance < 1.15) or (length_ratio is not None and length_ratio > 1.24):
        return "SLOW"
    if cond.name in {"sensor_noise", "range_limited"}:
        return "SLOW"
    return "PROCEED"


def classify_oracle(
    truth_path: Optional[List[Cell]],
    truth_ratio: Optional[float],
    truth_clearance: Optional[float],
    truth_collisions: int,
    force_blocked: bool,
    dynamic_inserted: bool,
    cond: Condition,
) -> str:
    if force_blocked or not truth_path:
        return 'STOP'
    if truth_collisions > 0 and cond.name == 'dynamic_intrusion':
        return 'STOP'
    if dynamic_inserted or cond.drift_cells > 0:
        return 'RECOVERY'
    if (truth_clearance is not None and truth_clearance < 1.15) or (truth_ratio is not None and truth_ratio > 1.24):
        return 'SLOW'
    if cond.name in {'sensor_noise', 'range_limited'}:
        return 'SLOW'
    return 'PROCEED'


def summarize(rows: List[Dict[str, object]], keys: Sequence[str]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)

    out = []
    for key_values, group in sorted(grouped.items()):
        feasible = [r for r in group if r["oracle_state"] != "STOP"]
        success_rate = sum(int(r["success"]) for r in feasible) / len(feasible) if feasible else 0.0
        decision_accuracy = sum(int(r["decision_correct"]) for r in group) / len(group)
        latency = [float(r["latency_ms"]) for r in group]
        path_ratios = [float(r["path_length_ratio"]) for r in group if r["path_length_ratio"] != ""]
        clearance = [float(r["clearance_cells"]) for r in group if r["clearance_cells"] != ""]
        item = {k: v for k, v in zip(keys, key_values)}
        item.update(
            {
                "trials": len(group),
                "feasible_trials": len(feasible),
                "success_rate": success_rate,
                "decision_accuracy": decision_accuracy,
                "mean_latency_ms": statistics.mean(latency),
                "median_latency_ms": statistics.median(latency),
                "std_latency_ms": statistics.pstdev(latency) if len(latency) > 1 else 0.0,
                "p95_latency_ms": percentile(latency, 95),
                "fps": 1000.0 / statistics.mean(latency) if statistics.mean(latency) > 0 else 0.0,
                "mean_path_length_ratio": statistics.mean(path_ratios) if path_ratios else "",
                "mean_clearance_cells": statistics.mean(clearance) if clearance else "",
                "mean_expansions": statistics.mean(float(r["expansions"]) for r in group),
            }
        )
        out.append(item)
    return out


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[int(idx)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


def bootstrap_ci(values: Sequence[float], rng: random.Random, iterations: int = 800) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    means = []
    n = len(values)
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    return percentile(means, 2.5), percentile(means, 97.5)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            clean = {}
            for key, value in row.items():
                if isinstance(value, float):
                    clean[key] = f"{value:.6f}"
                else:
                    clean[key] = value
            writer.writerow(clean)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def command_output(cmd: Sequence[str]) -> Dict[str, object]:
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        return {"cmd": list(cmd), "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"cmd": list(cmd), "error": str(exc)}


def publish_ros_probe(probe: RosProbe, grid: Grid, origin: Cell, cond: Condition, rng: random.Random) -> None:
    probe.map_pub.publish(grid_to_occ_msg(grid))
    probe.scan_pub.publish(grid_to_scan_msg(grid, origin, cond, rng))
    for _ in range(4):
        rclpy.spin_once(probe, timeout_sec=0.03)


def add_runtime_log() -> None:
    payload = {
        "experiment_seed": SEED,
        "python": sys.version,
        "platform": sys.platform,
        "ros_distro": os.environ.get("ROS_DISTRO", ""),
        "ros_version": os.environ.get("ROS_VERSION", ""),
        "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION", ""),
        "commands": [
            command_output(["ros2", "pkg", "list"]),
            command_output(["ros2", "interface", "show", "nav_msgs/msg/OccupancyGrid"]),
            command_output(["ros2", "interface", "show", "sensor_msgs/msg/LaserScan"]),
        ],
    }
    write_json(LOG_DIR / "ros2_install_verification.json", payload)


def run_experiment() -> Dict[str, object]:
    ensure_dirs()
    rng = random.Random(SEED)
    rclpy.init(args=None)
    probe = RosProbe()
    trial_rows: List[Dict[str, object]] = []
    perception_rows: List[Dict[str, object]] = []
    examples: Dict[str, object] = {}

    try:
        trial_id = 0
        for cond in CONDITIONS:
            for _ in range(TRIALS_PER_CONDITION):
                trial_id += 1
                truth_grid, start, goal, truth_path, force_blocked = generate_grid(cond, rng)
                perceived_grid = apply_perception_model(truth_grid, cond, rng, start)
                executed_grid, dynamic_inserted = add_dynamic_intrusion(truth_grid, truth_path, cond, rng)
                publish_ros_probe(probe, perceived_grid, start, cond, rng)

                truth_dist = distance_transform(executed_grid)
                truth_len = path_length(truth_path)
                straight = heuristic(start, goal)
                truth_ratio = (truth_len / straight) if truth_len else None
                truth_clearance = path_clearance(truth_path, truth_dist)
                truth_collisions = collision_count(truth_path, executed_grid)
                oracle_state = classify_oracle(
                    truth_path,
                    truth_ratio,
                    truth_clearance,
                    truth_collisions,
                    force_blocked,
                    dynamic_inserted,
                    cond,
                )

                ps = perception_scores(truth_grid, perceived_grid)
                perception_rows.append(
                    {
                        "trial_id": trial_id,
                        "condition": cond.name,
                        "precision": ps["precision"],
                        "recall": ps["recall"],
                        "f1": ps["f1"],
                        "tp": ps["tp"],
                        "fp": ps["fp"],
                        "fn": ps["fn"],
                        "tn": ps["tn"],
                    }
                )

                for planner_name, weight in PLANNERS:
                    start_time = time.perf_counter()
                    path, expansions = plan_path(perceived_grid, start, goal, weight)
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    collisions = collision_count(path, executed_grid)
                    plen = path_length(path)
                    ratio = (plen / truth_len) if plen and truth_len else None
                    clearance = path_clearance(path, truth_dist)
                    success = bool(path and oracle_state != "STOP" and collisions == 0)
                    predicted_state = decide_state(path, collisions, ratio, clearance, dynamic_inserted, cond)
                    row = {
                        "trial_id": trial_id,
                        "condition": cond.name,
                        "planner": planner_name,
                        "success": int(success),
                        "truth_path_available": int(bool(truth_path)),
                        "dynamic_inserted": int(dynamic_inserted),
                        "oracle_state": oracle_state,
                        "predicted_state": predicted_state,
                        "decision_correct": int(predicted_state == oracle_state),
                        "latency_ms": latency_ms,
                        "expansions": expansions,
                        "path_length": plen if plen is not None else "",
                        "truth_path_length": truth_len if truth_len is not None else "",
                        "path_length_ratio": ratio if ratio is not None else "",
                        "clearance_cells": clearance if clearance is not None else "",
                        "collisions": collisions,
                        "perception_precision": ps["precision"],
                        "perception_recall": ps["recall"],
                        "perception_f1": ps["f1"],
                    }
                    trial_rows.append(row)

                    if cond.name == "cluttered_static" and planner_name == "a_star" and "example" not in examples:
                        examples["example"] = {
                            "truth_grid": truth_grid,
                            "perceived_grid": perceived_grid,
                            "path": path,
                            "start": start,
                            "goal": goal,
                            "condition": cond.name,
                            "planner": planner_name,
                        }
    finally:
        ros_counts = {"scan_messages_received": probe.scan_count, "map_messages_received": probe.map_count}
        probe.destroy_node()
        rclpy.shutdown()

    overall = summarize(trial_rows, ["planner"])
    by_condition = summarize(trial_rows, ["planner", "condition"])
    decision_matrix = build_confusion_matrix(trial_rows)
    perception_summary = summarize_perception(perception_rows)
    ci_rows = confidence_rows(trial_rows, rng)
    tradeoff = build_tradeoff(overall)

    write_csv(TABLE_DIR / "trial_results.csv", trial_rows)
    write_csv(TABLE_DIR / "overall_metrics.csv", overall)
    write_csv(TABLE_DIR / "condition_metrics.csv", by_condition)
    write_csv(TABLE_DIR / "perception_metrics.csv", perception_summary)
    write_csv(TABLE_DIR / "decision_confusion_matrix.csv", decision_matrix)
    write_csv(TABLE_DIR / "statistical_ci.csv", ci_rows)
    write_csv(TABLE_DIR / "planner_tradeoff.csv", tradeoff)
    add_runtime_log()

    summary = {
        "seed": SEED,
        "grid_size": [GRID_W, GRID_H],
        "trials_per_condition": TRIALS_PER_CONDITION,
        "conditions": [c.__dict__ for c in CONDITIONS],
        "planners": [name for name, _ in PLANNERS],
        "ros_message_probe": ros_counts,
        "rows": {
            "trial_results": len(trial_rows),
            "perception_trials": len(perception_rows),
            "overall_metrics": len(overall),
            "condition_metrics": len(by_condition),
        },
        "best_success_planner": max(overall, key=lambda r: float(r["success_rate"]))["planner"],
        "fastest_planner": min(overall, key=lambda r: float(r["mean_latency_ms"]))["planner"],
    }
    write_json(LOG_DIR / "ros2_experiment_summary.json", summary)
    write_figures(overall, by_condition, perception_summary, decision_matrix, examples.get("example"))
    return summary


def summarize_perception(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)
    out = []
    for condition, group in sorted(grouped.items()):
        out.append(
            {
                "condition": condition,
                "trials": len(group),
                "mean_precision": statistics.mean(float(r["precision"]) for r in group),
                "mean_recall": statistics.mean(float(r["recall"]) for r in group),
                "mean_f1": statistics.mean(float(r["f1"]) for r in group),
                "total_tp": sum(int(r["tp"]) for r in group),
                "total_fp": sum(int(r["fp"]) for r in group),
                "total_fn": sum(int(r["fn"]) for r in group),
            }
        )
    return out


def build_confusion_matrix(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    labels = ["PROCEED", "SLOW", "RECOVERY", "STOP"]
    counts = Counter((r["oracle_state"], r["predicted_state"]) for r in rows)
    out = []
    for oracle in labels:
        row = {"oracle_state": oracle}
        for predicted in labels:
            row[predicted] = counts.get((oracle, predicted), 0)
        out.append(row)
    return out


def confidence_rows(rows: List[Dict[str, object]], rng: random.Random) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["planner"])].append(row)
    out = []
    for planner, group in sorted(grouped.items()):
        feasible = [r for r in group if r["oracle_state"] != "STOP"]
        success_values = [float(r["success"]) for r in feasible]
        latency_values = [float(r["latency_ms"]) for r in group]
        decision_values = [float(r["decision_correct"]) for r in group]
        for metric, values in (
            ("success_rate", success_values),
            ("mean_latency_ms", latency_values),
            ("decision_accuracy", decision_values),
        ):
            low, high = bootstrap_ci(values, rng)
            out.append(
                {
                    "planner": planner,
                    "metric": metric,
                    "n": len(values),
                    "mean": statistics.mean(values) if values else 0.0,
                    "ci95_low": low,
                    "ci95_high": high,
                    "method": "non-parametric bootstrap, 800 resamples, fixed seed",
                }
            )
    return out


def build_tradeoff(overall: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    for row in overall:
        out.append(
            {
                "planner": row["planner"],
                "success_rate": row["success_rate"],
                "decision_accuracy": row["decision_accuracy"],
                "mean_latency_ms": row["mean_latency_ms"],
                "p95_latency_ms": row["p95_latency_ms"],
                "fps": row["fps"],
                "mean_path_length_ratio": row["mean_path_length_ratio"],
            }
        )
    return out


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#FFFFFF"/>\n'
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1F2933}'
        '.title{font-size:22px;font-weight:700}.label{font-size:13px}'
        '.small{font-size:11px;fill:#52606D}.axis{stroke:#BCCCDC;stroke-width:1}'
        '.navy{fill:#183B56}.blue{fill:#2F80ED}.teal{fill:#56B4A9}'
        '.pale{fill:#EAF2F8}.grid{stroke:#BCCCDC;stroke-width:1}</style>\n'
    )


def save_svg(path: Path, body: str, width: int = 920, height: int = 560) -> None:
    path.write_text(svg_header(width, height) + body + "</svg>\n", encoding="utf-8")


def write_figures(
    overall: List[Dict[str, object]],
    by_condition: List[Dict[str, object]],
    perception: List[Dict[str, object]],
    matrix: List[Dict[str, object]],
    example: Optional[Dict[str, object]],
) -> None:
    write_architecture_svg()
    write_success_condition_svg(by_condition)
    write_latency_tradeoff_svg(overall)
    write_perception_svg(perception)
    write_confusion_svg(matrix)
    if example:
        write_example_map_svg(example)


def write_architecture_svg() -> None:
    labels = [
        ("ROS 2 sensor topics", "LaserScan and occupancy-grid messages"),
        ("Perception layer", "Noise-aware obstacle representation"),
        ("Planning layer", "Dijkstra/NavFn-style, A*, weighted A*"),
        ("Decision layer", "PROCEED, SLOW, RECOVERY, STOP"),
        ("Evidence export", "CSV metrics, logs and dissertation figures"),
    ]
    body = '<text x="40" y="44" class="title">ROS 2 AMR Evaluation Architecture</text>\n'
    x = 42
    for i, (h, sub) in enumerate(labels):
        y = 92 + i * 84
        body += f'<rect x="{x}" y="{y}" width="710" height="54" rx="6" fill="#EAF2F8" stroke="#BCCCDC"/>\n'
        body += f'<text x="{x + 22}" y="{y + 23}" class="label" font-weight="700">{h}</text>\n'
        body += f'<text x="{x + 22}" y="{y + 42}" class="small">{sub}</text>\n'
        if i < len(labels) - 1:
            body += f'<line x1="{x + 355}" y1="{y + 54}" x2="{x + 355}" y2="{y + 82}" stroke="#486581" stroke-width="1.5"/>\n'
            body += f'<path d="M{x + 350},{y + 76} L{x + 355},{y + 84} L{x + 360},{y + 76}" fill="none" stroke="#486581" stroke-width="1.5"/>\n'
    body += '<text x="770" y="130" class="small">Headless runtime: ROS 2 Humble</text>\n'
    body += '<text x="770" y="154" class="small">No physical robot claim</text>\n'
    body += '<text x="770" y="178" class="small">No Gazebo claim</text>\n'
    save_svg(FIGURE_DIR / "figure_1_ros2_evaluation_architecture.svg", body)


def write_success_condition_svg(rows: List[Dict[str, object]]) -> None:
    planners = [p[0] for p in PLANNERS]
    conditions = [c.name for c in CONDITIONS]
    lookup = {(r["planner"], r["condition"]): float(r["success_rate"]) for r in rows}
    width, height = 960, 540
    left, top, chart_w, chart_h = 86, 82, 800, 340
    body = '<text x="40" y="44" class="title">Navigation Success by Condition</text>\n'
    body += f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" class="axis"/>\n'
    body += f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" class="axis"/>\n'
    colors = {"dijkstra_navfn_style": "#183B56", "a_star": "#2F80ED", "weighted_a_star": "#56B4A9"}
    group_w = chart_w / len(conditions)
    bar_w = group_w / 5
    for i, cond in enumerate(conditions):
        gx = left + i * group_w + 16
        body += f'<text x="{gx}" y="{top + chart_h + 35}" class="small" transform="rotate(32 {gx},{top + chart_h + 35})">{cond}</text>\n'
        for j, planner in enumerate(planners):
            val = lookup.get((planner, cond), 0.0)
            h = val * chart_h
            x = gx + j * bar_w
            y = top + chart_h - h
            body += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 3:.1f}" height="{h:.1f}" fill="{colors[planner]}"/>\n'
    for t in range(0, 101, 20):
        y = top + chart_h - (t / 100) * chart_h
        body += f'<line x1="{left - 4}" y1="{y}" x2="{left}" y2="{y}" class="axis"/><text x="42" y="{y + 4}" class="small">{t}%</text>\n'
    legend_x = 630
    for i, planner in enumerate(planners):
        body += f'<rect x="{legend_x}" y="{32 + i * 20}" width="12" height="12" fill="{colors[planner]}"/>\n'
        body += f'<text x="{legend_x + 20}" y="{43 + i * 20}" class="small">{planner.replace("_", " ")}</text>\n'
    save_svg(FIGURE_DIR / "figure_2_success_by_condition.svg", body, width, height)


def write_latency_tradeoff_svg(rows: List[Dict[str, object]]) -> None:
    width, height = 860, 500
    left, top, chart_w, chart_h = 90, 82, 660, 300
    max_latency = max(float(r["mean_latency_ms"]) for r in rows) * 1.15
    body = '<text x="40" y="44" class="title">Accuracy-Latency Trade-off</text>\n'
    body += f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" class="axis"/>\n'
    body += f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" class="axis"/>\n'
    colors = ["#183B56", "#2F80ED", "#56B4A9"]
    for i, row in enumerate(rows):
        x = left + (float(row["mean_latency_ms"]) / max_latency) * chart_w
        y = top + chart_h - float(row["success_rate"]) * chart_h
        body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{colors[i]}"/>\n'
        body += f'<text x="{x + 14:.1f}" y="{y + 5:.1f}" class="label">{str(row["planner"]).replace("_", " ")}</text>\n'
    body += f'<text x="{left + chart_w / 2 - 65}" y="{top + chart_h + 48}" class="small">Mean planning latency (ms)</text>\n'
    body += f'<text x="28" y="{top + 10}" class="small" transform="rotate(-90 28,{top + 10})">Success rate</text>\n'
    save_svg(FIGURE_DIR / "figure_3_accuracy_latency_tradeoff.svg", body, width, height)


def write_perception_svg(rows: List[Dict[str, object]]) -> None:
    width, height = 930, 520
    left, top, chart_w, chart_h = 86, 82, 760, 310
    metrics = [("mean_precision", "#183B56"), ("mean_recall", "#2F80ED"), ("mean_f1", "#56B4A9")]
    body = '<text x="40" y="44" class="title">Perception Reliability by Condition</text>\n'
    body += f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" class="axis"/>\n'
    body += f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" class="axis"/>\n'
    group_w = chart_w / len(rows)
    bar_w = group_w / 5
    for i, row in enumerate(rows):
        gx = left + i * group_w + 18
        body += f'<text x="{gx}" y="{top + chart_h + 35}" class="small" transform="rotate(30 {gx},{top + chart_h + 35})">{row["condition"]}</text>\n'
        for j, (metric, color) in enumerate(metrics):
            val = float(row[metric])
            h = val * chart_h
            x = gx + j * bar_w
            y = top + chart_h - h
            body += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 3:.1f}" height="{h:.1f}" fill="{color}"/>\n'
    for t in range(0, 101, 20):
        y = top + chart_h - (t / 100) * chart_h
        body += f'<line x1="{left - 4}" y1="{y}" x2="{left}" y2="{y}" class="axis"/><text x="42" y="{y + 4}" class="small">{t}%</text>\n'
    for i, (metric, color) in enumerate(metrics):
        body += f'<rect x="{630}" y="{32 + i * 20}" width="12" height="12" fill="{color}"/>\n'
        body += f'<text x="{650}" y="{43 + i * 20}" class="small">{metric.replace("mean_", "").upper()}</text>\n'
    save_svg(FIGURE_DIR / "figure_4_perception_reliability.svg", body, width, height)


def write_confusion_svg(rows: List[Dict[str, object]]) -> None:
    labels = ["PROCEED", "SLOW", "RECOVERY", "STOP"]
    max_val = max(max(int(row[label]) for label in labels) for row in rows) or 1
    cell = 74
    left, top = 180, 104
    body = '<text x="40" y="44" class="title">Decision-State Confusion Matrix</text>\n'
    body += '<text x="344" y="82" class="small">Predicted state</text>\n'
    body += '<text x="52" y="250" class="small" transform="rotate(-90 52,250)">Oracle state</text>\n'
    for i, row in enumerate(rows):
        body += f'<text x="{left - 95}" y="{top + i * cell + 43}" class="small">{row["oracle_state"]}</text>\n'
        for j, label in enumerate(labels):
            val = int(row[label])
            intensity = 0.15 + 0.75 * (val / max_val)
            color = blend("#EAF2F8", "#183B56", intensity)
            x = left + j * cell
            y = top + i * cell
            body += f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" fill="{color}" stroke="#FFFFFF"/>\n'
            body += f'<text x="{x + 26}" y="{y + 42}" class="label" fill="#1F2933">{val}</text>\n'
            if i == 0:
                body += f'<text x="{x + 4}" y="{top - 16}" class="small" transform="rotate(-28 {x + 4},{top - 16})">{label}</text>\n'
    save_svg(FIGURE_DIR / "figure_5_decision_confusion_matrix.svg", body, 660, 470)


def blend(hex_a: str, hex_b: str, t: float) -> str:
    def parse(h: str) -> Tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    a = parse(hex_a)
    b = parse(hex_b)
    vals = [int(a[i] + (b[i] - a[i]) * t) for i in range(3)]
    return "#" + "".join(f"{v:02X}" for v in vals)


def write_example_map_svg(example: Dict[str, object]) -> None:
    grid: Grid = example["truth_grid"]  # type: ignore[assignment]
    path: Optional[List[Cell]] = example["path"]  # type: ignore[assignment]
    start: Cell = example["start"]  # type: ignore[assignment]
    goal: Cell = example["goal"]  # type: ignore[assignment]
    scale = 7
    margin = 42
    width = GRID_W * scale + margin * 2
    height = GRID_H * scale + 116
    body = '<text x="40" y="36" class="title">Representative Planned Path in Cluttered Map</text>\n'
    for y in range(GRID_H):
        for x in range(GRID_W):
            fill = "#F4F6F8" if grid[y][x] == 0 else "#486581"
            body += f'<rect x="{margin + x * scale}" y="{70 + y * scale}" width="{scale}" height="{scale}" fill="{fill}"/>\n'
    if path:
        pts = " ".join(f"{margin + x * scale + scale / 2:.1f},{70 + y * scale + scale / 2:.1f}" for x, y in path)
        body += f'<polyline points="{pts}" fill="none" stroke="#2F80ED" stroke-width="2.2"/>\n'
    for cell, color, label in ((start, "#56B4A9", "S"), (goal, "#183B56", "G")):
        cx = margin + cell[0] * scale + scale / 2
        cy = 70 + cell[1] * scale + scale / 2
        body += f'<circle cx="{cx}" cy="{cy}" r="7" fill="{color}"/><text x="{cx - 4}" y="{cy + 4}" font-size="10" fill="#FFFFFF">{label}</text>\n'
    body += '<text x="40" y="500" class="small">Figure generated from the reproducible experiment output, not from AI imagery.</text>\n'
    save_svg(FIGURE_DIR / "figure_6_representative_map_path.svg", body, width, height)


def main() -> int:
    summary = run_experiment()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



