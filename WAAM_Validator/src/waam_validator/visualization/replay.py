"""Compact self-contained Plotly replay generation."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import trimesh
from plotly.subplots import make_subplots

from ..config.models import Config
from ..constants import MODE_D
from ..errors import ComputationError, OutputWriteError
from ..models import CollisionEvent, CollisionSimulationResult, RobotTrajectory, TrajectorySet
from ..trajectory.interpolation import interpolate_all_states

_COLORS = ("#2f8fff", "#ff9d42", "#34d399")
_MAX_REPLAY_FRAMES = 2_000


def _rgba(hex_color: str, alpha: float) -> str:
    return (
        f"rgba({int(hex_color[1:3], 16)},{int(hex_color[3:5], 16)},"
        f"{int(hex_color[5:7], 16)},{alpha})"
    )


@dataclass(slots=True, frozen=True)
class ReplayGenerationStats:
    """Small generation summary persisted by the UI worker."""

    frame_count: int
    deposition_point_count: int


def _events(
    collision: CollisionSimulationResult | Sequence[CollisionEvent],
) -> Sequence[CollisionEvent]:
    return collision.events if isinstance(collision, CollisionSimulationResult) else collision


def _evenly_sample_times(values: Sequence[float], count: int) -> list[float]:
    """Select deterministic, time-distributed values without exceeding ``count``."""
    unique = sorted(set(values))
    if count <= 0 or not unique:
        return []
    if len(unique) <= count:
        return unique
    indices = np.linspace(0, len(unique) - 1, num=count, dtype=np.int64)
    return [unique[int(index)] for index in np.unique(indices)]


def _collision_keyframes(
    events: Sequence[CollisionEvent],
    makespan: float,
    budget: int,
    excluded: set[float],
) -> list[float]:
    """Prioritize worst instants, then distribute remaining event boundaries in time."""
    if budget <= 0 or not events:
        return []

    valid_events = [
        event for event in events if 0.0 <= float(event.minimum_distance_time_s) <= makespan
    ]
    selected: set[float] = set()
    minimum_times = sorted(
        {
            float(event.minimum_distance_time_s)
            for event in valid_events
            if float(event.minimum_distance_time_s) not in excluded
        }
    )
    if len(minimum_times) <= budget:
        selected.update(minimum_times)
    else:
        # Preserve half of the most severe collision instants and use the rest
        # for deterministic temporal coverage across the complete replay.
        severe_budget = max(1, budget // 2)
        for event in sorted(
            valid_events,
            key=lambda item: (
                float(item.minimum_safety_margin_mm),
                float(item.minimum_distance_time_s),
                item.collision_type,
                int(item.robot_a),
                int(item.robot_b),
                int(item.event_id),
            ),
        ):
            value = float(event.minimum_distance_time_s)
            if value in excluded or value in selected:
                continue
            selected.add(value)
            if len(selected) >= severe_budget:
                break
        remaining = budget - len(selected)
        selected.update(
            _evenly_sample_times(
                [value for value in minimum_times if value not in selected],
                remaining,
            )
        )

    remaining = budget - len(selected)
    if remaining > 0:
        boundaries = [
            float(value)
            for event in events
            for value in (event.start_s, event.end_s)
            if 0.0 <= float(value) <= makespan
            and float(value) not in excluded
            and float(value) not in selected
        ]
        selected.update(_evenly_sample_times(boundaries, remaining))
    return sorted(selected)


def build_replay_frame_times(
    trajectories: TrajectorySet,
    collision: CollisionSimulationResult | Sequence[CollisionEvent],
    interval_s: float,
    *,
    max_frames: int = _MAX_REPLAY_FRAMES,
) -> list[float]:
    """Build bounded frames with representative collision and mode keyframes.

    Regular timeline frames are mandatory. Collision minima and boundaries are
    supplemental keyframes: when a result contains more events than the safety
    budget permits, the worst collision instants and time-distributed event
    boundaries are retained instead of rejecting the complete Replay.
    """
    if not math.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("Replay frame interval must be a positive finite number.")
    makespan = max(float(item.time_s[-1]) for item in trajectories.robots)
    regular = {
        *np.arange(0.0, makespan, interval_s, dtype=np.float64).tolist(),
        0.0,
        makespan,
    }
    if len(regular) > max_frames:
        raise ValueError(
            f"Replay interval creates {len(regular):,} regular frames, "
            f"exceeding the {max_frames:,}-frame safety limit."
        )

    mode_changes: set[float] = set()
    for trajectory in trajectories.robots:
        changed = np.flatnonzero(trajectory.mode[1:] != trajectory.mode[:-1]) + 1
        mode_changes.update(float(trajectory.time_s[index]) for index in changed)
    optional_modes = sorted(mode_changes - regular)
    mode_budget = min(
        len(optional_modes),
        max_frames - len(regular),
        max(0, math.ceil(len(regular) * 0.1)),
    )
    collision_budget = max_frames - len(regular) - mode_budget
    collision_frames = _collision_keyframes(
        list(_events(collision)),
        makespan,
        collision_budget,
        regular,
    )
    mode_frames = _evenly_sample_times(optional_modes, mode_budget)
    return sorted({*regular, *collision_frames, *mode_frames})


def _deposition_path(trajectory: RobotTrajectory) -> dict[str, list[float | None]]:
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []
    times: list[float | None] = []
    active = False
    last_d_index = len(trajectory.mode) - 2
    for index, mode in enumerate(trajectory.mode[:-1]):
        if int(mode) != int(MODE_D):
            if active:
                end_time = float(trajectory.time_s[index])
                x_values.append(None)
                y_values.append(None)
                z_values.append(None)
                times.append(end_time)
                active = False
            continue
        if not active:
            start = trajectory.xyz_mm[index]
            x_values.append(round(float(start[0]), 4))
            y_values.append(round(float(start[1]), 4))
            z_values.append(round(float(start[2]), 4))
            times.append(float(trajectory.time_s[index]))
            active = True
        end = trajectory.xyz_mm[index + 1]
        x_values.append(round(float(end[0]), 4))
        y_values.append(round(float(end[1]), 4))
        z_values.append(round(float(end[2]), 4))
        times.append(float(trajectory.time_s[index + 1]))
        if index == last_d_index:
            x_values.append(None)
            y_values.append(None)
            z_values.append(None)
            times.append(float(trajectory.time_s[index + 1]))
    return {"x": x_values, "y": y_values, "z": z_values, "times": times}


def _circle(center: Sequence[float], radius: float) -> tuple[list[float], list[float], list[float]]:
    angles = np.linspace(0.0, 2.0 * math.pi, 33)
    return (
        (float(center[0]) + radius * np.cos(angles)).tolist(),
        (float(center[1]) + radius * np.sin(angles)).tolist(),
        np.full_like(angles, float(center[2])).tolist(),
    )


def _active_colors(time_s: float, events: Sequence[CollisionEvent]) -> list[str]:
    arm_active: set[int] = set()
    tcp_active: set[int] = set()
    for event in events:
        if not event.start_s <= time_s <= event.end_s:
            continue
        target = arm_active if event.collision_type == "ARM_ENVELOPE" else tcp_active
        target.update((event.robot_a, event.robot_b))
    return [
        "#ff4057"
        if robot_id in arm_active
        else "#f5b942"
        if robot_id in tcp_active
        else _COLORS[robot_id - 1]
        for robot_id in (1, 2, 3)
    ]


def _capsule_xy(
    base: Sequence[float], tcp: Sequence[float], radius: float
) -> tuple[list[float], list[float]]:
    direction_x = float(tcp[0]) - float(base[0])
    direction_y = float(tcp[1]) - float(base[1])
    if math.hypot(direction_x, direction_y) <= 1.0e-12:
        angles = np.linspace(0.0, 2.0 * math.pi, 34)
        return (
            (float(base[0]) + radius * np.cos(angles)).tolist(),
            (float(base[1]) + radius * np.sin(angles)).tolist(),
        )
    angle = math.atan2(direction_y, direction_x)
    base_angles = np.linspace(angle + math.pi / 2.0, angle + 3.0 * math.pi / 2.0, 17)
    tcp_angles = np.linspace(angle - math.pi / 2.0, angle + math.pi / 2.0, 17)
    x_values = [float(base[0]) + radius * math.cos(value) for value in base_angles]
    y_values = [float(base[1]) + radius * math.sin(value) for value in base_angles]
    x_values.extend(float(tcp[0]) + radius * math.cos(value) for value in tcp_angles)
    y_values.extend(float(tcp[1]) + radius * math.sin(value) for value in tcp_angles)
    x_values.append(x_values[0])
    y_values.append(y_values[0])
    return x_values, y_values


def _post_script(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""
const replay = {encoded};
const graph = document.getElementById('{{plot_id}}');
const controls = document.createElement('div');
controls.style.cssText = 'display:flex;gap:12px;align-items:center;padding:12px 4px;' +
  'color:#dce8f5;font-family:Segoe UI,sans-serif';
controls.innerHTML = '<button type="button" style="background:#19c37d;color:#06130e;' +
  'border:0;border-radius:5px;padding:8px 16px;font-weight:700;cursor:pointer">재생</button>' +
  '<input type="range" min="0" max="' + (replay.times.length - 1) + '" value="0" style="flex:1">' +
  '<strong style="min-width:170px;text-align:right"></strong>';
graph.parentNode.insertBefore(controls, graph);
const playButton = controls.querySelector('button');
const slider = controls.querySelector('input');
const label = controls.querySelector('strong');
let timer = null;
function upperBound(values, target) {{
  let low = 0, high = values.length;
  while (low < high) {{
    const mid = (low + high) >>> 1;
    const value = values[mid] === null ? -Infinity : values[mid];
    if (value <= target) low = mid + 1; else high = mid;
  }}
  return low;
}}
function capsule(base, tcp, radius) {{
  const dx = tcp[0] - base[0], dy = tcp[1] - base[1];
  const length = Math.hypot(dx, dy);
  const x = [], y = [];
  if (length <= 1e-12) {{
    for (let i = 0; i <= 32; i += 1) {{
      const angle = 2 * Math.PI * i / 32;
      x.push(base[0] + radius * Math.cos(angle));
      y.push(base[1] + radius * Math.sin(angle));
    }}
    return [x, y];
  }}
  const angle = Math.atan2(dy, dx);
  for (let i = 0; i <= 16; i += 1) {{
    const value = angle + Math.PI / 2 + Math.PI * i / 16;
    x.push(base[0] + radius * Math.cos(value));
    y.push(base[1] + radius * Math.sin(value));
  }}
  for (let i = 0; i <= 16; i += 1) {{
    const value = angle - Math.PI / 2 + Math.PI * i / 16;
    x.push(tcp[0] + radius * Math.cos(value));
    y.push(tcp[1] + radius * Math.sin(value));
  }}
  x.push(x[0]); y.push(y[0]);
  return [x, y];
}}
function translucent(hex, alpha) {{
  const red = parseInt(hex.slice(1, 3), 16);
  const green = parseInt(hex.slice(3, 5), 16);
  const blue = parseInt(hex.slice(5, 7), 16);
  return 'rgba(' + red + ',' + green + ',' + blue + ',' + alpha + ')';
}}
function showFrame(frameIndex) {{
  const index = Math.max(0, Math.min(replay.times.length - 1, Number(frameIndex)));
  const time = replay.times[index];
  const positions = replay.positions[index];
  const colors = replay.colors[index];
  for (let robot = 0; robot < 3; robot += 1) {{
    const base = replay.bases[robot];
    const tcp = positions[robot];
    const armTrace = replay.traceIndices.arms3d[robot];
    const circleTrace = replay.traceIndices.tcpCircles3d[robot];
    Plotly.restyle(graph, {{x:[[base[0],tcp[0]]],y:[[base[1],tcp[1]]],z:[[base[2],tcp[2]]],
      'line.color':[colors[robot]],'marker.color':[colors[robot]]}}, [armTrace]);
    const cx = [], cy = [], cz = [];
    for (let point = 0; point <= 32; point += 1) {{
      const angle = 2 * Math.PI * point / 32;
      cx.push(tcp[0] + replay.radii[robot] * Math.cos(angle));
      cy.push(tcp[1] + replay.radii[robot] * Math.sin(angle));
      cz.push(tcp[2]);
    }}
    Plotly.restyle(graph, {{x:[cx],y:[cy],z:[cz],'line.color':[colors[robot]]}}, [circleTrace]);
    const path = replay.deposition[robot];
    const end = upperBound(path.times, time);
    Plotly.restyle(graph, {{
      x:[path.x.slice(0,end)],
      y:[path.y.slice(0,end)],
      z:[path.z.slice(0,end)]
    }}, [replay.traceIndices.deposition3d[robot]]);
    const physical = capsule(base, tcp, replay.armRadii[robot]);
    const decision = capsule(
      base,
      tcp,
      replay.armRadii[robot] + replay.armClearance / 2
    );
    Plotly.restyle(graph, {{
      x:[physical[0]], y:[physical[1]],
      'line.color':[colors[robot]], 'fillcolor':[translucent(colors[robot], 0.22)]
    }}, [replay.traceIndices.physicalCapsules[robot]]);
    Plotly.restyle(graph, {{
      x:[decision[0]], y:[decision[1]], 'line.color':[colors[robot]]
    }}, [replay.traceIndices.decisionCapsules[robot]]);
    Plotly.restyle(graph, {{
      x:[[base[0],tcp[0]]], y:[[base[1],tcp[1]]], 'line.color':[colors[robot]]
    }}, [replay.traceIndices.centerlines2d[robot]]);
  }}
  slider.value = String(index);
  label.textContent = '시간 ' +
    time.toLocaleString(undefined, {{maximumFractionDigits:1}}) +
    ' s · ' + (index + 1) + '/' + replay.times.length;
}}
slider.addEventListener('input', () => showFrame(slider.value));
playButton.addEventListener('click', () => {{
  if (timer !== null) {{
    clearInterval(timer);
    timer = null;
    playButton.textContent = '재생';
    return;
  }}
  playButton.textContent = '일시정지';
  timer = setInterval(() => {{
    const next = Number(slider.value) + 1;
    if (next >= replay.times.length) {{
      clearInterval(timer);
      timer = null;
      playButton.textContent = '재생';
      return;
    }}
    showFrame(next);
  }}, 100);
}});
showFrame(0);
"""


def generate_replay_html(
    trajectories: TrajectorySet,
    target_mesh: trimesh.Trimesh,
    collision: CollisionSimulationResult | Sequence[CollisionEvent],
    config: Config,
    path: Path,
    *,
    interval_s: float | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ReplayGenerationStats:
    """Write an offline replay without duplicating the complete D path per frame."""
    try:
        interval = 1.0 if interval_s is None else interval_s
        events = list(_events(collision))
        frame_times = build_replay_frame_times(trajectories, events, interval)
        positions: list[list[list[float]]] = []
        colors: list[list[str]] = []
        total = len(frame_times)
        report_every = max(1, total // 100)
        for index, time_s in enumerate(frame_times, start=1):
            xyz, _ = interpolate_all_states(trajectories, time_s)
            positions.append(np.round(xyz.astype(np.float64), 4).tolist())
            colors.append(_active_colors(time_s, events))
            if progress_callback is not None and (index == total or index % report_every == 0):
                progress_callback(index, total)

        deposition = [_deposition_path(trajectory) for trajectory in trajectories.robots]
        vertices = np.asarray(target_mesh.vertices)
        faces = np.asarray(target_mesh.faces)
        robots_by_id = tuple(config.robot(robot_id) for robot_id in (1, 2, 3))
        bases = [list(map(float, robot.base_xyz_mm)) for robot in robots_by_id]
        radii = [float(robot.tcp_radius_mm) for robot in robots_by_id]
        arm_radii = [float(robot.arm_envelope_radius_mm) for robot in robots_by_id]
        initial = positions[0]
        figure = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scene"}, {"type": "xy"}]],
            column_widths=[0.62, 0.38],
            subplot_titles=("3D trajectory replay", "XY Capsule top view (actual scale)"),
        )
        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="lightgray",
                opacity=0.25,
                name="Target",
            ),
            row=1,
            col=1,
        )
        trace_indices: dict[str, list[int]] = {
            "arms3d": [],
            "tcpCircles3d": [],
            "deposition3d": [],
            "physicalCapsules": [],
            "decisionCapsules": [],
            "centerlines2d": [],
        }
        for index, robot in enumerate(robots_by_id):
            base = bases[index]
            tcp = initial[index]
            color = colors[0][index]
            trace_indices["arms3d"].append(len(figure.data))
            figure.add_trace(
                go.Scatter3d(
                    x=[base[0], tcp[0]],
                    y=[base[1], tcp[1]],
                    z=[base[2], tcp[2]],
                    mode="lines+markers",
                    line={"color": color, "width": 7},
                    marker={"size": 4, "color": color},
                    name=f"Robot {robot.id}",
                ),
                row=1,
                col=1,
            )
            circle_x, circle_y, circle_z = _circle(tcp, radii[index])
            trace_indices["tcpCircles3d"].append(len(figure.data))
            figure.add_trace(
                go.Scatter3d(
                    x=circle_x,
                    y=circle_y,
                    z=circle_z,
                    mode="lines",
                    line={"color": color, "width": 2},
                    name=f"R{robot.id} TCP radius",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
        for index, robot in enumerate(robots_by_id):
            trace_indices["deposition3d"].append(len(figure.data))
            figure.add_trace(
                go.Scatter3d(
                    x=[],
                    y=[],
                    z=[],
                    mode="lines",
                    line={"color": _COLORS[index], "width": 5},
                    name=f"Robot {robot.id} completed D path",
                ),
                row=1,
                col=1,
            )
        workspace_angles = np.linspace(0.0, 2.0 * math.pi, 65)
        workspace_center = config.workspace.center_xy_mm
        figure.add_trace(
            go.Scatter(
                x=workspace_center[0] + config.workspace.radius_mm * np.cos(workspace_angles),
                y=workspace_center[1] + config.workspace.radius_mm * np.sin(workspace_angles),
                mode="lines",
                fill="toself",
                fillcolor="rgba(56,189,248,0.06)",
                line={"color": "rgba(56,189,248,0.55)", "dash": "dot"},
                name="Workspace",
            ),
            row=1,
            col=2,
        )
        triangle = [*bases, bases[0]]
        figure.add_trace(
            go.Scatter(
                x=[point[0] for point in triangle],
                y=[point[1] for point in triangle],
                mode="lines+markers",
                line={"color": "#64748b", "dash": "dot"},
                marker={"symbol": "triangle-up", "size": 9},
                name="Robot bases",
            ),
            row=1,
            col=2,
        )
        for index, robot in enumerate(robots_by_id):
            base = bases[index]
            tcp = initial[index]
            color = colors[0][index]
            physical_x, physical_y = _capsule_xy(base, tcp, arm_radii[index])
            decision_x, decision_y = _capsule_xy(
                base,
                tcp,
                arm_radii[index] + config.collision.arm_clearance_mm / 2.0,
            )
            trace_indices["physicalCapsules"].append(len(figure.data))
            figure.add_trace(
                go.Scatter(
                    x=physical_x,
                    y=physical_y,
                    mode="lines",
                    fill="toself",
                    fillcolor=_rgba(color, 0.22),
                    line={"color": color, "width": 1},
                    name=f"R{robot.id} physical Capsule",
                ),
                row=1,
                col=2,
            )
            trace_indices["decisionCapsules"].append(len(figure.data))
            figure.add_trace(
                go.Scatter(
                    x=decision_x,
                    y=decision_y,
                    mode="lines",
                    line={"color": color, "width": 2, "dash": "dash"},
                    name=f"R{robot.id} decision outline",
                ),
                row=1,
                col=2,
            )
            trace_indices["centerlines2d"].append(len(figure.data))
            figure.add_trace(
                go.Scatter(
                    x=[base[0], tcp[0]],
                    y=[base[1], tcp[1]],
                    mode="lines+markers",
                    line={"color": color, "width": 2},
                    marker={"size": 5},
                    name=f"R{robot.id} centerline",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
        figure.update_layout(
            title=(
                f"WAAM three-robot replay · {total:,} frames · {interval:g} s interval · "
                "red=ARM_ENVELOPE, amber=TCP_RADIUS"
            ),
            paper_bgcolor="#08111d",
            plot_bgcolor="#08111d",
            font={"color": "#dce8f5"},
            scene={
                "xaxis_title": "X [mm]",
                "yaxis_title": "Y [mm]",
                "zaxis_title": "Z [mm]",
                "aspectmode": "data",
            },
            margin={"l": 0, "r": 0, "t": 54, "b": 0},
        )
        figure.update_xaxes(title_text="X [mm]", row=1, col=2)
        figure.update_yaxes(
            title_text="Y [mm]",
            scaleanchor="x",
            scaleratio=1.0,
            row=1,
            col=2,
        )
        payload: dict[str, object] = {
            "times": [round(value, 6) for value in frame_times],
            "positions": positions,
            "colors": colors,
            "bases": bases,
            "radii": radii,
            "armRadii": arm_radii,
            "armClearance": float(config.collision.arm_clearance_mm),
            "deposition": deposition,
            "traceIndices": trace_indices,
        }
        html_text = figure.to_html(
            include_plotlyjs=True,
            full_html=True,
            auto_play=False,
            post_script=_post_script(payload),
        )
        path.write_text(html_text, encoding="utf-8")
        point_count = sum(len(item["x"]) for item in deposition)
        return ReplayGenerationStats(total, point_count)
    except OSError as exc:
        raise OutputWriteError("OUTPUT_WRITE_FAILED", str(exc)) from exc
    except (OutputWriteError, ComputationError):
        raise
    except Exception as exc:
        raise ComputationError("VISUALIZATION_FAILED", str(exc)) from exc
