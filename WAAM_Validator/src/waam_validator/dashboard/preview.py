"""Bounded, deterministic Plotly previews for one inspected WAAM input set."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import plotly.graph_objects as go
import trimesh

from ..config.models import Config
from ..constants import MODE_D, MODE_T, MODE_W
from ..models import TrajectorySet
from ..schedule.metrics import compute_schedule_metrics
from ..trajectory.reach import compute_reach_metrics
from .views import robot_time_figure

TARGET_FACE_LIMIT: Final = 50_000
TRAJECTORY_POINT_LIMIT_PER_ROBOT: Final = 10_000
_ROBOT_COLORS: Final = {1: "#38bdf8", 2: "#f59e0b", 3: "#a78bfa"}
_MODE_LABELS: Final = {
    int(MODE_D): "Deposition",
    int(MODE_T): "Travel",
    int(MODE_W): "Wait",
}
_MODE_DASH: Final = {int(MODE_D): "solid", int(MODE_T): "dash", int(MODE_W): "dot"}


@dataclass(slots=True, frozen=True)
class InputPreview:
    scene: go.Figure
    gantt_figure: go.Figure
    time_figure: go.Figure
    motion_figure: go.Figure
    reach_figure: go.Figure
    warnings: list[str]
    target_faces_shown: int
    target_faces_total: int
    trajectory_points_shown: dict[int, int]


def _bounded_interval_indices(mode: np.ndarray, max_intervals: int) -> np.ndarray:
    count = max(0, len(mode) - 1)
    if count <= max_intervals:
        return np.arange(count, dtype=np.int64)
    changes = np.flatnonzero(mode[1:-1] != mode[:-2]) + 1
    mandatory = np.unique(
        np.clip(np.concatenate(([0, count - 1], changes - 1, changes)), 0, count - 1)
    )
    if len(mandatory) >= max_intervals:
        positions = np.linspace(0, len(mandatory) - 1, max_intervals, dtype=np.int64)
        return mandatory[positions]
    remaining = np.setdiff1d(np.arange(count, dtype=np.int64), mandatory, assume_unique=True)
    needed = max_intervals - len(mandatory)
    positions = np.linspace(0, len(remaining) - 1, needed, dtype=np.int64)
    return np.sort(np.concatenate((mandatory, remaining[positions])))


def _preview_mesh(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, list[str]]:
    warnings: list[str] = []
    if len(mesh.faces) <= TARGET_FACE_LIMIT:
        return (
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64),
            warnings,
        )
    try:
        simplified = mesh.simplify_quadric_decimation(face_count=TARGET_FACE_LIMIT)
        if not 0 < len(simplified.faces) <= TARGET_FACE_LIMIT:
            raise ValueError("mesh simplifier did not honor the preview face limit")
        warnings.append(
            f"Target preview를 {len(mesh.faces):,} face에서 "
            f"{len(simplified.faces):,} face로 단순화했습니다."
        )
        return (
            np.asarray(simplified.vertices, dtype=np.float64),
            np.asarray(simplified.faces, dtype=np.int64),
            warnings,
        )
    except Exception:
        selected = np.linspace(0, len(mesh.faces) - 1, TARGET_FACE_LIMIT, dtype=np.int64)
        faces = np.asarray(mesh.faces[selected], dtype=np.int64)
        used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
        warnings.append("Target preview 단순화에 실패하여 결정론적 face sampling을 사용했습니다.")
        return (
            np.asarray(mesh.vertices[used], dtype=np.float64),
            inverse.reshape((-1, 3)),
            warnings,
        )


def _workspace_trace(config: Config) -> go.Mesh3d:
    theta = np.linspace(0.0, 2.0 * math.pi, 73)
    center_x, center_y = config.workspace.center_xy_mm
    radius = config.workspace.radius_mm
    x = np.concatenate(([center_x], center_x + radius * np.cos(theta)))
    y = np.concatenate(([center_y], center_y + radius * np.sin(theta)))
    z = np.full_like(x, config.process.build_plane_z_mm)
    indices = np.arange(1, len(theta) + 1, dtype=np.int64)
    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=np.zeros(len(indices) - 1, dtype=np.int64),
        j=indices[:-1],
        k=indices[1:],
        name="Workspace",
        color="#22c55e",
        opacity=0.16,
        hovertemplate="Workspace<extra></extra>",
    )


def _reach_cylinder_trace(
    robot_id: int,
    base: np.ndarray,
    radius: float,
    z_min: float,
    z_max: float,
) -> go.Surface:
    angles, heights = np.meshgrid(
        np.linspace(0.0, 2.0 * math.pi, 49),
        np.asarray([z_min, z_max], dtype=np.float64),
    )
    color = _ROBOT_COLORS[robot_id]
    return go.Surface(
        x=base[0] + radius * np.cos(angles),
        y=base[1] + radius * np.sin(angles),
        z=heights,
        surfacecolor=np.zeros_like(angles),
        colorscale=[[0.0, color], [1.0, color]],
        cmin=0.0,
        cmax=1.0,
        name=f"R{robot_id} XY Reach",
        opacity=0.09,
        showscale=False,
        showlegend=True,
        hovertemplate=(
            f"R{robot_id} XY Reach<br>Radius {radius:,.2f} mm<br>Z ignored<extra></extra>"
        ),
    )


def _reach_footprint_trace(robot_id: int, base: np.ndarray, radius: float) -> go.Scatter3d:
    angles = np.linspace(0.0, 2.0 * math.pi, 73)
    return go.Scatter3d(
        x=base[0] + radius * np.cos(angles),
        y=base[1] + radius * np.sin(angles),
        z=np.full_like(angles, base[2]),
        mode="lines",
        name=f"R{robot_id} XY Reach footprint",
        line={"color": _ROBOT_COLORS[robot_id], "width": 3, "dash": "dot"},
        showlegend=False,
        hovertemplate=(
            f"R{robot_id} XY Reach footprint<br>Radius {radius:,.2f} mm"
            "<br>Z ignored<extra></extra>"
        ),
    )


def _scene(
    config: Config, trajectories: TrajectorySet, mesh: trimesh.Trimesh
) -> tuple[go.Figure, list[str], int, dict[int, int]]:
    vertices, faces, warnings = _preview_mesh(mesh)
    figure = go.Figure()
    robots_by_id = tuple(config.robot(robot_id) for robot_id in (1, 2, 3))
    bases = np.asarray([robot.base_xyz_mm for robot in robots_by_id], dtype=np.float64)
    scene_z_min = min(
        float(vertices[:, 2].min()),
        float(bases[:, 2].min()),
        *(float(trajectory.xyz_mm[:, 2].min()) for trajectory in trajectories.robots),
    )
    scene_z_max = max(
        float(vertices[:, 2].max()),
        float(bases[:, 2].max()),
        *(
            float(robot.home_xyz_mm[2])
            for robot in robots_by_id
            if robot.home_xyz_mm is not None
        ),
        *(float(trajectory.xyz_mm[:, 2].max()) for trajectory in trajectories.robots),
    )
    if scene_z_max <= scene_z_min:
        scene_z_max = scene_z_min + 1.0
    figure.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name="Target STL",
            color="#cbd5e1",
            opacity=0.34,
            flatshading=True,
            hoverinfo="skip",
        )
    )
    figure.add_trace(_workspace_trace(config))
    closed_bases = np.vstack((bases, bases[0]))
    figure.add_trace(
        go.Scatter3d(
            x=closed_bases[:, 0],
            y=closed_bases[:, 1],
            z=closed_bases[:, 2],
            mode="lines",
            name="Robot base triangle",
            line={"color": "#64748b", "width": 3, "dash": "dot"},
            hoverinfo="skip",
        )
    )
    points_shown: dict[int, int] = {}
    max_intervals = max(1, TRAJECTORY_POINT_LIMIT_PER_ROBOT // 3)
    for trajectory in trajectories.robots:
        robot = config.robot(trajectory.robot_id)
        base = np.asarray(robot.base_xyz_mm, dtype=np.float64)
        figure.add_trace(
            go.Scatter3d(
                x=[base[0]],
                y=[base[1]],
                z=[base[2]],
                mode="markers+text",
                name=f"R{robot.id} base",
                text=[f"R{robot.id}"],
                textposition="top center",
                marker={"size": 8, "color": _ROBOT_COLORS[robot.id], "symbol": "diamond"},
                hovertemplate=(
                    f"R{robot.id} Base<br>로봇 설치 기준점"
                    f"<br>X {base[0]:,.2f} mm · Y {base[1]:,.2f} mm · Z {base[2]:,.2f} mm"
                    "<extra></extra>"
                ),
            )
        )
        if robot.home_xyz_mm is not None:
            home = np.asarray(robot.home_xyz_mm, dtype=np.float64)
            figure.add_trace(
                go.Scatter3d(
                    x=[home[0]],
                    y=[home[1]],
                    z=[home[2]],
                    mode="markers+text",
                    name=f"R{robot.id} home TCP",
                    text=[f"R{robot.id} Home"],
                    textposition="bottom center",
                    marker={
                        "size": 7,
                        "color": _ROBOT_COLORS[robot.id],
                        "symbol": "x",
                    },
                    hovertemplate=(
                        f"R{robot.id} Home TCP<br>공구 기준점의 명목 대기 위치"
                        f"<br>X {home[0]:,.2f} mm · Y {home[1]:,.2f} mm · Z {home[2]:,.2f} mm"
                        "<extra></extra>"
                    ),
                )
            )
        figure.add_trace(
            _reach_cylinder_trace(
                robot.id,
                base,
                robot.xy_reach_radius_mm,
                scene_z_min,
                scene_z_max,
            )
        )
        figure.add_trace(_reach_footprint_trace(robot.id, base, robot.xy_reach_radius_mm))
        selected = _bounded_interval_indices(trajectory.mode, max_intervals)
        points_shown[robot.id] = min(TRAJECTORY_POINT_LIMIT_PER_ROBOT, int(len(selected) * 3))
        for mode_value in (int(MODE_D), int(MODE_T), int(MODE_W)):
            indices = selected[trajectory.mode[selected] == mode_value]
            if len(indices):
                # Plotly recursively validates Python lists element by element.
                # A dense numeric array is both bounded and substantially more
                # stable when users inspect several large jobs in one UI process.
                coordinates = np.full((len(indices), 3, 3), np.nan, dtype=np.float64)
                coordinates[:, 0] = trajectory.xyz_mm[indices]
                coordinates[:, 1] = trajectory.xyz_mm[indices + 1]
                flattened = coordinates.reshape((-1, 3))
                figure.add_trace(
                    go.Scatter3d(
                        x=flattened[:, 0],
                        y=flattened[:, 1],
                        z=flattened[:, 2],
                        mode="lines+markers" if mode_value == int(MODE_W) else "lines",
                        name=f"R{robot.id} {_MODE_LABELS[mode_value]}",
                        line={
                            "color": _ROBOT_COLORS[robot.id],
                            "width": 5 if mode_value == int(MODE_D) else 2,
                            "dash": _MODE_DASH[mode_value],
                        },
                        marker={"size": 2, "color": _ROBOT_COLORS[robot.id]},
                        opacity=1.0 if mode_value == int(MODE_D) else 0.65,
                        hoverinfo="skip",
                    )
                )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#07101b",
        plot_bgcolor="#07101b",
        height=680,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
        scene={
            "aspectmode": "data",
            "xaxis_title": "X [mm]",
            "yaxis_title": "Y [mm]",
            "zaxis_title": "Z [mm]",
            "bgcolor": "#07101b",
        },
    )
    return figure, warnings, len(faces), points_shown


def _statistics_figures(
    config: Config, trajectories: TrajectorySet
) -> tuple[go.Figure, go.Figure, go.Figure]:
    schedule = compute_schedule_metrics(trajectories)
    reach = compute_reach_metrics(trajectories, config)
    labels = [f"R{item.robot_id}" for item in schedule.robots]
    times = robot_time_figure(
        [
            {
                "robot_id": item.robot_id,
                "completion_s": item.completion_s,
                "deposition_time_s": item.deposition_time_s,
                "travel_time_s": item.travel_time_s,
                "wait_time_s": item.wait_time_s,
                "inactive_after_completion_s": max(0.0, schedule.makespan_s - item.completion_s),
            }
            for item in schedule.robots
        ]
    )
    motion = go.Figure()
    for name, length_field, time_field, speed_field, color in (
        (
            "적층 거리 [mm]",
            "deposition_length_mm",
            "deposition_time_s",
            "mean_deposition_speed_mm_s",
            "#f97316",
        ),
        (
            "이동 거리 [mm]",
            "travel_length_mm",
            "travel_time_s",
            "mean_travel_speed_mm_s",
            "#38bdf8",
        ),
    ):
        motion.add_bar(
            x=labels,
            y=[float(getattr(item, length_field)) for item in schedule.robots],
            name=name,
            marker_color=color,
            customdata=[
                [
                    float(getattr(item, time_field)),
                    float(getattr(item, speed_field) or 0.0),
                ]
                for item in schedule.robots
            ],
            hovertemplate=(
                "%{x}<br>거리 %{y:,.2f} mm"
                "<br>시간 %{customdata[0]:,.2f} s"
                "<br>평균 속도 %{customdata[1]:,.2f} mm/s<extra>"
                f"{name}</extra>"
            ),
        )
    motion.update_layout(
        template="plotly_dark",
        barmode="group",
        height=320,
        margin={"l": 55, "r": 15, "t": 50, "b": 40},
        yaxis_title="거리 [mm]",
        annotations=[
            {
                "text": (
                    f"설정 속도: Deposition {config.process.deposition_speed_mm_s:g} mm/s · "
                    f"Travel {config.process.travel_speed_mm_s:g} mm/s"
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 1.0,
                "y": 1.16,
                "xanchor": "right",
                "showarrow": False,
                "font": {"color": "#aebed0", "size": 12},
            }
        ],
        paper_bgcolor="#0d1927",
        plot_bgcolor="#0d1927",
    )
    reach_figure = go.Figure()
    reach_percentages = [item.xy_utilization_ratio * 100.0 for item in reach.robots]
    reach_figure.add_bar(
        x=reach_percentages,
        y=labels,
        orientation="h",
        marker_color=["#22c55e" if item.passed else "#ef4444" for item in reach.robots],
        text=[f"{value:.1f}%" for value in reach_percentages],
        textposition="inside",
        customdata=[
            [
                item.maximum_xy_distance_mm,
                item.xy_reach_radius_mm,
                item.minimum_xy_margin_mm,
                item.xy_violation_point_count,
            ]
            for item in reach.robots
        ],
        hovertemplate=(
            "%{y}<br>XY Reach 사용률 %{x:.2f}%"
            "<br>최대 Base–TCP XY 거리 %{customdata[0]:,.2f} mm"
            "<br>설정 XY Reach %{customdata[1]:,.2f} mm"
            "<br>최소 XY 여유 %{customdata[2]:,.2f} mm"
            "<br>위반 절점 %{customdata[3]:,.0f}개<extra></extra>"
        ),
        showlegend=False,
    )
    reach_maximum = max([110.0, *(value * 1.1 for value in reach_percentages)])
    reach_figure.add_vline(
        x=100.0,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="설정 XY Reach 한계 100%",
        annotation_position="top left",
    )
    reach_figure.update_layout(
        template="plotly_dark",
        height=320,
        margin={"l": 55, "r": 15, "t": 45, "b": 40},
        xaxis={
            "title": "XY Reach 사용률 [%]",
            "range": [0, reach_maximum],
            "ticksuffix": "%",
        },
        yaxis={"autorange": "reversed"},
        paper_bgcolor="#0d1927",
        plot_bgcolor="#0d1927",
    )
    return times, motion, reach_figure


def _gantt_run_count(trajectories: TrajectorySet) -> int:
    return sum(
        1 + int(np.count_nonzero(item.mode[1:-1] != item.mode[:-2])) for item in trajectories.robots
    )


def _proportional_bin_states(bin_durations: np.ndarray) -> np.ndarray:
    """Choose one raster color per bin while preserving local state-time shares.

    A dominant-state choice systematically hides short but frequent states.  The
    accumulated-share allocator instead dithers colors across neighboring bins,
    while only selecting states that are actually present in the current bin.
    """
    category_count, bin_count = bin_durations.shape
    carry = np.zeros(category_count, dtype=np.float64)
    states = np.full(bin_count, category_count - 1, dtype=np.uint8)
    for bin_index in range(bin_count):
        durations = bin_durations[:, bin_index]
        total = float(np.sum(durations))
        if total <= 0.0:
            continue
        carry += durations / total
        eligible = durations > max(1.0e-12, total * 1.0e-12)
        scores = np.where(eligible, carry, -np.inf)
        selected = int(np.argmax(scores))
        states[bin_index] = np.uint8(selected)
        carry[selected] -= 1.0
    return states


def _overview_gantt_figure(trajectories: TrajectorySet, bins: int = 2_000) -> go.Figure:
    """Render a bounded proportional raster with exact per-bin duration hover data."""
    makespan = max(float(item.time_s[-1]) for item in trajectories.robots)
    edges = np.linspace(0.0, makespan, bins + 1, dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    states = np.full((3, bins), 3, dtype=np.uint8)
    custom = np.zeros((3, bins, 6), dtype=np.float64)
    custom[:, :, 0] = edges[:-1]
    custom[:, :, 1] = edges[1:]
    for robot_index, trajectory in enumerate(trajectories.robots):
        time_s = trajectory.time_s
        modes = trajectory.mode[:-1]
        durations = np.diff(time_s)
        mode_durations = np.zeros((3, bins), dtype=np.float64)
        for mode_value in (int(MODE_T), int(MODE_D), int(MODE_W)):
            cumulative = np.concatenate(
                ([0.0], np.cumsum(durations * (modes == mode_value), dtype=np.float64))
            )
            clipped = np.minimum(edges, float(time_s[-1]))
            indices = np.searchsorted(time_s, clipped, side="right") - 1
            indices = np.clip(indices, 0, len(modes) - 1)
            values = cumulative[indices] + (clipped - time_s[indices]) * (
                modes[indices] == mode_value
            )
            values[clipped >= time_s[-1]] = cumulative[-1]
            mode_durations[mode_value] = np.diff(values)
        inactive_durations = np.maximum(0.0, np.diff(edges) - mode_durations.sum(axis=0))
        state_durations = np.vstack((mode_durations, inactive_durations))
        states[robot_index] = _proportional_bin_states(state_durations)
        custom[robot_index, :, 2] = mode_durations[int(MODE_D)]
        custom[robot_index, :, 3] = mode_durations[int(MODE_T)]
        custom[robot_index, :, 4] = mode_durations[int(MODE_W)]
        custom[robot_index, :, 5] = inactive_durations
    colors = ["#38bdf8", "#f97316", "#64748b", "#1e293b"]
    scale: list[list[float | str]] = []
    for index, color in enumerate(colors):
        scale.extend([[index / 4.0, color], [(index + 1) / 4.0, color]])
    figure = go.Figure(
        go.Heatmap(
            x=centers,
            y=["R1", "R2", "R3"],
            z=states,
            zmin=-0.5,
            zmax=3.5,
            colorscale=scale,
            showscale=False,
            customdata=custom,
            hovertemplate=(
                "%{y}<br>%{customdata[0]:,.2f}–%{customdata[1]:,.2f} s"
                "<br>Deposition %{customdata[2]:,.2f} s"
                "<br>Travel %{customdata[3]:,.2f} s"
                "<br>Wait %{customdata[4]:,.2f} s"
                "<br>완료 후 비활성 %{customdata[5]:,.2f} s"
                "<extra>시간 비율 bin 요약</extra>"
            ),
        )
    )
    for name, color in (
        ("Travel · 비적층 이동", colors[0]),
        ("Deposition · 적층", colors[1]),
        ("Wait · 위치 유지 대기", colors[2]),
        ("완료 후 비활성", colors[3]),
    ):
        figure.add_scatter(x=[None], y=[None], mode="markers", marker_color=color, name=name)
    figure.update_layout(
        template="plotly_dark",
        height=380,
        margin={"l": 65, "r": 20, "t": 100, "b": 55},
        xaxis={"title": "시간 [s]", "range": [0.0, makespan], "rangeslider": {"visible": True}},
        yaxis={"autorange": "reversed", "fixedrange": True},
        legend={"orientation": "h", "x": 0.0, "y": 1.08, "yanchor": "bottom"},
        annotations=[
            {
                "text": (
                    "전체보기는 2,000개 시간 bin의 상태시간 비율 요약입니다. "
                    "확대하면 정확한 구간을 불러옵니다."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 1.0,
                "y": 1.2,
                "xanchor": "right",
                "showarrow": False,
                "font": {"color": "#aebed0", "size": 11},
            }
        ],
        paper_bgcolor="#0d1927",
        plot_bgcolor="#0d1927",
    )
    return figure


def build_gantt_figure(
    trajectories: TrajectorySet,
    *,
    start_s: float | None = None,
    end_s: float | None = None,
    maximum_segments: int = 2_000,
) -> go.Figure:
    """Build a bounded overview or exact state-changing bars for a selected range."""
    makespan = max(float(item.time_s[-1]) for item in trajectories.robots)
    full_range = (
        start_s is None
        and end_s is None
        or float(start_s or 0.0) <= 0.0
        and float(end_s if end_s is not None else makespan) >= makespan
    )
    if full_range and _gantt_run_count(trajectories) > maximum_segments:
        return _overview_gantt_figure(trajectories, maximum_segments)
    view_start = max(0.0, float(start_s or 0.0))
    view_end = min(makespan, float(end_s if end_s is not None else makespan))
    if view_end <= view_start:
        view_start, view_end = 0.0, makespan
    state_specs = (
        (int(MODE_D), "Deposition · 적층", "#f97316"),
        (int(MODE_T), "Travel · 비적층 이동", "#38bdf8"),
        (int(MODE_W), "Wait · 위치 유지 대기", "#64748b"),
        (None, "완료 후 비활성 · 작업 종료", "#1e293b"),
    )
    segments: dict[int | None, tuple[list[float], list[float], list[float], list[str]]] = {
        mode_value: ([], [], [], []) for mode_value, _, _ in state_specs
    }
    for trajectory in trajectories.robots:
        interval_modes = trajectory.mode[:-1]
        changes = np.flatnonzero(interval_modes[1:] != interval_modes[:-1]) + 1
        run_starts = np.concatenate((np.array([0]), changes))
        run_stops = np.concatenate((changes, np.array([len(interval_modes)])))
        robot_label = f"R{trajectory.robot_id}"
        for start_raw, stop_raw in zip(run_starts, run_stops, strict=True):
            start_index = int(start_raw)
            stop_index = int(stop_raw)
            interval_start = float(trajectory.time_s[start_index])
            interval_end = float(trajectory.time_s[stop_index])
            if interval_end <= view_start or interval_start >= view_end:
                continue
            clipped_start = max(interval_start, view_start)
            clipped_end = min(interval_end, view_end)
            starts, ends, durations, robots = segments[int(interval_modes[start_index])]
            starts.append(clipped_start)
            ends.append(clipped_end)
            durations.append(clipped_end - clipped_start)
            robots.append(robot_label)

        completion_s = float(trajectory.time_s[-1])
        if completion_s < view_end and makespan > view_start:
            starts, ends, durations, robots = segments[None]
            inactive_start = max(completion_s, view_start)
            inactive_end = min(makespan, view_end)
            if inactive_end > inactive_start:
                starts.append(inactive_start)
                ends.append(inactive_end)
                durations.append(inactive_end - inactive_start)
                robots.append(robot_label)

    figure = go.Figure()
    for mode_value, name, color in state_specs:
        starts, ends, durations, robots = segments[mode_value]
        display_starts = starts if starts else [0.0]
        display_ends = ends if ends else [0.0]
        display_durations = durations if durations else [0.0]
        display_robots = robots if robots else ["R1"]
        customdata = np.column_stack((display_starts, display_ends, display_durations))
        figure.add_bar(
            x=display_durations,
            base=display_starts,
            y=display_robots,
            orientation="h",
            width=0.66,
            name=name,
            marker={"color": color, "line": {"width": 0}},
            customdata=customdata,
            hovertemplate=(
                "%{y}<br>상태: " + name + "<br>시작 %{customdata[0]:,.2f} s"
                "<br>종료 %{customdata[1]:,.2f} s"
                "<br>지속시간 %{customdata[2]:,.2f} s<extra></extra>"
            ),
        )

    active_durations = segments[int(MODE_D)][2] + segments[int(MODE_T)][2]
    typical_active_duration = (
        float(np.percentile(active_durations, 75)) if active_durations else makespan
    )
    detail_window_s = min(makespan, max(300.0, typical_active_duration * 80.0))
    first_active_times: list[float] = []
    for trajectory in trajectories.robots:
        active_indices = np.flatnonzero(
            (trajectory.mode[:-1] == MODE_D) | (trajectory.mode[:-1] == MODE_T)
        )
        if len(active_indices):
            first_active_times.append(float(trajectory.time_s[int(active_indices[0])]))
    focus_time_s = max(first_active_times, default=0.0)
    detail_start_s = min(
        max(0.0, focus_time_s - detail_window_s * 0.1),
        max(0.0, makespan - detail_window_s),
    )
    detail_end_s = detail_start_s + detail_window_s
    figure.update_layout(
        template="plotly_dark",
        barmode="overlay",
        height=420,
        margin={"l": 65, "r": 20, "t": 105, "b": 55},
        xaxis={
            "range": [view_start, view_end],
            "title": "시간 [s]",
            "rangeslider": {"visible": True, "thickness": 0.12},
        },
        yaxis={
            "autorange": "reversed",
            "categoryorder": "array",
            "categoryarray": [f"R{item.robot_id}" for item in trajectories.robots],
            "fixedrange": True,
        },
        legend={
            "orientation": "h",
            "x": 0.0,
            "xanchor": "left",
            "y": 1.08,
            "yanchor": "bottom",
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 1.0,
                "xanchor": "right",
                "y": 1.24,
                "yanchor": "top",
                "showactive": False,
                "bgcolor": "#19304c",
                "bordercolor": "#5caeff",
                "font": {"color": "#f3f8ff"},
                "buttons": [
                    {
                        "label": "상세 보기",
                        "method": "relayout",
                        "args": [{"xaxis.range": [detail_start_s, detail_end_s]}],
                    },
                    {
                        "label": "전체 보기",
                        "method": "relayout",
                        "args": [{"xaxis.range": [0.0, makespan]}],
                    },
                ],
            }
        ],
        hovermode="closest",
        paper_bgcolor="#0d1927",
        plot_bgcolor="#0d1927",
    )
    return figure


def build_input_preview(config: Config, trajectories: TrajectorySet, mesh: Any) -> InputPreview:
    """Build bounded input preview figures without modifying validation inputs."""
    scene, warnings, shown_faces, shown_points = _scene(config, trajectories, mesh)
    gantt_figure = build_gantt_figure(trajectories)
    time_figure, motion_figure, reach_figure = _statistics_figures(config, trajectories)
    return InputPreview(
        scene=scene,
        gantt_figure=gantt_figure,
        time_figure=time_figure,
        motion_figure=motion_figure,
        reach_figure=reach_figure,
        warnings=warnings,
        target_faces_shown=shown_faces,
        target_faces_total=int(len(mesh.faces)),
        trajectory_points_shown=shown_points,
    )
