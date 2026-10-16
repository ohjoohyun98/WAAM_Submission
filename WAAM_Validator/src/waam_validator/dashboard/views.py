"""Reusable figures and compact presentation components for the single-job UI."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from dash import html
from plotly.subplots import make_subplots

JsonDict = dict[str, Any]


def metric_card(label: str, value: str, detail: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(label, className="metric-label"),
            html.Div(value, className="metric-value"),
            html.Div(detail, className="metric-detail"),
        ],
        className="metric-card",
    )


def style_figure(figure: go.Figure, *, height: int = 330) -> go.Figure:
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#dce8f5", "family": "Segoe UI, Malgun Gothic, sans-serif"},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin={"l": 56, "r": 24, "t": 54, "b": 48},
        height=height,
        hoverlabel={"bgcolor": "#122338", "font_color": "#e8f1fb"},
    )
    figure.update_xaxes(gridcolor="#203249", zerolinecolor="#314861")
    figure.update_yaxes(gridcolor="#203249", zerolinecolor="#314861")
    return figure


def empty_figure(message: str, *, height: int = 330) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, font={"color": "#8fa3b8"})
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return style_figure(figure, height=height)


def threshold_content(thresholds: JsonDict) -> list[html.Div | html.P]:
    if "error" in thresholds:
        return [html.P(str(thresholds["error"]), className="error-code")]
    rows = [
        ("전체 Coverage ≥", thresholds.get("minimum_overall_coverage")),
        ("전체 Overfill ≤", thresholds.get("maximum_overall_overfill_ratio")),
        ("전체 IoU ≥", thresholds.get("minimum_overall_iou")),
        ("Layer IoU ≥", thresholds.get("minimum_layer_iou")),
        ("실패 Layer 비율 ≤", thresholds.get("maximum_failed_layer_ratio")),
    ]
    return [
        html.Div(
            [html.Span(label), html.Strong(f"{float(value):.2%}")],
            className="threshold-row",
        )
        for label, value in rows
        if isinstance(value, int | float)
    ]


def robot_time_figure(rows: list[JsonDict]) -> go.Figure:
    """Render all robot states against the shared schedule makespan."""
    if not rows:
        return empty_figure("robot_metrics.csv가 없습니다")
    robots = [f"R{int(row['robot_id'])}" for row in rows]
    makespan = max(
        float(row.get("completion_s", 0.0) or 0.0)
        + float(row.get("inactive_after_completion_s", 0.0) or 0.0)
        for row in rows
    )
    figure = go.Figure()
    for field, name, color in (
        ("deposition_time_s", "Deposition · 적층", "#f97316"),
        ("travel_time_s", "Travel · 비적층 이동", "#38bdf8"),
        ("wait_time_s", "Wait · 위치 유지 대기", "#64748b"),
        ("inactive_after_completion_s", "완료 후 비활성", "#1e293b"),
    ):
        seconds = [float(row.get(field, 0.0) or 0.0) for row in rows]
        percentages = [value / makespan * 100.0 if makespan else 0.0 for value in seconds]
        figure.add_bar(
            x=robots,
            y=percentages,
            name=name,
            marker={"color": color, "line": {"width": 0}},
            text=[f"{value:.1f}%" if value >= 3.0 else "" for value in percentages],
            textposition="inside",
            customdata=seconds,
            hovertemplate=(
                "%{x}<br>비율 %{y:.2f}%<br>시간 %{customdata:,.2f} s<extra>" + name + "</extra>"
            ),
        )
    figure.update_layout(
        barmode="stack",
        yaxis={"title": "Makespan 대비 비율 [%]", "range": [0, 100], "ticksuffix": "%"},
        uniformtext={"mode": "show", "minsize": 10},
        annotations=[
            {
                "text": f"기준 Makespan: {makespan:,.2f} s = 100%",
                "xref": "paper",
                "yref": "paper",
                "x": 1.0,
                "y": 1.15,
                "xanchor": "right",
                "showarrow": False,
                "font": {"color": "#aebed0", "size": 12},
            }
        ],
    )
    return style_figure(figure, height=360)


def collision_timeline(rows: list[JsonDict]) -> go.Figure:
    if not rows:
        return empty_figure("충돌 이벤트 없음", height=300)
    figure = go.Figure()
    colors = {"ARM_ENVELOPE": "#ff6376", "TCP_RADIUS": "#f5b942"}
    for collision_type in ("ARM_ENVELOPE", "TCP_RADIUS"):
        typed = [row for row in rows if row.get("type") == collision_type]
        if not typed:
            continue
        labels = [
            f"{collision_type} · R{row.get('robot_a')}-R{row.get('robot_b')}" for row in typed
        ]
        durations = np.asarray(
            [float(row.get("duration_s", 0.0) or 0.0) for row in typed],
            dtype=np.float64,
        )
        starts = np.asarray(
            [float(row.get("start_s", 0.0) or 0.0) for row in typed],
            dtype=np.float64,
        )
        customdata = np.asarray(
            [
            [
                int(row.get("event_id", 0) or 0),
                float(row.get("end_s", 0.0) or 0.0),
                float(row.get("minimum_safety_margin_mm", 0.0) or 0.0),
            ]
            for row in typed
            ],
            dtype=np.float64,
        )
        figure.add_trace(
            go.Bar(
                y=labels,
                x=durations,
                base=starts,
                orientation="h",
                name=collision_type,
                marker={"color": colors[collision_type], "line": {"width": 0}},
                customdata=customdata,
                hovertemplate=(
                    "%{fullData.name} · %{y}<br>Event %{customdata[0]}"
                    "<br>시작 %{base:,.3f} s · 종료 %{customdata[1]:,.3f} s"
                    "<br>지속 %{x:,.3f} s · 최소 여유 %{customdata[2]:,.3f} mm"
                    "<extra></extra>"
                ),
            )
        )
        zero_duration = durations <= 0.0
        if np.any(zero_duration):
            figure.add_trace(
                go.Scattergl(
                    x=starts[zero_duration],
                    y=np.asarray(labels)[zero_duration],
                    mode="markers",
                    name=f"{collision_type} 순간 이벤트",
                    marker={"color": colors[collision_type], "size": 8, "symbol": "line-ns"},
                    customdata=customdata[zero_duration],
                    hovertemplate=(
                        "%{fullData.name} · %{y}<br>Event %{customdata[0]}"
                        "<br>시각 %{x:,.3f} s · 최소 여유 %{customdata[2]:,.3f} mm"
                        "<extra></extra>"
                    ),
                )
            )
    lane_count = len(
        {
            (row.get("type"), row.get("robot_a"), row.get("robot_b"))
            for row in rows
        }
    )
    figure.update_layout(
        xaxis_title="시간 [s]",
        yaxis_title="Robot pair",
        barmode="overlay",
        annotations=[
            {
                "text": f"전체 {len(rows):,}개 이벤트 표시",
                "xref": "paper",
                "yref": "paper",
                "x": 1.0,
                "y": 1.18,
                "xanchor": "right",
                "showarrow": False,
                "font": {"color": "#aebed0", "size": 12},
            }
        ],
    )
    return style_figure(figure, height=max(300, 170 + 42 * lane_count))


def shape_figure(rows: list[JsonDict], thresholds: JsonDict) -> go.Figure:
    if not rows:
        return empty_figure("layer_metrics.csv가 없습니다", height=430)
    x_values = [int(row.get("layer_index", 0)) for row in rows]
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.16,
        subplot_titles=("Coverage · IoU", "Underfill · Overfill · IoU 손실"),
    )
    for field, name, color in (
        ("coverage", "Coverage", "#21d4a3"),
        ("iou", "IoU", "#5caeff"),
    ):
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=[float(row.get(field, 0.0) or 0.0) for row in rows],
                name=name,
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker={"size": 5},
                hovertemplate=f"Layer %{{x}}<br>{name} %{{y:.4%}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    iou_values = [float(row.get("iou", 0.0) or 0.0) for row in rows]
    for field, name, color in (
        ("underfill_ratio", "Underfill", "#f5b942"),
        ("overfill_ratio", "Overfill", "#ff6376"),
        ("iou_loss", "IoU 손실", "#a78bfa"),
    ):
        values = [
            (max(0.0, 1.0 - iou) if field == "iou_loss" else _optional_ratio(row, field))
            for row, iou in zip(rows, iou_values, strict=True)
        ]
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=values,
                name=name,
                mode="lines+markers",
                line={"color": color, "width": 1.6},
                marker={"size": 4},
                hovertemplate=f"Layer %{{x}}<br>{name} %{{y:.4%}}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    iou_min = thresholds.get("minimum_layer_iou")
    if isinstance(iou_min, int | float):
        figure.add_hline(
            y=float(iou_min),
            line_dash="dot",
            line_color="#5caeff",
            annotation_text="Layer IoU 기준",
            row=1,
            col=1,
        )
    failed = [row for row in rows if not bool(row.get("passed", False))]
    if failed:
        figure.add_trace(
            go.Scatter(
                x=[int(row.get("layer_index", 0)) for row in failed],
                y=[float(row.get("iou", 0.0) or 0.0) for row in failed],
                name="실패 Layer",
                mode="markers",
                marker={"color": "#ff6376", "size": 9, "symbol": "x"},
            ),
            row=1,
            col=1,
        )
    error_values = [float(row.get("underfill_ratio", 0.0) or 0.0) for row in rows]
    error_values.extend(
        value for row in rows if (value := _optional_ratio(row, "overfill_ratio")) is not None
    )
    error_values.extend(max(0.0, 1.0 - value) for value in iou_values)
    error_max = max(error_values, default=0.0)
    figure.update_yaxes(title_text="일치율", tickformat=".3%", row=1, col=1)
    figure.update_yaxes(
        title_text="오차율",
        tickformat=".3%",
        range=[0.0, max(0.0001, error_max * 1.15)],
        row=2,
        col=1,
    )
    figure.update_xaxes(title_text="Layer 번호", row=2, col=1)
    return style_figure(figure, height=460)


def _optional_ratio(row: JsonDict, field: str) -> float | None:
    value = row.get(field)
    return None if value is None else float(value)


def shape_metrics_are_uniform(rows: list[JsonDict], tolerance: float = 1.0e-6) -> bool:
    """Return whether every metric is constant across layers within tolerance."""
    if not rows:
        return False
    spreads = []
    for field in ("coverage", "underfill_ratio", "overfill_ratio", "iou"):
        optional_values = [_optional_ratio(row, field) for row in rows]
        if any(value is None for value in optional_values) and not all(
            value is None for value in optional_values
        ):
            return False
        values = [value for value in optional_values if value is not None]
        if not values:
            continue
        spreads.append(max(values) - min(values))
    return max(spreads, default=0.0) <= tolerance


def shape_variation_message(rows: list[JsonDict], tolerance: float = 1.0e-6) -> str:
    if not rows:
        return "Layer 수치가 없습니다."
    if any(row.get("overfill_ratio") is None for row in rows):
        return (
            "Target 단면이 없는 Layer의 Overfill 비율은 정의되지 않으며 "
            "그래프에서 빈 값으로 표시합니다."
        )
    spreads = []
    for field in ("coverage", "underfill_ratio", "overfill_ratio", "iou"):
        values = [float(row.get(field, 0.0) or 0.0) for row in rows]
        spreads.append(max(values) - min(values))
    spread = max(spreads)
    return (
        "모든 Layer 지표가 표시 정밀도 안에서 동일합니다."
        if shape_metrics_are_uniform(rows, tolerance)
        else f"Layer 지표 전체 범위 차이: {spread:.4%}"
    )


def issues_content(payload: JsonDict) -> html.Div:
    raw = payload.get("issues", [])
    issues = raw if isinstance(raw, list) else []
    if not issues:
        return html.Div("추가 판정 기록이 없습니다.", className="empty-state success-border")
    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        "위반" if item.get("severity") == "violation" else "경고",
                        className="issue-level",
                    ),
                    html.Strong(str(item.get("code", "UNKNOWN"))),
                    html.P(str(item.get("message", ""))),
                ],
                className=f"issue-item issue-{item.get('severity', 'warning')}",
            )
            for value in issues
            if isinstance(value, dict)
            for item in [value]
        ],
        className="issue-list",
    )


def _section(payload: JsonDict, name: str) -> JsonDict:
    value = payload.get(name)
    return value if isinstance(value, dict) else {}


def _file_size_value(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def _detail_rows(items: list[tuple[str, object]]) -> html.Dl:
    children: list[Any] = []
    for label, value in items:
        children.extend([html.Dt(label), html.Dd("—" if value is None else str(value))])
    return html.Dl(children, className="input-detail-list")


def _vector(value: object) -> str:
    if not isinstance(value, list | tuple):
        return "—"
    return "[" + ", ".join(f"{float(item):,.3f}" for item in value) + "]"


def _duration_units(seconds: float) -> str:
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{seconds:,.2f}초 · {hours:,}시간 {minutes:02d}분 {seconds_part:02d}초"


def _inspection_issues(payload: JsonDict) -> Any:
    groups = (
        ("차단 오류", payload.get("blocking_errors", []), "error"),
        ("예상 FAIL", payload.get("expected_failures", []), "expected-fail"),
        ("경고", payload.get("warnings", []), "warning"),
    )
    sections: list[Any] = []
    for label, raw_issues, tone in groups:
        issues = raw_issues if isinstance(raw_issues, list) else []
        if not issues:
            continue
        rows = [
            html.Li(
                [
                    html.Code(str(item.get("code", "UNKNOWN"))),
                    html.Span(str(item.get("message", ""))),
                ]
            )
            for value in issues
            if isinstance(value, dict)
            for item in [value]
        ]
        sections.append(
            html.Div(
                [html.H4(f"{label} · {len(rows)}"), html.Ul(rows)],
                className=f"input-issues {tone}",
            )
        )
    return sections or html.Div("오류나 경고가 없습니다.", className="empty-state")


def _config_inspection(config: JsonDict) -> Any:
    simulation = _section(config, "simulation")
    process = _section(config, "process")
    workspace = _section(config, "workspace")
    collision = _section(config, "collision")
    validation = _section(config, "validation")
    shape = _section(config, "shape_validation")
    rows: list[tuple[str, object]] = [
        ("충돌 샘플 최대 시간 간격", f"{simulation.get('max_time_step_s', '—')} s"),
        ("충돌 샘플당 최대 TCP 이동거리", f"{simulation.get('max_tcp_step_mm', '—')} mm"),
        ("충돌 이벤트 병합 최대 간격", f"{simulation.get('event_merge_gap_s', '—')} s"),
        ("스트리밍 처리 묶음 크기", simulation.get("batch_size", "—")),
        (
            "Deposition / Travel 기준 속도",
            f"{process.get('deposition_speed_mm_s', '—')} / "
            f"{process.get('travel_speed_mm_s', '—')} mm/s",
        ),
        (
            "Layer 높이 / 명목 Bead 폭",
            f"{process.get('layer_height_mm', '—')} / {process.get('bead_width_mm', '—')} mm",
        ),
        (
            "Build plane / TCP Z 기준",
            f"{process.get('build_plane_z_mm', '—')} mm / {process.get('tcp_z_reference', '—')}",
        ),
        ("Travel 안전 높이", f"{process.get('safe_travel_z_mm', '—')} mm"),
        (
            "Arc 시작 / 종료 대기시간",
            f"{process.get('arc_on_time_s', '—')} / {process.get('arc_off_time_s', '—')} s",
        ),
        (
            "Arm Envelope / TCP Radius 검사",
            f"{collision.get('check_arm_envelope', '—')} / "
            f"{collision.get('check_tcp_radius', '—')}",
        ),
        ("Arm 공통 안전거리", f"{collision.get('arm_clearance_mm', '—')} mm"),
        ("접촉을 충돌로 판정", collision.get("touching_is_collision", "—")),
        ("기하 허용 오차", f"{collision.get('geometry_epsilon_mm', '—')} mm"),
        (
            "Workspace",
            f"{workspace.get('shape', '—')} · center {_vector(workspace.get('center_xy_mm'))} · "
            f"R {workspace.get('radius_mm', '—')} mm",
        ),
        (
            "속도 위반 FAIL / 상대 허용 오차",
            f"{validation.get('fail_on_speed_violation', '—')} / "
            f"{validation.get('speed_relative_tolerance', '—')}",
        ),
        ("Wait 허용 이동량", f"{validation.get('wait_position_tolerance_mm', '—')} mm"),
        ("Deposition Layer Z 허용 오차", f"{validation.get('layer_z_tolerance_mm', '—')} mm"),
        (
            "Target watertight 필수 / repair",
            f"{validation.get('require_watertight_target', '—')} / "
            f"{validation.get('attempt_target_repair', '—')}",
        ),
        (
            "Coverage / IoU / Overfill",
            f"≥ {shape.get('minimum_overall_coverage', '—')} / "
            f"≥ {shape.get('minimum_overall_iou', '—')} / "
            f"≤ {shape.get('maximum_overall_overfill_ratio', '—')}",
        ),
        (
            "Layer IoU / 실패 Layer 비율",
            f"≥ {shape.get('minimum_layer_iou', '—')} / "
            f"≤ {shape.get('maximum_failed_layer_ratio', '—')}",
        ),
    ]
    robots = config.get("robots", [])
    robot_rows = (
        [item for item in robots if isinstance(item, dict)] if isinstance(robots, list) else []
    )
    for robot in robot_rows:
        robot_id = robot.get("id")
        rows.extend(
            [
                (f"R{robot_id} Base", _vector(robot.get("base_xyz_mm"))),
                (
                    f"R{robot_id} Home TCP",
                    _vector(robot.get("home_xyz_mm")) if robot.get("home_xyz_mm") else "미설정",
                ),
                (
                    f"R{robot_id} TCP / Arm 반경 / XY Reach",
                    f"{robot.get('tcp_radius_mm', '—')} / "
                    f"{robot.get('arm_envelope_radius_mm', '—')} / "
                    f"{robot.get('xy_reach_radius_mm', '—')} mm",
                ),
            ]
        )
    clearance = collision.get("arm_clearance_mm")
    if isinstance(clearance, int | float):
        for index, left in enumerate(robot_rows):
            for right in robot_rows[index + 1 :]:
                left_radius = left.get("arm_envelope_radius_mm")
                right_radius = right.get("arm_envelope_radius_mm")
                if isinstance(left_radius, int | float) and isinstance(right_radius, int | float):
                    required = float(left_radius) + float(right_radius) + float(clearance)
                    rows.append(
                        (
                            f"R{left.get('id')}–R{right.get('id')} 요구 중심선 거리",
                            f"{required:,.1f} mm",
                        )
                    )
    return _detail_rows(rows)


def _trajectory_inspection(trajectory: JsonDict) -> Any:
    makespan_s = float(trajectory.get("makespan_s", 0.0))
    rows = []
    raw_robots = trajectory.get("robots", [])
    for robot in raw_robots if isinstance(raw_robots, list) else []:
        if not isinstance(robot, dict):
            continue
        counts = robot.get("mode_interval_counts", {})
        rows.append(
            html.Tr(
                [
                    html.Td(f"R{robot.get('robot_id')}"),
                    html.Td(f"{int(robot.get('row_count', 0)):,}"),
                    html.Td(_duration_units(float(robot.get("end_s", 0.0)))),
                    html.Td(
                        f"D {counts.get('D', 0)} / T {counts.get('T', 0)} / W {counts.get('W', 0)}"
                        if isinstance(counts, dict)
                        else "—"
                    ),
                    html.Td(
                        f"{float(robot.get('deposition_time_s', 0.0)):,.2f} / "
                        f"{float(robot.get('travel_time_s', 0.0)):,.2f} / "
                        f"{float(robot.get('wait_time_s', 0.0)):,.2f}"
                    ),
                    html.Td(f"{float(robot.get('deposition_length_mm', 0.0)):,.2f}"),
                    html.Td(f"{float(robot.get('travel_length_mm', 0.0)):,.2f}"),
                ]
            )
        )
    return html.Section(
        [
            html.H3("Trajectory"),
            _detail_rows(
                [
                    ("전체 행", f"{int(trajectory.get('row_count', 0)):,}"),
                    ("Makespan", _duration_units(makespan_s)),
                    (
                        "Workload imbalance",
                        f"{float(trajectory.get('workload_imbalance_s', 0.0)):,.2f} s",
                    ),
                ]
            ),
            html.Div(
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Robot"),
                                    html.Th("행"),
                                    html.Th("완료"),
                                    html.Th("D/T/W 구간"),
                                    html.Th("D/T/W 시간 [s]"),
                                    html.Th("D 거리 [mm]"),
                                    html.Th("T 거리 [mm]"),
                                ]
                            )
                        ),
                        html.Tbody(rows),
                    ],
                    className="input-trajectory-table",
                ),
                className="input-table-wrap",
            ),
        ],
        className="input-panel input-panel-wide",
    )


def inspection_content(payload: JsonDict) -> Any:
    status = str(payload.get("status", "BLOCKED"))
    copy = {
        "READY": ("실행 준비 완료", "입력 구조와 좌표계 검사가 완료되었습니다."),
        "WARNING": ("실행 가능 · 경고", "경고를 검토한 뒤 Validation을 실행할 수 있습니다."),
        "EXPECTED_FAIL": ("실행 가능 · FAIL 예상", "판정 위반이 있어 결과가 FAIL일 수 있습니다."),
        "BLOCKED": ("실행 불가", "차단 오류를 수정하고 입력 확인을 다시 실행하세요."),
    }
    title, description = copy.get(status, copy["BLOCKED"])
    files = _section(payload, "files")
    file_cards = []
    for filename in ("config.yaml", "trajectory.csv", "target.stl"):
        details = _section(files, filename)
        size = details.get("size_bytes")
        file_cards.append(
            html.Div(
                [
                    html.Strong(filename),
                    html.Span(
                        _file_size_value(int(size)) if isinstance(size, int | float) else "—"
                    ),
                    html.Small(f"{details.get('status', '—')} · {details.get('modified_at', '—')}"),
                    html.Code(str(details.get("path", "—"))),
                ],
                className="input-file-card",
            )
        )
    config = _section(payload, "config")
    trajectory = _section(payload, "trajectory")
    target = _section(payload, "target")
    volume = target.get("volume_mm3")
    return html.Div(
        [
            html.Div(
                [
                    html.Div([html.Strong(title), html.P(description)]),
                    html.Span(status, className=f"input-status-pill status-{status.lower()}"),
                ],
                className=f"input-verdict is-{status.lower()}",
            ),
            html.Div(file_cards, className="input-file-grid"),
            html.Div(
                [
                    html.Section(
                        [html.H3("Config"), _config_inspection(config)],
                        className="input-panel",
                    ),
                    html.Section(
                        [
                            html.H3("Target STL"),
                            _detail_rows(
                                [
                                    (
                                        "정점 / 면 / Body",
                                        f"{target.get('vertex_count', '—')} / "
                                        f"{target.get('face_count', '—')} / "
                                        f"{target.get('body_count', '—')}",
                                    ),
                                    ("최소 XYZ [mm]", _vector(target.get("bounds_min_mm"))),
                                    ("최대 XYZ [mm]", _vector(target.get("bounds_max_mm"))),
                                    ("치수 [mm]", _vector(target.get("dimensions_mm"))),
                                    (
                                        "체적 [mm³]",
                                        f"{float(volume):,.3f}"
                                        if isinstance(volume, int | float)
                                        else "—",
                                    ),
                                    ("Watertight", target.get("watertight")),
                                ]
                            ),
                        ],
                        className="input-panel",
                    ),
                ],
                className="input-detail-grid",
            ),
            _trajectory_inspection(trajectory),
            html.Section(
                [html.H3("종합 검사"), _inspection_issues(payload)],
                className="input-panel input-panel-wide",
            ),
            html.P(f"검사 완료: {payload.get('checked_at', '—')}", className="input-checked-at"),
        ],
        className="input-inspection-result",
    )
