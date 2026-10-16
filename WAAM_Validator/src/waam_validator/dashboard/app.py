"""Single-job WAAM Validator local web interface."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote
from urllib.request import urlopen

import dash_ag_grid as dag
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
from flask import abort, send_file

from .._version import __version__
from ..config.models import Config
from ..models import TrajectorySet
from ..provenance import RESULT_SCHEMA_VERSION, verify_validation_inputs
from ..visualization.replay import _capsule_xy
from .data import (
    ALLOWED_ARTIFACTS,
    DashboardDataError,
    JobRecord,
    RunRecord,
    load_latest_matching_run,
    load_latest_run,
    load_result_config,
    load_run_directory,
    read_csv_records,
    read_csv_window,
    resolve_single_job_artifact,
    thresholds_from_config,
)
from .input_inspector import (
    inspect_dashboard_input_bundle,
    inspection_is_current,
    resolve_input_directory,
)
from .preview import build_gantt_figure, build_input_preview
from .replay_service import (
    estimate_deposited_stl,
    estimate_replay,
    preset_interval_s,
    read_deposited_stl_status,
    read_replay_status,
)
from .runner import (
    ReplayAlreadyRunningError,
    ReplayRunManager,
    ValidationAlreadyRunningError,
    ValidationRunManager,
)
from .views import (
    collision_timeline as _collision_timeline,
)
from .views import (
    empty_figure as _empty_figure,
)
from .views import (
    inspection_content as _inspection_content,
)
from .views import (
    issues_content as _issues_content,
)
from .views import (
    metric_card as _metric_card,
)
from .views import (
    robot_time_figure as _robot_figure,
)
from .views import (
    shape_figure as _shape_figure,
)
from .views import (
    shape_variation_message,
)
from .views import (
    style_figure as _style_figure,
)
from .views import (
    threshold_content as _threshold_content,
)

JsonDict = dict[str, Any]
_GRAPH_CONFIG: Any = {"displaylogo": False, "responsive": True}
_ARTIFACT_LABELS = {
    "summary.json": "종합 판정 데이터",
    "validation_report.md": "검증 보고서",
    "robot_metrics.csv": "로봇 작업·XY Reach 수치",
    "collision_events.csv": "충돌 이벤트 수치",
    "layer_metrics.csv": "Layer 형상 수치",
    "run.log": "실행 기록",
    "deposited.stl": "명목 적층 형상 STL",
    "replay.html": "3D Replay HTML",
    "validation_inputs.json": "입력 파일 지문",
    "error.json": "실행 오류 데이터",
}
_STAGE_LABELS = {
    "starting": "프로세스 시작",
    "loading_inputs": "입력 로딩",
    "collision": "충돌 검사",
    "deposition": "적층 형상 생성",
    "target_slicing": "Target slicing",
    "shape_metrics": "형상 지표 계산",
    "results": "결과 기록",
    "completed": "완료",
    "failed": "오류",
    "process_exit": "프로세스 종료",
}


class _ContextRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paths: dict[str, Path] = {}
        self._trajectories: dict[str, TrajectorySet] = {}
        self._tokens_by_path: dict[Path, str] = {}

    def add(self, path: Path, trajectories: TrajectorySet) -> str:
        resolved = path.resolve()
        with self._lock:
            token = self._tokens_by_path.get(resolved)
            if token is None:
                token = uuid.uuid4().hex
            # The product is a single-job UI. Retaining every previously
            # inspected large trajectory would make memory grow with each
            # folder change, so only the active context stays addressable.
            self._paths.clear()
            self._trajectories.clear()
            self._tokens_by_path.clear()
            self._tokens_by_path[resolved] = token
            self._paths[token] = resolved
            self._trajectories[token] = trajectories
        return token

    def get(self, token: str) -> Path:
        with self._lock:
            path = self._paths.get(token)
        if path is None:
            raise DashboardDataError("알 수 없거나 만료된 입력 context입니다.")
        return path

    def trajectories(self, token: str) -> TrajectorySet:
        with self._lock:
            trajectories = self._trajectories.get(token)
        if trajectories is None:
            raise DashboardDataError("알 수 없거나 만료된 입력 context입니다.")
        return trajectories


def _screen(class_name: str, screen_id: str, children: list[Any]) -> html.Section:
    return html.Section(children, id=screen_id, className=class_name)


def _layout(initial_job_dir: Path) -> html.Div:
    empty_scene = _empty_figure("입력 확인 후 3D 작업 공간을 표시합니다.", height=680)
    empty_chart = _empty_figure("입력 확인 후 통계를 표시합니다.")
    return html.Div(
        [
            dcc.Store(id="view-store", data={"view": "input"}),
            dcc.Store(id="active-input-store", data={}),
            dcc.Store(id="selected-job-dir-store", data=str(initial_job_dir.resolve())),
            dcc.Store(id="current-run-store", data={}),
            dcc.Store(id="recent-run-store", data={}),
            dcc.Store(id="runtime-store", data={"state": "IDLE", "running": False}),
            dcc.Store(id="replay-runtime-store", data={"state": "IDLE", "running": False}),
            dcc.Store(id="replay-refresh", data={}),
            dcc.Store(id="result-tab-store", data={}),
            dcc.Interval(id="validation-poller", interval=1000, disabled=True),
            dcc.Interval(id="replay-poller", interval=1000, disabled=True),
            html.Header(
                [
                    html.Div("WV", className="brand-mark"),
                    html.Div(
                        [
                            html.H1("WAAM Validator"),
                            html.P(f"v{__version__} · 단일 WAAM 작업 검증"),
                        ],
                        className="brand-copy",
                    ),
                    html.Div(
                        [
                            html.Span("1 입력 준비"),
                            html.Span("2 Validation"),
                            html.Span("3 결과"),
                        ],
                        id="workflow-steps",
                        className="workflow-steps step-input",
                    ),
                ],
                className="validator-header",
            ),
            html.Main(
                [
                    _screen(
                        "validator-screen",
                        "input-screen",
                        [
                            html.Div(
                                [
                                    html.P("입력 준비", className="eyebrow"),
                                    html.H2("검증할 입력 폴더를 확인하세요"),
                                    html.P(
                                        "세 개의 고정 입력 파일을 원본 위치에서 직접 읽습니다.",
                                        className="muted-copy",
                                    ),
                                ],
                                className="screen-heading",
                            ),
                            html.Section(
                                [
                                    html.Label("입력 폴더", htmlFor="job-dir-input"),
                                    html.Div(
                                        [
                                            dcc.Input(
                                                id="job-dir-input",
                                                value=str(initial_job_dir.resolve()),
                                                type="text",
                                                # Keep the callback state synchronized when a
                                                # pasted path is followed immediately by a click.
                                                debounce=False,
                                            ),
                                            html.Button(
                                                "찾아보기",
                                                id="browse-button",
                                                className="secondary-button",
                                            ),
                                            html.Button("입력 확인", id="inspect-button"),
                                        ],
                                        className="folder-row",
                                    ),
                                    html.Div(
                                        [
                                            html.Code("config.yaml"),
                                            html.Code("trajectory.csv"),
                                            html.Code("target.stl"),
                                        ],
                                        className="required-files",
                                    ),
                                    html.P(id="folder-message", className="field-message"),
                                ],
                                className="folder-card",
                            ),
                            dcc.Loading(
                                html.Div(
                                    "입력 확인을 실행하면 상세 정보와 preview가 나타납니다.",
                                    id="inspection-content",
                                    className="input-inspection-empty",
                                ),
                                type="circle",
                            ),
                            html.Section(
                                [
                                    html.Div(
                                        [
                                            html.H3("로봇 작업 일정 (Gantt)"),
                                            html.P(
                                                "각 Robot을 하나의 가로 막대로 표시합니다. "
                                                "Deposition은 적층, Travel은 비적층 이동, "
                                                "Wait는 위치 유지 대기, 완료 후 비활성은 "
                                                "해당 Robot의 trajectory 종료 이후를 뜻합니다. "
                                                "짧은 상태도 보이도록 처음에는 자동 확대하며, "
                                                "하단 범위 조절기 또는 전체 보기 버튼으로 시간축을 "
                                                "이동할 수 있습니다. 마우스를 올리면 상태의 "
                                                "시작·종료·지속시간을 볼 수 있습니다.",
                                                className="muted-copy",
                                            ),
                                        ],
                                        className="panel-heading",
                                    ),
                                    dcc.Graph(
                                        id="input-gantt",
                                        figure=empty_chart,
                                        config=_GRAPH_CONFIG,
                                    ),
                                ],
                                className="panel input-gantt-panel",
                            ),
                            html.Section(
                                [
                                    html.Div(
                                        [
                                            html.H3("입력 형상 및 로봇 작업 공간"),
                                            html.P(id="preview-note", className="muted-copy"),
                                        ],
                                        className="panel-heading",
                                    ),
                                    dcc.Graph(
                                        id="input-scene", figure=empty_scene, config=_GRAPH_CONFIG
                                    ),
                                ],
                                className="panel preview-panel",
                            ),
                            html.Div(
                                [
                                    html.Section(
                                        [
                                            html.H3("로봇별 상태 시간 비율"),
                                            html.P(
                                                "각 로봇의 Deposition / Travel / Wait 시간이 "
                                                "전체 Makespan에서 차지하는 비율입니다. 먼저 "
                                                "끝난 로봇의 잔여 시간은 Wait가 아니라 완료 후 "
                                                "비활성으로 구분합니다.",
                                                className="muted-copy",
                                            ),
                                            dcc.Graph(
                                                id="input-time-chart",
                                                figure=empty_chart,
                                                config=_GRAPH_CONFIG,
                                            ),
                                        ],
                                        className="panel",
                                    ),
                                    html.Section(
                                        [
                                            html.H3("로봇별 경로 길이"),
                                            html.P(
                                                "거리 = 구간 시간 × 실제 평균 속도입니다. Travel이 "
                                                "더 빠르면 시간 비율이 작아도 누적 거리는 더 클 수 "
                                                "있습니다.",
                                                className="muted-copy",
                                            ),
                                            dcc.Graph(
                                                id="input-motion-chart",
                                                figure=empty_chart,
                                                config=_GRAPH_CONFIG,
                                            ),
                                        ],
                                        className="panel",
                                    ),
                                    html.Section(
                                        [
                                            html.H3("로봇별 XY Reach 사용률"),
                                            html.P(
                                                "Robot Base에서 trajectory의 가장 먼 TCP까지의 "
                                                "XY 거리 ÷ 설정 XY Reach입니다. 100%를 초과하면 "
                                                "XY Reach 판정이 FAIL입니다. Z는 판정에서 "
                                                "제외됩니다.",
                                                className="muted-copy",
                                            ),
                                            dcc.Graph(
                                                id="input-reach-chart",
                                                figure=empty_chart,
                                                config=_GRAPH_CONFIG,
                                            ),
                                        ],
                                        className="panel",
                                    ),
                                ],
                                className="input-chart-grid",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Validation 실행",
                                        id="validation-button",
                                        type="button",
                                        disabled=True,
                                    ),
                                    html.Button(
                                        "최근 결과 보기",
                                        id="recent-result-button",
                                        disabled=True,
                                        className="secondary-button",
                                    ),
                                    html.Span(
                                        id="recent-result-readiness",
                                        className="field-message",
                                    ),
                                    html.Span(id="validation-readiness", className="field-message"),
                                    html.Span(
                                        id="validation-start-message",
                                        className="field-message validation-start-message",
                                    ),
                                ],
                                className="input-footer-actions",
                            ),
                        ],
                    ),
                    _screen(
                        "validator-screen is-hidden",
                        "progress-screen",
                        [
                            html.Div(
                                [
                                    html.P("검증 진행", className="eyebrow"),
                                    html.H2(
                                        id="progress-title",
                                        children="Validation을 시작하고 있습니다",
                                    ),
                                    html.P(id="progress-output", className="muted-copy"),
                                ],
                                className="screen-heading",
                            ),
                            html.Section(
                                [
                                    html.Div(
                                        [
                                            html.Strong(id="overall-progress-label", children="0%"),
                                            html.Span(
                                                id="progress-stage", children="프로세스 시작"
                                            ),
                                        ],
                                        className="progress-heading",
                                    ),
                                    html.Progress(id="overall-progress", value="0", max="100"),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Span("단계 진행률"),
                                                    html.Strong(
                                                        id="stage-progress-label", children="0%"
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                [
                                                    html.Span("처리량"),
                                                    html.Strong(id="progress-units", children="—"),
                                                ]
                                            ),
                                            html.Div(
                                                [
                                                    html.Span("경과 시간"),
                                                    html.Strong(
                                                        id="progress-elapsed", children="0.0 s"
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                [
                                                    html.Span("예상 잔여"),
                                                    html.Strong(
                                                        id="progress-eta", children="계산 중"
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="progress-metrics",
                                    ),
                                    html.P(id="progress-message", className="progress-message"),
                                ],
                                className="progress-card",
                            ),
                            html.Pre(id="progress-log", className="live-log"),
                        ],
                    ),
                    _screen(
                        "validator-screen is-hidden",
                        "result-screen",
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.P("검증 결과", className="eyebrow"),
                                            html.H2("검증 결과"),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "처음으로 돌아가기",
                                                id="new-input-button",
                                                className="secondary-button",
                                            ),
                                        ],
                                        className="result-actions",
                                    ),
                                ],
                                className="result-topline",
                            ),
                            html.Div(id="result-content"),
                        ],
                    ),
                ],
                className="validator-main",
            ),
        ],
        className="validator-shell",
    )


def _format_size(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:,.1f} {unit}"
        size /= 1024.0
    return f"{size:,.1f} GB"


def _replay_estimate_text(job: JobRecord, run: RunRecord, interval_s: float) -> str:
    try:
        estimate = estimate_replay(job, run, interval_s)
    except (DashboardDataError, OSError, ValueError) as exc:
        return str(exc)
    return (
        f"예상 {estimate.frame_count:,} frames · "
        f"생성 시간 {estimate.time_low_s:,.0f}–{estimate.time_high_s:,.0f} s · "
        f"파일 크기 {_format_size(estimate.size_low_bytes)}–"
        f"{_format_size(estimate.size_high_bytes)}"
        + (f" · {estimate.warning}" if estimate.warning else "")
    )


def _stl_estimate_text(run: RunRecord) -> str:
    estimate = estimate_deposited_stl(run)
    return (
        f"예상 생성 시간 {estimate.time_low_s:,.0f}–{estimate.time_high_s:,.0f} s · "
        f"파일 크기 {_format_size(estimate.size_low_bytes)}–"
        f"{_format_size(estimate.size_high_bytes)}"
    )


def _preview_note(preview: Any) -> str:
    point_text = ", ".join(
        f"R{robot_id} {count:,}점" for robot_id, count in preview.trajectory_points_shown.items()
    )
    base = (
        f"Target {preview.target_faces_shown:,}/{preview.target_faces_total:,} faces · "
        f"Trajectory preview {point_text}"
    )
    return base + (" · " + " ".join(preview.warnings) if preview.warnings else "")


def _artifact_url(token: str, run: RunRecord, filename: str) -> str:
    return f"/artifacts/{quote(token)}/{quote(run.run_name)}/{quote(filename)}"


def _grid(
    grid_id: str,
    columns: list[tuple[str, str]],
    *,
    rows: list[JsonDict] | None = None,
    page_size: int = 20,
    server_side: bool = False,
) -> Any:
    options: JsonDict = {
        "pagination": True,
        "paginationPageSize": page_size,
        "paginationPageSizeSelector": False,
        "domLayout": "autoHeight",
    }
    if server_side:
        options["cacheBlockSize"] = page_size
    properties: JsonDict = {
        "id": grid_id,
        "columnDefs": [
            {"headerName": label, "field": field, "sortable": True, "resizable": True}
            for label, field in columns
        ],
        "defaultColDef": {"filter": False, "minWidth": 110},
        "dashGridOptions": options,
        "className": "ag-theme-quartz-dark result-grid",
        "style": {"width": "100%"},
    }
    if not server_side:
        properties["rowData"] = rows or []
    else:
        properties["rowModelType"] = "infinite"
    return dag.AgGrid(**properties)


def _layer_display_rows(rows: list[JsonDict]) -> list[JsonDict]:
    """Format layer metrics for scanning while retaining numeric CSV sorting."""
    formatted: list[JsonDict] = []
    for row in rows:
        formatted.append(
            {
                "layer_index": int(row.get("layer_index", 0)),
                "z_slice_mm": f"{float(row.get('z_slice_mm', 0.0)):,.2f}",
                "coverage": f"{float(row.get('coverage', 0.0)):.4%}",
                "underfill_ratio": f"{float(row.get('underfill_ratio', 0.0)):.4%}",
                "overfill_ratio": (
                    "—"
                    if row.get("overfill_ratio") is None
                    else f"{float(row['overfill_ratio']):.4%}"
                ),
                "iou": f"{float(row.get('iou', 0.0)):.4%}",
                "passed": "PASS" if bool(row.get("passed")) else "FAIL",
            }
        )
    return formatted


def _collision_display_rows(rows: list[JsonDict]) -> list[JsonDict]:
    """Format collision events consistently for initial and paged table data."""
    formatted: list[JsonDict] = []
    for row in rows:
        formatted.append(
            {
                "event_id": int(row.get("event_id", 0)),
                "type": str(row.get("type", "")),
                "robot_a": f"R{int(row.get('robot_a', 0))}",
                "robot_b": f"R{int(row.get('robot_b', 0))}",
                "start_s": f"{float(row.get('start_s', 0.0)):,.3f}",
                "end_s": f"{float(row.get('end_s', 0.0)):,.3f}",
                "duration_s": f"{float(row.get('duration_s', 0.0)):,.3f}",
                "minimum_distance_mm": f"{float(row.get('minimum_distance_mm', 0.0)):,.2f}",
                "required_distance_mm": f"{float(row.get('required_distance_mm', 0.0)):,.2f}",
                "minimum_safety_margin_mm": (
                    f"{float(row.get('minimum_safety_margin_mm', 0.0)):,.2f}"
                ),
                "minimum_capsule_surface_clearance_mm": (
                    "—"
                    if row.get("minimum_capsule_surface_clearance_mm") is None
                    else f"{float(row['minimum_capsule_surface_clearance_mm']):,.2f}"
                ),
                "minimum_distance_time_s": (
                    f"{float(row.get('minimum_distance_time_s', 0.0)):,.3f}"
                ),
                "closest_a_xy_mm": (
                    f"({float(row.get('closest_a_x_mm', 0.0)):,.2f}, "
                    f"{float(row.get('closest_a_y_mm', 0.0)):,.2f})"
                ),
                "closest_b_xy_mm": (
                    f"({float(row.get('closest_b_x_mm', 0.0)):,.2f}, "
                    f"{float(row.get('closest_b_y_mm', 0.0)):,.2f})"
                ),
            }
        )
    return formatted


def _status_metric_card(label: str, passed: object, detail: str) -> html.Div:
    verdict = "✓ PASS" if passed is True else "✕ FAIL" if passed is False else "— 확인 불가"
    tone = "pass" if passed is True else "fail" if passed is False else "unknown"
    return html.Div(
        [
            html.Div(label, className="metric-label"),
            html.Div(verdict, className="metric-value"),
            html.Div(detail, className="metric-detail"),
        ],
        className=f"metric-card result-primary-card check-{tone}",
    )


def _duration_label(seconds: float) -> str:
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}시간 {minutes:02d}분 {remaining_seconds:02d}초"
    if minutes:
        return f"{minutes}분 {remaining_seconds:02d}초"
    return f"{remaining_seconds}초"


def _pair_label(value: object) -> str:
    if isinstance(value, list) and len(value) == 2:
        return f"R{value[0]}–R{value[1]}"
    return "Robot pair 확인 불가"


def _robot_result_rows(robot_rows: list[JsonDict], reach_rows: list[JsonDict]) -> list[JsonDict]:
    reach_by_robot = {
        int(row.get("robot_id", 0)): row
        for row in reach_rows
        if isinstance(row.get("robot_id"), int | float | str)
    }
    combined: list[JsonDict] = []
    for robot in robot_rows:
        robot_id = int(robot.get("robot_id", 0))
        reach = reach_by_robot.get(robot_id, {})
        completion_s = float(robot.get("completion_s", 0.0))
        combined.append(
            {
                "robot_id": f"R{robot_id}",
                "completion_s": f"{completion_s:,.2f} s · {_duration_label(completion_s)}",
                "state_time_s": (
                    f"{float(robot.get('deposition_time_s', 0.0)):,.1f} / "
                    f"{float(robot.get('travel_time_s', 0.0)):,.1f} / "
                    f"{float(robot.get('wait_time_s', 0.0)):,.1f}"
                ),
                "path_length_mm": (
                    f"{float(robot.get('deposition_length_mm', 0.0)):,.1f} / "
                    f"{float(robot.get('travel_length_mm', 0.0)):,.1f}"
                ),
                "mean_speed_mm_s": (
                    f"{float(robot.get('mean_deposition_speed_mm_s', 0.0) or 0.0):,.2f} / "
                    f"{float(robot.get('mean_travel_speed_mm_s', 0.0) or 0.0):,.2f}"
                ),
                "xy_reach_use": (
                    f"{float(reach.get('maximum_xy_distance_mm', 0.0)):,.1f} / "
                    f"{float(reach.get('xy_reach_radius_mm', 0.0)):,.1f} mm · "
                    f"{float(reach.get('xy_utilization_ratio', 0.0)):.1%}"
                ),
                "xy_reach_margin_mm": (
                    f"{float(reach.get('minimum_xy_margin_mm', 0.0)):,.2f}"
                ),
                "xy_reach_result": "PASS" if bool(reach.get("passed")) else "FAIL",
            }
        )
    return combined


def _collision_evidence_card(label: str, data: JsonDict, *, arm: bool) -> html.Div:
    enabled = bool(data.get("enabled", False))
    passed = bool(data.get("passed", False))
    distance_key = "centerline_distance_at_worst_mm" if arm else "minimum_distance_mm"
    required_key = "required_distance_at_worst_mm" if arm else "required_distance_at_minimum_mm"
    distance = float(data.get(distance_key, 0.0))
    required = float(data.get(required_key, 0.0))
    margin = distance - required
    verdict = "검사 꺼짐" if not enabled else "PASS" if passed else "FAIL"
    tone = "unknown" if not enabled else "pass" if passed else "fail"
    time_s = float(data.get("time_s", 0.0))
    return html.Div(
        [
            html.Div(
                [html.H3(label), html.Strong(verdict)],
                className="safety-card-heading",
            ),
            html.Div(
                [
                    html.Div([html.Span("측정 최소거리"), html.Strong(f"{distance:,.2f} mm")]),
                    html.Div([html.Span("요구 최소거리"), html.Strong(f"{required:,.2f} mm")]),
                    html.Div([html.Span("안전 여유"), html.Strong(f"{margin:,.2f} mm")]),
                ],
                className="safety-values",
            ),
            html.P(
                f"{distance:,.2f} − {required:,.2f} = {margin:,.2f} mm",
                className="safety-equation",
            ),
            html.P(
                f"최악 조건: {_pair_label(data.get('pair'))} · {time_s:,.3f} s · "
                f"이벤트 {int(data.get('event_count', 0)):,}건",
                className="muted-copy",
            ),
        ],
        className=f"safety-card check-{tone}",
    )


def _result_findings_content(payload: JsonDict) -> html.Div:
    reasons = payload.get("failure_reasons", [])
    reason_list = [str(reason) for reason in reasons] if isinstance(reasons, list) else []
    raw_issues = payload.get("issues", [])
    issues = (
        [item for item in raw_issues if isinstance(item, dict)]
        if isinstance(raw_issues, list)
        else []
    )
    warnings = [item for item in issues if item.get("severity") == "warning"]
    violations = [item for item in issues if item.get("severity") == "violation"]
    children: list[Any] = []
    if reason_list:
        children.extend(
            [
                html.H4("FAIL 사유"),
                html.Ol([html.Li(reason) for reason in reason_list], className="failure-list"),
            ]
        )
    else:
        children.append(
            html.Div(
                [html.Strong("PASS"), html.Span("활성화된 모든 판정 기준을 충족했습니다.")],
                className="result-clear-message",
            )
        )
    if warnings:
        children.extend(
            [
                html.H4("경고와 추가 확인사항"),
                _issues_content({"issues": warnings}),
            ]
        )
    if violations:
        violation_content = _issues_content({"issues": violations})
        if reason_list:
            children.append(
                html.Details(
                    [
                        html.Summary(f"FAIL 세부 판정 기록 {len(violations):,}건 보기"),
                        violation_content,
                    ],
                    className="result-issue-details",
                )
            )
        else:
            children.extend([html.H4("세부 판정 기록"), violation_content])
    if not reason_list and not issues:
        children.append(html.P("별도로 확인할 경고나 위반이 없습니다.", className="muted-copy"))
    return html.Div(children, className="result-findings")


def _arm_envelope_snapshot(payload: JsonDict, config: Config | None) -> go.Figure:
    collision = payload.get("collision")
    arm = collision.get("arm_envelope") if isinstance(collision, dict) else None
    if not isinstance(arm, dict):
        return _empty_figure("Arm Envelope 최악 시점 정보가 없습니다", height=520)
    tcp_positions = arm.get("tcp_positions_xy_mm")
    closest = arm.get("closest_points_xy_mm")
    if not isinstance(tcp_positions, list) or len(tcp_positions) != 3:
        return _empty_figure("Arm Envelope TCP 위치 정보가 없습니다", height=520)
    if config is None:
        return _empty_figure("Validation 당시 Config snapshot이 없습니다", height=520)
    try:
        figure = go.Figure()
        workspace_x, workspace_y = _capsule_xy(
            config.workspace.center_xy_mm,
            config.workspace.center_xy_mm,
            config.workspace.radius_mm,
        )
        figure.add_trace(
            go.Scatter(
                x=workspace_x,
                y=workspace_y,
                mode="lines",
                fill="toself",
                fillcolor="rgba(56,189,248,0.06)",
                line={"color": "rgba(56,189,248,0.55)", "dash": "dot"},
                name="Workspace",
            )
        )
        robots_by_id = tuple(config.robot(robot_id) for robot_id in (1, 2, 3))
        bases = [robot.base_xyz_mm[:2] for robot in robots_by_id]
        triangle = [*bases, bases[0]]
        figure.add_trace(
            go.Scatter(
                x=[point[0] for point in triangle],
                y=[point[1] for point in triangle],
                mode="lines+markers",
                line={"color": "#64748b", "dash": "dot"},
                marker={"symbol": "triangle-up", "size": 9},
                name="Robot bases",
            )
        )
        colors = ("#2f8fff", "#ff9d42", "#34d399")
        for index, (robot, tcp_raw) in enumerate(zip(robots_by_id, tcp_positions, strict=True)):
            if not isinstance(tcp_raw, list) or len(tcp_raw) != 2:
                continue
            tcp = (float(tcp_raw[0]), float(tcp_raw[1]))
            physical_x, physical_y = _capsule_xy(
                robot.base_xyz_mm[:2], tcp, robot.arm_envelope_radius_mm
            )
            decision_x, decision_y = _capsule_xy(
                robot.base_xyz_mm[:2],
                tcp,
                robot.arm_envelope_radius_mm + config.collision.arm_clearance_mm / 2.0,
            )
            figure.add_trace(
                go.Scatter(
                    x=physical_x,
                    y=physical_y,
                    mode="lines",
                    fill="toself",
                    fillcolor=(
                        f"rgba({int(colors[index][1:3], 16)},"
                        f"{int(colors[index][3:5], 16)},"
                        f"{int(colors[index][5:7], 16)},0.22)"
                    ),
                    line={"color": colors[index], "width": 1},
                    name=f"R{robot.id} 실제 Capsule",
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=decision_x,
                    y=decision_y,
                    mode="lines",
                    line={"color": colors[index], "width": 2, "dash": "dash"},
                    name=f"R{robot.id} 판정 외곽선",
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=[robot.base_xyz_mm[0], tcp[0]],
                    y=[robot.base_xyz_mm[1], tcp[1]],
                    mode="lines+markers",
                    line={"color": colors[index], "width": 2},
                    marker={"size": 6},
                    name=f"R{robot.id} 중심선",
                    showlegend=False,
                )
            )
        if isinstance(closest, list) and len(closest) == 2:
            left, right = closest
            if isinstance(left, list) and isinstance(right, list):
                passed = bool(arm.get("passed", False))
                figure.add_trace(
                    go.Scatter(
                        x=[left[0], right[0]],
                        y=[left[1], right[1]],
                        mode="lines+markers",
                        line={"color": "#21d4a3" if passed else "#ff6376", "width": 4},
                        marker={"size": 8},
                        name="최단 중심선 거리",
                    )
                )
        pair = arm.get("pair", [])
        pair_label = f"R{pair[0]}–R{pair[1]}" if isinstance(pair, list) and len(pair) == 2 else "—"
        figure.update_layout(
            title=(
                f"{pair_label} · {float(arm.get('time_s', 0.0)):,.3f} s · "
                f"안전 여유 {float(arm.get('minimum_safety_margin_mm', 0.0)):,.3f} mm · "
                f"{'PASS' if arm.get('passed') else 'FAIL'}"
            ),
            xaxis_title="X [mm]",
            yaxis_title="Y [mm]",
        )
        x_values = [
            float(value) for trace in figure.data for value in (trace.x or ()) if value is not None
        ]
        y_values = [
            float(value) for trace in figure.data for value in (trace.y or ()) if value is not None
        ]
        if x_values and y_values:
            x_span = max(x_values) - min(x_values)
            y_span = max(y_values) - min(y_values)
            padding = max(50.0, max(x_span, y_span) * 0.05)
            figure.update_xaxes(
                range=[min(x_values) - padding, max(x_values) + padding],
                constrain="domain",
            )
            figure.update_yaxes(
                range=[min(y_values) - padding, max(y_values) + padding],
                scaleanchor="x",
                scaleratio=1.0,
                constrain="domain",
            )
        return _style_figure(figure, height=960)
    except (OSError, ValueError):
        return _empty_figure("Arm Envelope 최악 시점을 표시할 수 없습니다", height=960)


def _shape_summary_cards(shape: JsonDict, layer_rows: list[JsonDict]) -> list[html.Div]:
    maximum_underfill = max(
        (float(row.get("underfill_ratio", 0.0) or 0.0) for row in layer_rows),
        default=0.0,
    )
    maximum_overfill = max(
        (float(row.get("overfill_ratio", 0.0) or 0.0) for row in layer_rows),
        default=0.0,
    )
    return [
        _metric_card("전체 Coverage", f"{float(shape.get('coverage', 0.0)):.4%}"),
        _metric_card("전체 IoU", f"{float(shape.get('iou', 0.0)):.4%}"),
        _metric_card("최대 Layer Underfill", f"{maximum_underfill:.4%}"),
        _metric_card("최대 Layer Overfill", f"{maximum_overfill:.4%}"),
        _metric_card(
            "실패 Layer",
            f"{int(shape.get('failed_layer_count', 0)):,} / "
            f"{int(shape.get('evaluated_layer_count', 0)):,}",
            f"비율 {float(shape.get('failed_layer_ratio', 0.0)):.2%}",
        ),
    ]


def _result_view(
    token: str,
    job: JobRecord,
    run: RunRecord,
    selected_tab: str = "overview",
) -> html.Div:
    payload = run.payload
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        return html.Div(
            [
                html.Div(
                    [
                        html.Strong("재검증 필요"),
                        html.Span(job.path.name),
                        html.Span(run.completed_label),
                    ],
                    className="result-status result-error",
                ),
                html.Section(
                    [
                        html.H3("현재 결과 형식이 아닙니다"),
                        html.P(
                            "이 결과는 간결한 결과 bundle과 구조화 issue를 사용하는 schema "
                            f"{RESULT_SCHEMA_VERSION} 형식이 아닙니다. 입력 파일과 기존 "
                            "결과는 그대로 유지되며, "
                            "현재 화면에서 보려면 Validation을 다시 실행해야 합니다."
                        ),
                    ],
                    className="panel error-result-panel",
                ),
            ]
        )
    if run.status == "ERROR":
        artifact_links = [
            html.A(
                [html.Strong(filename), html.Span(_format_size(path.stat().st_size))],
                href=_artifact_url(token, run, filename),
                target="_blank",
                className="artifact-link",
            )
            for filename in ("error.json", "run.log")
            for path in [run.directory / filename]
            if path.is_file()
        ]
        return html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Strong("ERROR"),
                                html.Span(job.path.name),
                                html.Span(run.completed_label),
                            ],
                            className="result-status result-error",
                        ),
                        html.Code(str(run.directory)),
                    ],
                    className="result-banner",
                ),
                html.Section(
                    [
                        html.H3("Validation 실행 오류"),
                        html.Code(
                            str(payload.get("code", "VALIDATION_ERROR")),
                            className="error-code",
                        ),
                        html.P(str(payload.get("message", run.load_error or "알 수 없는 오류"))),
                        html.P(
                            "상세 실행 기록은 아래 error.json과 run.log에서 확인할 수 있습니다.",
                            className="muted-copy",
                        ),
                        html.Div(artifact_links, className="artifact-list"),
                    ],
                    className="panel error-result-panel",
                ),
            ]
        )
    schedule = payload.get("schedule", {}) if isinstance(payload.get("schedule"), dict) else {}
    collision = payload.get("collision", {}) if isinstance(payload.get("collision"), dict) else {}
    shape = payload.get("shape", {}) if isinstance(payload.get("shape"), dict) else {}
    reach = payload.get("reach", {}) if isinstance(payload.get("reach"), dict) else {}
    result_config = load_result_config(run)
    result_thresholds = thresholds_from_config(result_config)
    csv_errors: list[JsonDict] = []

    def load_rows(filename: str) -> list[JsonDict]:
        try:
            return read_csv_records(run.directory, filename)
        except DashboardDataError as exc:
            csv_errors.append(
                {
                    "severity": "warning",
                    "code": "RESULT_CSV_LOAD_ERROR",
                    "message": str(exc),
                    "robot_id": None,
                    "start_s": None,
                    "end_s": None,
                }
            )
            return []

    robot_rows = load_rows("robot_metrics.csv")
    collision_rows = load_rows("collision_events.csv")
    layer_rows = load_rows("layer_metrics.csv")
    if csv_errors:
        payload = {
            **payload,
            "issues": [
                *(payload.get("issues", []) if isinstance(payload.get("issues"), list) else []),
                *csv_errors,
            ],
        }
    status = run.status
    arm = (
        collision.get("arm_envelope", {}) if isinstance(collision.get("arm_envelope"), dict) else {}
    )
    tcp = collision.get("tcp_radius", {}) if isinstance(collision.get("tcp_radius"), dict) else {}
    minimum_tcp = float(tcp.get("minimum_distance_mm", 0.0))
    required_tcp = float(tcp.get("required_distance_at_minimum_mm", 0.0))
    tcp_margin = minimum_tcp - required_tcp
    failed_layer_count = int(shape.get("failed_layer_count", 0))
    evaluated_layer_count = int(shape.get("evaluated_layer_count", 0))
    makespan_s = float(schedule.get("makespan_s", 0.0))
    reach_rows = reach.get("robots", []) if isinstance(reach.get("robots"), list) else []
    typed_reach_rows = [row for row in reach_rows if isinstance(row, dict)]
    worst_reach = min(
        typed_reach_rows,
        key=lambda row: float(row.get("minimum_xy_margin_mm", 0.0)),
        default={},
    )
    worst_reach_margin = float(worst_reach.get("minimum_xy_margin_mm", 0.0))
    overall_coverage = float(shape.get("coverage", 0.0))
    overall_iou = float(shape.get("iou", 0.0))
    target_volume = float(shape.get("target_volume_mm3", 0.0))
    deposited_volume = float(shape.get("deposited_volume_mm3", 0.0))
    underfill_volume = float(
        shape.get(
            "underfill_volume_mm3",
            target_volume * float(shape.get("underfill_ratio", 0.0)),
        )
    )
    overfill_volume = float(
        shape.get(
            "overfill_volume_mm3",
            target_volume * float(shape.get("overfill_ratio", 0.0)),
        )
    )
    arm_events = int(arm.get("event_count", 0))
    tcp_events = int(tcp.get("event_count", 0))
    cards = [
        _metric_card(
            "전체 작업시간",
            _duration_label(makespan_s),
            f"Makespan {makespan_s:,.2f} s",
        ),
        _status_metric_card(
            "Robot XY Reach",
            reach.get("passed"),
            f"최소 XY 여유 {worst_reach_margin:,.2f} mm · "
            f"R{int(worst_reach.get('robot_id', 0))}",
        ),
        _status_metric_card(
            "로봇 간 충돌 안전",
            collision.get("passed"),
            f"이벤트 Arm {arm_events:,}건 / TCP {tcp_events:,}건 · "
            f"최소 여유 Arm {float(arm.get('minimum_safety_margin_mm', 0.0)):,.1f} mm / "
            f"TCP {tcp_margin:,.1f} mm",
        ),
        _status_metric_card(
            "적층 형상",
            shape.get("passed"),
            f"Coverage {overall_coverage:.2%} · "
            f"IoU {overall_iou:.2%} · "
            f"실패 Layer {failed_layer_count:,}/{evaluated_layer_count:,}",
        ),
    ]
    robot_result_rows = _robot_result_rows(robot_rows, typed_reach_rows)
    artifact_links = []
    for filename in sorted(ALLOWED_ARTIFACTS):
        path = run.directory / filename
        if path.is_file():
            artifact_links.append(
                html.A(
                    [
                        html.Div(
                            [
                                html.Strong(_ARTIFACT_LABELS.get(filename, filename)),
                                html.Small(filename),
                            ]
                        ),
                        html.Span(_format_size(path.stat().st_size)),
                    ],
                    href=_artifact_url(token, run, filename),
                    target="_blank",
                    className="artifact-link",
                )
            )
    has_replay = (run.directory / "replay.html").is_file()
    has_deposited_stl = (run.directory / "deposited.stl").is_file()
    replay = (
        html.Iframe(
            src=_artifact_url(token, run, "replay.html"),
            title="WAAM 3D replay",
            className="replay-frame",
        )
        if has_replay
        else html.Div("이 결과에는 Replay가 없습니다.", className="empty-state")
    )
    replay_status_payload = read_replay_status(run.directory) or {}
    replay_verdict = str(replay_status_payload.get("verdict", "")).upper()
    replay_state = str(replay_status_payload.get("state", "IDLE")).upper()
    replay_status_message = str(replay_status_payload.get("message", ""))
    if replay_verdict == "ERROR":
        retained = " 기존 Replay 파일은 유지했습니다." if has_replay else ""
        replay_status_message = f"Replay 생성 실패: {replay_status_message}{retained}"
        replay_status_class = "replay-status is-error"
    elif replay_state == "RUNNING":
        replay_status_message = replay_status_message or "Replay를 생성하고 있습니다."
        replay_status_class = "replay-status is-running"
    elif has_replay:
        replay_status_message = replay_status_message or "Replay 생성이 완료되었습니다."
        replay_status_class = "replay-status is-success"
    else:
        replay_status_message = replay_status_message or "Replay 생성 대기"
        replay_status_class = "replay-status is-idle"
    replay_progress = min(
        1.0,
        max(0.0, float(replay_status_payload.get("progress", 0.0) or 0.0)),
    )
    stl_status_payload = read_deposited_stl_status(run.directory) or {}
    stl_progress = min(1.0, max(0.0, float(stl_status_payload.get("progress", 0.0) or 0.0)))
    if str(stl_status_payload.get("verdict", "")).upper() == "ERROR":
        retained = " 기존 deposited.stl 파일은 유지했습니다." if has_deposited_stl else ""
        stl_status_message = (
            f"적층 STL 생성 실패: {stl_status_payload.get('message', '')}{retained}"
        )
        stl_status_class = "replay-status is-error"
    elif str(stl_status_payload.get("state", "")).upper() == "RUNNING":
        stl_status_message = str(stl_status_payload.get("message", "적층 STL 생성 중입니다."))
        stl_status_class = "replay-status is-running"
    elif has_deposited_stl:
        stl_status_message = "deposited.stl 생성이 완료되었습니다."
        stl_status_class = "replay-status is-success"
    else:
        stl_status_message = "적층 STL 생성 대기"
        stl_status_class = "replay-status is-idle"
    replay_interval = preset_interval_s(float(schedule.get("makespan_s", 0.0)), "standard")
    if collision_rows:
        collision_event_sections: list[html.Section] = [
            html.Section(
                [
                    html.H3("충돌 이벤트 시간축"),
                    dcc.Graph(
                        figure=_collision_timeline(collision_rows),
                        config=_GRAPH_CONFIG,
                    ),
                ],
                className="panel",
            ),
            html.Section(
                [
                    html.H3("충돌 이벤트 상세"),
                    _grid(
                        "collision-table",
                        [
                            ("ID", "event_id"),
                            ("유형", "type"),
                            ("Robot A", "robot_a"),
                            ("Robot B", "robot_b"),
                            ("시작 [s]", "start_s"),
                            ("종료 [s]", "end_s"),
                            ("지속 [s]", "duration_s"),
                            ("측정 거리 [mm]", "minimum_distance_mm"),
                            ("요구 거리 [mm]", "required_distance_mm"),
                            ("안전 여유 [mm]", "minimum_safety_margin_mm"),
                            ("최악 시각 [s]", "minimum_distance_time_s"),
                        ],
                        page_size=20,
                        server_side=True,
                    ),
                ],
                className="panel",
            ),
        ]
    else:
        collision_event_sections = [
            html.Section(
                [
                    html.H3("충돌 이벤트"),
                    html.Div(
                        "기록된 Arm Envelope 또는 TCP Radius 충돌 이벤트가 없습니다.",
                        className="empty-state success-border",
                    ),
                ],
                className="panel panel-span compact-panel",
            )
        ]
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Strong(status),
                            html.Span(job.path.name),
                            html.Span(run.completed_label),
                        ],
                        className=f"result-status result-{status.lower()}",
                    ),
                    html.Code(str(run.directory)),
                ],
                className="result-banner",
            ),
            html.Section(cards, className="metric-grid result-metrics"),
            dcc.Tabs(
                id="result-tabs",
                value=(
                    selected_tab
                    if selected_tab in {"overview", "robots", "collision", "shape", "artifacts"}
                    else "overview"
                ),
                className="detail-tabs",
                children=[
                    dcc.Tab(
                        label="판정 요약",
                        value="overview",
                        children=html.Div(
                            [
                                html.Section(
                                    [
                                        html.H3("판정 결과와 확인사항"),
                                        _result_findings_content(payload),
                                    ],
                                    className="panel panel-span result-findings-panel",
                                ),
                            ],
                            className="overview-grid",
                        ),
                    ),
                    dcc.Tab(
                        label="로봇·일정",
                        value="robots",
                        children=html.Div(
                            [
                                html.Section(
                                    [
                                        html.H3("로봇별 상태 시간 비율"),
                                        html.P(
                                            "Deposition(적층), Travel(비적층 이동), "
                                            "Wait(위치 유지 대기), 완료 후 비활성의 "
                                            "Makespan 대비 시간입니다.",
                                            className="muted-copy",
                                        ),
                                        dcc.Graph(
                                            figure=_robot_figure(robot_rows), config=_GRAPH_CONFIG
                                        ),
                                    ],
                                    className="panel panel-span",
                                ),
                                html.Section(
                                    [
                                        html.H3("로봇별 작업·XY Reach 상세"),
                                        html.P(
                                            "D/T/W 값은 상태별 시간, D/T 값은 적층/비적층 "
                                            "이동을 뜻합니다. XY Reach는 Z를 제외한 최대 "
                                            "수평거리와 설정 한계를 한 항목에서 비교합니다.",
                                            className="muted-copy",
                                        ),
                                        _grid(
                                            "robot-metrics-table",
                                            [
                                                ("Robot", "robot_id"),
                                                ("완료", "completion_s"),
                                                ("D/T/W 시간 [s]", "state_time_s"),
                                                ("D/T 거리 [mm]", "path_length_mm"),
                                                ("D/T 평균속도", "mean_speed_mm_s"),
                                                ("최대/한계 XY Reach", "xy_reach_use"),
                                                ("XY Reach 여유 [mm]", "xy_reach_margin_mm"),
                                                ("XY Reach 판정", "xy_reach_result"),
                                            ],
                                            rows=robot_result_rows,
                                            page_size=10,
                                        ),
                                    ],
                                    className="panel panel-span",
                                ),
                            ],
                            className="overview-grid",
                        ),
                    ),
                    dcc.Tab(
                        label="충돌 안전",
                        value="collision",
                        children=html.Div(
                            [
                                html.Section(
                                    [
                                        html.H3("안전거리 판정"),
                                        html.P(
                                            "측정 최소거리가 요구 최소거리보다 크면 안전 여유가 "
                                            "양수입니다. Arm은 Base–TCP Capsule 중심선, TCP는 "
                                            "두 끝점 사이의 XY 거리를 검사합니다.",
                                            className="muted-copy",
                                        ),
                                        html.Div(
                                            [
                                                _collision_evidence_card(
                                                    "Arm Envelope 안전거리", arm, arm=True
                                                ),
                                                _collision_evidence_card(
                                                    "TCP Radius 안전거리", tcp, arm=False
                                                ),
                                            ],
                                            className="safety-card-grid",
                                        ),
                                    ],
                                    className="panel panel-span safety-summary-panel",
                                ),
                                html.Section(
                                    [
                                        html.H3("Arm Envelope 최악 시점"),
                                        html.P(
                                            "채움 영역은 실제 Capsule, 점선은 공통 안전거리의 "
                                            "절반을 추가한 판정 외곽선입니다. 연결선은 두 중심선의 "
                                            "최단거리를 나타냅니다.",
                                            className="muted-copy",
                                        ),
                                        dcc.Graph(
                                            figure=_arm_envelope_snapshot(payload, result_config),
                                            config=_GRAPH_CONFIG,
                                            className="arm-snapshot-graph",
                                        ),
                                    ],
                                    className="panel panel-span",
                                ),
                                *collision_event_sections,
                            ],
                            className="overview-grid",
                        ),
                    ),
                    dcc.Tab(
                        label="형상",
                        value="shape",
                        children=html.Div(
                            [
                                html.Section(
                                    [
                                        html.H3("형상 판정과 핵심 지표"),
                                        html.P(
                                            "Coverage는 Target이 채워진 비율, IoU는 Target과 "
                                            "Deposition의 전체 겹침 정도입니다. Underfill과 "
                                            "Overfill은 낮을수록 좋습니다.",
                                            className="muted-copy",
                                        ),
                                        html.Div(
                                            _shape_summary_cards(shape, layer_rows),
                                            className="metric-grid shape-summary-grid",
                                        ),
                                        html.P(
                                            f"Layer 누계 체적 · Target "
                                            f"{target_volume:,.1f} mm³ / "
                                            f"Deposition {deposited_volume:,.1f} mm³ / "
                                            f"Underfill {underfill_volume:,.1f} mm³ / "
                                            f"Overfill {overfill_volume:,.1f} mm³",
                                            className="shape-volume-note",
                                        ),
                                        html.Div(
                                            [
                                                html.H4("형상 판정 기준"),
                                                html.Div(
                                                    _threshold_content(result_thresholds),
                                                    className="threshold-details",
                                                ),
                                            ],
                                            className="shape-threshold-details",
                                        ),
                                    ],
                                    className="panel panel-span shape-summary-panel",
                                ),
                                html.Section(
                                    [
                                        html.H3("Layer별 일치율과 오차"),
                                        html.P(
                                            "위 그래프는 Coverage·IoU, 아래 그래프는 작은 "
                                            "Underfill·Overfill·IoU 손실을 확대합니다. "
                                            "Layer 판정은 IoU 기준을 사용합니다.",
                                            className="muted-copy",
                                        ),
                                        html.P(
                                            shape_variation_message(layer_rows),
                                            className="shape-volume-note",
                                        ),
                                        dcc.Graph(
                                            figure=_shape_figure(
                                                layer_rows, result_thresholds
                                            ),
                                            config=_GRAPH_CONFIG,
                                        ),
                                    ],
                                    className="panel panel-span",
                                ),
                                html.Section(
                                    [
                                        html.H3("Layer별 수치"),
                                        _grid(
                                            "layer-table",
                                            [
                                                ("Layer", "layer_index"),
                                                ("Z [mm]", "z_slice_mm"),
                                                ("Coverage", "coverage"),
                                                ("Underfill", "underfill_ratio"),
                                                ("Overfill", "overfill_ratio"),
                                                ("IoU", "iou"),
                                                ("판정", "passed"),
                                            ],
                                            page_size=25,
                                            server_side=True,
                                        ),
                                    ],
                                    className="panel panel-span",
                                ),
                            ],
                            className="overview-grid",
                        ),
                    ),
                    dcc.Tab(
                        label="산출물",
                        value="artifacts",
                        children=html.Div(
                            [
                                html.Section(
                                    [
                                        html.H3("결과 파일"),
                                        html.P(
                                            "핵심 결과 파일과 생성이 완료된 추가 산출물을 "
                                            "열거나 내려받을 수 있습니다.",
                                            className="muted-copy",
                                        ),
                                        html.Div(artifact_links, className="artifact-list"),
                                    ],
                                    className=(
                                        "panel panel-span compact-panel artifact-files-panel"
                                    ),
                                ),
                                html.Section(
                                    [
                                        html.H3("추가 산출물 생성"),
                                        html.P(
                                            "Validation은 가벼운 핵심 결과만 저장합니다. 필요한 "
                                            "산출물만 현재 입력 지문을 확인한 뒤 생성합니다.",
                                            className="muted-copy",
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.H4("명목 적층 형상 STL"),
                                                        html.Div(
                                                            _stl_estimate_text(run),
                                                            id="stl-estimate",
                                                        ),
                                                        html.Button(
                                                            (
                                                                "deposited.stl 다시 생성"
                                                                if has_deposited_stl
                                                                else "deposited.stl 생성"
                                                            ),
                                                            id="stl-generate-button",
                                                        ),
                                                        html.Div(
                                                            stl_status_message,
                                                            id="stl-status",
                                                            className=stl_status_class,
                                                        ),
                                                        html.Progress(
                                                            id="stl-progress",
                                                            value=f"{stl_progress:.6g}",
                                                            max="1",
                                                        ),
                                                    ],
                                                    className="derived-artifact-card",
                                                ),
                                                html.Div(
                                                    [
                                                        html.H4("3D Replay"),
                                                        html.Label(
                                                            "프레임 간격 [s]",
                                                            htmlFor="replay-interval",
                                                        ),
                                                        dcc.Input(
                                                            id="replay-interval",
                                                            type="number",
                                                            min=1,
                                                            step=1,
                                                            value=replay_interval,
                                                        ),
                                                        html.Div(
                                                            _replay_estimate_text(
                                                                job, run, replay_interval
                                                            ),
                                                            id="replay-estimate",
                                                        ),
                                                        html.Button(
                                                            (
                                                                "Replay 다시 생성"
                                                                if has_replay
                                                                else "Replay 생성"
                                                            ),
                                                            id="replay-generate-button",
                                                        ),
                                                        html.Div(
                                                            replay_status_message,
                                                            id="replay-status",
                                                            className=replay_status_class,
                                                        ),
                                                        html.Progress(
                                                            id="replay-progress",
                                                            value=f"{replay_progress:.6g}",
                                                            max="1",
                                                        ),
                                                    ],
                                                    className="derived-artifact-card",
                                                ),
                                            ],
                                            className="derived-artifact-grid",
                                        ),
                                        replay,
                                    ],
                                    className="panel replay-panel",
                                ),
                            ],
                            className="overview-grid artifact-tab-grid",
                        ),
                    ),
                ],
            ),
        ]
    )


def create_validator_app(
    initial_job_dir: Path,
    *,
    run_manager: ValidationRunManager | None = None,
    replay_run_manager: ReplayRunManager | None = None,
) -> Dash:
    initial = initial_job_dir.expanduser().resolve()
    registry = _ContextRegistry()
    manager = run_manager or ValidationRunManager()
    replay_manager = replay_run_manager or ReplayRunManager()
    app = Dash(
        __name__,
        assets_folder=str(Path(__file__).with_name("assets")),
        title="WAAM Validator",
        update_title=cast(str, None),
        suppress_callback_exceptions=True,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )
    app.layout = _layout(initial)
    app.server.config["WAAM_INITIAL_JOB_DIR"] = str(initial)

    @app.server.get("/artifacts/<token>/<run_name>/<filename>")  # type: ignore[untyped-decorator]
    def serve_artifact(token: str, run_name: str, filename: str) -> Any:
        try:
            path = resolve_single_job_artifact(registry.get(token), run_name, filename)
        except DashboardDataError:
            abort(404)
        inline = path.suffix.lower() in {".html", ".md", ".log"}
        return send_file(path, as_attachment=not inline, download_name=path.name)

    @app.server.get("/artifacts/<path:_invalid>")  # type: ignore[untyped-decorator]
    def reject_malformed_artifact(_invalid: str) -> Any:
        abort(404)

    def context_run(run_data: JsonDict) -> tuple[Path, RunRecord]:
        token = str(run_data.get("context_id", ""))
        job_dir = registry.get(token)
        output_root = (job_dir / "output").resolve()
        requested = Path(str(run_data.get("output_dir", ""))).expanduser().resolve()
        if requested.parent != output_root:
            raise DashboardDataError("입력 context 밖의 결과입니다.")
        return job_dir, load_run_directory(requested)

    @app.callback(
        Output("job-dir-input", "value"),
        Output("folder-message", "children"),
        Input("browse-button", "n_clicks"),
        State("job-dir-input", "value"),
        prevent_initial_call=True,
    )
    def browse_folder(n_clicks: int | None, current: str | None) -> tuple[Any, str]:
        if not n_clicks:
            return no_update, ""
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "waam_validator.dashboard.folder_picker", current or ""],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return no_update, f"폴더 창을 열 수 없습니다. 경로를 직접 입력하세요: {exc}"
        selected = completed.stdout.strip()
        if not selected:
            return no_update, "폴더 선택을 취소했습니다."
        return selected, "폴더를 선택했습니다. 입력 확인을 실행하세요."

    @app.callback(
        Output("selected-job-dir-store", "data"),
        Input("job-dir-input", "value"),
    )
    def remember_job_directory(value: str | None) -> Any:
        # Some browsers restore the visible input before Dash has populated the
        # component's callback State. Preserve the CLI-provided initial path
        # until a real input value arrives.
        return no_update if value is None else value

    @app.callback(
        Output("active-input-store", "data"),
        Output("recent-run-store", "data"),
        Output("inspection-content", "children"),
        Output("input-gantt", "figure"),
        Output("input-scene", "figure"),
        Output("input-time-chart", "figure"),
        Output("input-motion-chart", "figure"),
        Output("input-reach-chart", "figure"),
        Output("preview-note", "children"),
        Input("inspect-button", "n_clicks"),
        State("job-dir-input", "value"),
        State("selected-job-dir-store", "data"),
        prevent_initial_call=True,
    )
    def inspect_input(
        n_clicks: int | None,
        value: str | None,
        remembered_value: str | None,
    ) -> tuple[Any, ...]:
        if not n_clicks:
            # Dynamic screen updates can re-register this callback with a null
            # click count. Never let that initialization erase a valid result.
            raise PreventUpdate
        if value is None:
            value = remembered_value
        if not value:
            empty = _empty_figure("입력 폴더를 지정하세요.")
            return (
                {},
                {},
                html.Div("입력 폴더를 지정하세요."),
                empty,
                empty,
                empty,
                empty,
                empty,
                "",
            )
        try:
            paths = resolve_input_directory(value)
            inspection, config, trajectories, mesh = inspect_dashboard_input_bundle(paths)
            if config is None or trajectories is None or mesh is None:
                empty = _empty_figure("차단 오류를 수정한 뒤 다시 확인하세요.")
                return (
                    inspection.to_dict(),
                    {},
                    _inspection_content(inspection.to_dict()),
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    "Preview를 생성할 수 없습니다.",
                )
            preview = build_input_preview(config, trajectories, mesh)
            token = registry.add(paths.job_dir, trajectories)
            active = {
                **inspection.to_dict(),
                "context_id": token,
                "recent_result_note": (f"완료된 schema {RESULT_SCHEMA_VERSION} 결과가 없습니다."),
            }
            recent_data: JsonDict = {}
            preview_note = _preview_note(preview)
            recent = load_latest_matching_run(paths.job_dir)
            if recent is not None:
                matches, match_note = verify_validation_inputs(paths.job_dir, recent.directory)
                if matches:
                    recent_data = {
                        "context_id": token,
                        "job_dir": str(paths.job_dir),
                        "output_dir": str(recent.directory),
                        "run_name": recent.run_name,
                        "status": recent.status,
                    }
                    if match_note:
                        preview_note += f" · {match_note}"
                    active["recent_result_note"] = "현재 입력과 일치하는 최근 결과가 있습니다."
            else:
                latest = load_latest_run(paths.job_dir)
                if (
                    latest is not None
                    and latest.payload.get("schema_version") != RESULT_SCHEMA_VERSION
                ):
                    active["recent_result_note"] = (
                        f"완료 결과가 있지만 현재 schema {RESULT_SCHEMA_VERSION} 형식이 "
                        "아니어서 다시 실행해야 합니다."
                    )
                    preview_note += (
                        f" · 기존 결과는 현재 결과 형식(schema {RESULT_SCHEMA_VERSION})이 "
                        "아니므로 "
                        "Validation을 다시 실행해야 합니다."
                    )
                elif latest is not None:
                    _, match_note = verify_validation_inputs(paths.job_dir, latest.directory)
                    active["recent_result_note"] = match_note
            return (
                active,
                recent_data,
                _inspection_content(active),
                preview.gantt_figure,
                preview.scene,
                preview.time_figure,
                preview.motion_figure,
                preview.reach_figure,
                preview_note,
            )
        except (DashboardDataError, OSError, ValueError) as exc:
            empty = _empty_figure("입력 확인에 실패했습니다.")
            content = html.Div(
                [html.Strong("입력 확인 실패"), html.P(str(exc))],
                className="input-verdict is-blocked",
            )
            return {}, {}, content, empty, empty, empty, empty, empty, ""

    @app.callback(
        Output("input-gantt", "figure", allow_duplicate=True),
        Input("input-gantt", "relayoutData"),
        State("active-input-store", "data"),
        prevent_initial_call=True,
    )
    def update_gantt_detail(
        relayout: JsonDict | None,
        active: JsonDict | None,
    ) -> Any:
        if not relayout or not active:
            raise PreventUpdate
        range_value = relayout.get("xaxis.range")
        if isinstance(range_value, list) and len(range_value) == 2:
            start_raw, end_raw = range_value
        else:
            start_raw = relayout.get("xaxis.range[0]")
            end_raw = relayout.get("xaxis.range[1]")
        if not isinstance(start_raw, int | float) or not isinstance(end_raw, int | float):
            raise PreventUpdate
        token = str(active.get("context_id", ""))
        return build_gantt_figure(
            registry.trajectories(token),
            start_s=float(start_raw),
            end_s=float(end_raw),
        )

    @app.callback(
        Output("validation-button", "disabled"),
        Output("validation-readiness", "children"),
        Output("recent-result-button", "disabled"),
        Output("recent-result-readiness", "children"),
        Input("active-input-store", "data"),
        Input("recent-run-store", "data"),
        Input("job-dir-input", "value"),
        Input("selected-job-dir-store", "data"),
        Input("runtime-store", "data"),
    )
    def input_action_state(
        active: JsonDict | None,
        recent: JsonDict | None,
        value: str | None,
        remembered_value: str | None,
        runtime: JsonDict | None,
    ) -> tuple[bool, str, bool, str]:
        if (runtime or {}).get("running"):
            return True, "Validation 실행 중", True, "실행이 끝나면 최근 결과가 갱신됩니다."
        if not active or not active.get("can_run"):
            return (
                True,
                "입력 확인을 완료해야 Validation을 실행할 수 있습니다.",
                True,
                "입력 확인 후 현재 입력과 일치하는 결과를 찾습니다.",
            )
        if value is None:
            value = remembered_value
        try:
            current_path = str(Path(value or "").expanduser().resolve())
        except OSError:
            current_path = ""
        inspected_path = str((active.get("paths") or {}).get("job_dir", ""))
        if current_path != inspected_path:
            return (
                True,
                "폴더가 변경되었습니다. 입력 확인을 다시 실행하세요.",
                True,
                "변경된 폴더의 입력 확인이 필요합니다.",
            )
        status = str(active.get("status", "READY"))
        recent_note = str(
            active.get("recent_result_note") or "현재 입력과 일치하는 최근 결과가 없습니다."
        )
        return False, f"Validation 준비 완료 · 입력 상태 {status}", not bool(recent), recent_note

    @app.callback(
        Output("view-store", "data"),
        Input("recent-result-button", "n_clicks"),
        Input("new-input-button", "n_clicks"),
        State("recent-run-store", "data"),
        prevent_initial_call=True,
    )
    def navigate(
        recent_clicks: int | None,
        new_clicks: int | None,
        recent: JsonDict | None,
    ) -> Any:
        if ctx.triggered_id == "recent-result-button" and recent_clicks and recent:
            return {"view": "result", "source": "recent", "nonce": time.time_ns()}
        if ctx.triggered_id == "new-input-button" and new_clicks:
            return {"view": "input", "nonce": time.time_ns()}
        return no_update

    @app.callback(
        Output("current-run-store", "data", allow_duplicate=True),
        Input("recent-run-store", "data"),
        Input("view-store", "data"),
        prevent_initial_call=True,
    )
    def select_recent_run(recent: JsonDict | None, view: JsonDict | None) -> Any:
        if (view or {}).get("source") == "recent" and recent:
            return recent
        return no_update

    @app.callback(
        Output("runtime-store", "data"),
        Output("validation-poller", "disabled"),
        Output("view-store", "data", allow_duplicate=True),
        Output("current-run-store", "data", allow_duplicate=True),
        Output("recent-run-store", "data", allow_duplicate=True),
        Output("overall-progress", "value"),
        Output("overall-progress-label", "children"),
        Output("stage-progress-label", "children"),
        Output("progress-stage", "children"),
        Output("progress-units", "children"),
        Output("progress-elapsed", "children"),
        Output("progress-eta", "children"),
        Output("progress-message", "children"),
        Output("progress-log", "children"),
        Output("progress-title", "children"),
        Output("progress-output", "children"),
        Output("validation-start-message", "children"),
        Input("validation-button", "n_clicks"),
        Input("validation-poller", "n_intervals"),
        State("active-input-store", "data"),
        prevent_initial_call=True,
    )
    def manage_validation(
        validation_clicks: int | None,
        _ticks: int,
        active: JsonDict | None,
    ) -> tuple[Any, ...]:
        triggered = ctx.triggered_id
        if triggered == "validation-button" and validation_clicks:
            try:
                if not active or not active.get("can_run"):
                    raise DashboardDataError("입력 확인을 다시 실행하세요.")
                token = str(active.get("context_id", ""))
                job_dir = registry.get(token)
                paths = resolve_input_directory(job_dir)
                if not inspection_is_current(active, paths):
                    raise DashboardDataError(
                        "입력 파일이 변경되었습니다. 입력 확인을 다시 실행하세요."
                    )
                if replay_manager.snapshot().get("running"):
                    raise ValidationAlreadyRunningError("추가 산출물 생성이 끝난 뒤 실행하세요.")
                manager.start(JobRecord(job_dir.name, job_dir))
            except (
                DashboardDataError,
                ValidationAlreadyRunningError,
                OSError,
                RuntimeError,
            ) as exc:
                error = {"state": "ERROR", "running": False, "message": str(exc)}
                return (
                    error,
                    True,
                    {"view": "input"},
                    no_update,
                    no_update,
                    0,
                    "0%",
                    "0%",
                    "시작 실패",
                    "—",
                    "0.0 s",
                    "—",
                    str(exc),
                    "",
                    "Validation을 시작하지 못했습니다",
                    "",
                    f"Validation 시작 실패: {exc}",
                )
        snapshot = manager.snapshot()
        running = bool(snapshot.get("running"))
        overall = min(1.0, max(0.0, float(snapshot.get("overall_progress", 0.0) or 0.0)))
        stage_fraction = min(1.0, max(0.0, float(snapshot.get("stage_progress", 0.0) or 0.0)))
        elapsed = float(snapshot.get("elapsed_s", 0.0) or 0.0)
        eta = elapsed * (1.0 - overall) / overall if overall >= 0.02 and running else None
        completed = snapshot.get("completed_units")
        total = snapshot.get("total_units")
        unit = str(snapshot.get("unit") or "")
        units = (
            f"{float(completed):,.0f} / {float(total):,.0f} {unit}"
            if isinstance(completed, int | float) and isinstance(total, int | float)
            else "—"
        )
        run_data: Any = no_update
        recent_data: Any = no_update
        view = {"view": "running", "nonce": time.time_ns()}
        if not running and snapshot.get("state") == "FINISHED":
            output_dir = Path(str(snapshot.get("output_directory", ""))).resolve()
            token = str((active or {}).get("context_id", ""))
            run_data = {
                "context_id": token,
                "job_dir": str(registry.get(token)),
                "output_dir": str(output_dir),
                "run_name": output_dir.name,
                "status": str(snapshot.get("verdict", "ERROR")),
            }
            recent_data = run_data
            view = {"view": "result", "nonce": time.time_ns()}
        stage = str(snapshot.get("stage", "starting"))
        return (
            snapshot,
            not running,
            view,
            run_data,
            recent_data,
            overall * 100.0,
            f"{overall:.0%}",
            f"{stage_fraction:.0%}",
            _STAGE_LABELS.get(stage, stage),
            units,
            f"{elapsed:,.1f} s",
            f"약 {eta:,.0f} s" if eta is not None else "계산 중",
            str(snapshot.get("message", "")),
            str(snapshot.get("log", "")),
            f"{snapshot.get('job_name', '')} Validation",
            str(snapshot.get("output_directory", "")),
            (
                "Validation을 시작했습니다. 진행 화면으로 이동합니다."
                if triggered == "validation-button"
                else no_update
            ),
        )

    @app.callback(
        Output("input-screen", "className"),
        Output("progress-screen", "className"),
        Output("result-screen", "className"),
        Output("workflow-steps", "className"),
        Input("view-store", "data"),
    )
    def switch_screen(view_data: JsonDict | None) -> tuple[str, str, str, str]:
        view = str((view_data or {}).get("view", "input"))
        return (
            "validator-screen" + ("" if view == "input" else " is-hidden"),
            "validator-screen" + ("" if view == "running" else " is-hidden"),
            "validator-screen" + ("" if view == "result" else " is-hidden"),
            f"workflow-steps step-{view}",
        )

    @app.callback(
        Output("result-content", "children"),
        Input("current-run-store", "data"),
        Input("replay-refresh", "data"),
        State("result-tab-store", "data"),
    )
    def render_result(
        run_data: JsonDict | None,
        _refresh: JsonDict | None,
        tab_data: JsonDict | None = None,
    ) -> Any:
        if not run_data:
            return html.Div("표시할 결과가 없습니다.", className="empty-state")
        try:
            token = str(run_data.get("context_id", ""))
            job_dir, run = context_run(run_data)
            selected_tab = "overview"
            if (tab_data or {}).get("run_name") == run.run_name:
                selected_tab = str((tab_data or {}).get("value", "overview"))
            return _result_view(
                token,
                JobRecord(job_dir.name, job_dir),
                run,
                selected_tab,
            )
        except (DashboardDataError, OSError, ValueError) as exc:
            return html.Div(
                [html.Strong("결과 로딩 실패"), html.P(str(exc))],
                className="input-verdict is-blocked",
            )

    @app.callback(
        Output("result-tab-store", "data"),
        Input("result-tabs", "value"),
        State("current-run-store", "data"),
        prevent_initial_call=True,
    )
    def remember_result_tab(
        value: str | None,
        run_data: JsonDict | None,
    ) -> JsonDict:
        if value not in {"overview", "robots", "collision", "shape", "artifacts"} or not run_data:
            raise PreventUpdate
        return {"run_name": str(run_data.get("run_name", "")), "value": value}

    def result_csv_window(
        run_data: JsonDict | None,
        filename: str,
        request: JsonDict | None,
    ) -> JsonDict:
        if not run_data:
            return {"rowData": [], "rowCount": 0}
        try:
            _, run = context_run(run_data)
            request_data = request or {}
            start = max(0, int(request_data.get("startRow", 0) or 0))
            end = max(start + 1, int(request_data.get("endRow", start + 20) or start + 20))
            sort_model = request_data.get("sortModel", [])
            sort_by = (
                [
                    {
                        "column_id": str(item.get("colId", "")),
                        "direction": str(item.get("sort", "asc")),
                    }
                    for item in sort_model
                    if isinstance(item, dict)
                ]
                if isinstance(sort_model, list)
                else []
            )
            rows, row_count = read_csv_window(
                run.directory,
                filename,
                start=start,
                end=end,
                sort_by=sort_by,
            )
            if filename == "collision_events.csv":
                rows = _collision_display_rows(rows)
            elif filename == "layer_metrics.csv":
                rows = _layer_display_rows(rows)
            return {"rowData": rows, "rowCount": row_count}
        except (DashboardDataError, OSError, ValueError):
            return {"rowData": [], "rowCount": 0}

    @app.callback(
        Output("collision-table", "getRowsResponse"),
        Input("collision-table", "getRowsRequest"),
        State("current-run-store", "data"),
        prevent_initial_call=True,
    )
    def collision_table_page(
        request: JsonDict | None,
        run_data: JsonDict | None,
    ) -> JsonDict:
        return result_csv_window(run_data, "collision_events.csv", request)

    @app.callback(
        Output("layer-table", "getRowsResponse"),
        Input("layer-table", "getRowsRequest"),
        State("current-run-store", "data"),
        prevent_initial_call=True,
    )
    def layer_table_page(
        request: JsonDict | None,
        run_data: JsonDict | None,
    ) -> JsonDict:
        return result_csv_window(run_data, "layer_metrics.csv", request)

    @app.callback(
        Output("replay-estimate", "children"),
        Input("replay-interval", "value"),
        State("current-run-store", "data"),
        prevent_initial_call=True,
    )
    def replay_estimate(interval: float | None, run_data: JsonDict | None) -> str:
        if interval is None or not run_data:
            return "프레임 간격을 입력하세요."
        try:
            job_dir, run = context_run(run_data)
            return _replay_estimate_text(JobRecord(job_dir.name, job_dir), run, float(interval))
        except (DashboardDataError, OSError, ValueError) as exc:
            return str(exc)

    @app.callback(
        Output("replay-generate-button", "disabled"),
        Output("stl-generate-button", "disabled"),
        Input("replay-runtime-store", "data"),
    )
    def derived_artifact_buttons(runtime: JsonDict | None) -> tuple[bool, bool]:
        running = bool((runtime or {}).get("running"))
        return running, running

    @app.callback(
        Output("replay-runtime-store", "data"),
        Output("replay-poller", "disabled"),
        Output("replay-status", "children"),
        Output("replay-progress", "value"),
        Output("stl-status", "children"),
        Output("stl-progress", "value"),
        Output("replay-refresh", "data"),
        Input("replay-generate-button", "n_clicks"),
        Input("stl-generate-button", "n_clicks"),
        Input("replay-poller", "n_intervals"),
        State("replay-interval", "value"),
        State("current-run-store", "data"),
        prevent_initial_call=True,
    )
    def manage_replay(
        clicks: int | None,
        stl_clicks: int | None,
        _ticks: int,
        interval: float | None,
        run_data: JsonDict | None,
    ) -> tuple[Any, ...]:
        triggered = ctx.triggered_id
        if triggered in {"replay-generate-button", "stl-generate-button"}:
            try:
                if manager.snapshot().get("running"):
                    raise ReplayAlreadyRunningError("Validation이 끝난 뒤 산출물을 생성하세요.")
                if not run_data:
                    raise DashboardDataError("결과를 확인하세요.")
                job_dir, run = context_run(run_data)
                job = JobRecord(job_dir.name, job_dir)
                if triggered == "replay-generate-button" and clicks:
                    if interval is None:
                        raise DashboardDataError("프레임 간격을 확인하세요.")
                    estimate = estimate_replay(job, run, float(interval))
                    if not estimate.allowed:
                        raise DashboardDataError(estimate.warning)
                    replay_manager.start(job, run, float(interval))
                elif triggered == "stl-generate-button" and stl_clicks:
                    replay_manager.start_deposited_stl(job, run)
                else:
                    raise PreventUpdate
            except (DashboardDataError, ReplayAlreadyRunningError, OSError, ValueError) as exc:
                if triggered == "stl-generate-button":
                    return (
                        {"state": "ERROR", "running": False},
                        True,
                        no_update,
                        no_update,
                        str(exc),
                        0,
                        no_update,
                    )
                return (
                    {"state": "ERROR", "running": False},
                    True,
                    str(exc),
                    0,
                    no_update,
                    no_update,
                    no_update,
                )
        snapshot = replay_manager.snapshot()
        running = bool(snapshot.get("running"))
        progress = min(1.0, max(0.0, float(snapshot.get("progress", 0.0) or 0.0)))
        is_stl = snapshot.get("artifact_kind") == "deposited_stl"
        message = str(snapshot.get("message", ""))
        return (
            snapshot,
            not running,
            no_update if is_stl else message,
            no_update if is_stl else progress,
            message if is_stl else no_update,
            progress if is_stl else no_update,
            no_update if running else {"nonce": time.time_ns()},
        )

    return app


def _open_browser_when_ready(url: str) -> None:
    for _ in range(50):
        time.sleep(0.1)
        try:
            with urlopen(url, timeout=0.2):  # noqa: S310 - fixed local server URL.
                pass
        except OSError:
            continue
        webbrowser.open_new_tab(url)
        return


def run_validator_ui(
    job_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8050,
    open_browser: bool = True,
) -> None:
    resolved = job_dir.expanduser().resolve()
    if not resolved.is_dir():
        raise DashboardDataError(f"입력 폴더가 존재하지 않습니다: {resolved}")
    app = create_validator_app(resolved)
    url = f"http://{host}:{port}/"
    if open_browser:
        threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()
    print(f"WAAM Validator is running on {url}")
    app.run(host=host, port=port, debug=False, use_reloader=False)
