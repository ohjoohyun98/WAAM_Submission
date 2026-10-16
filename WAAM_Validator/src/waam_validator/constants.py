"""Project-wide constants."""

from __future__ import annotations

from typing import Final

import numpy as np

MODE_T: Final[np.uint8] = np.uint8(0)
MODE_D: Final[np.uint8] = np.uint8(1)
MODE_W: Final[np.uint8] = np.uint8(2)

MODE_FROM_TEXT: Final[dict[str, np.uint8]] = {
    "T": MODE_T,
    "D": MODE_D,
    "W": MODE_W,
}
MODE_TO_TEXT: Final[dict[int, str]] = {0: "T", 1: "D", 2: "W"}
ROBOT_IDS: Final[tuple[int, int, int]] = (1, 2, 3)
ROBOT_PAIRS: Final[tuple[tuple[int, int], ...]] = ((1, 2), (1, 3), (2, 3))
CSV_COLUMNS: Final[tuple[str, ...]] = (
    "robot_id",
    "time_s",
    "x_mm",
    "y_mm",
    "z_mm",
    "mode",
)
EXTREME_COORDINATE_WARNING_MM: Final[float] = 1_000_000.0
LIMITATIONS_TEXT: Final[str] = (
    "본 검증기는 각 로봇을 폭이 고정된 Base–TCP 2D Capsule로 단순화하고, XY "
    "평면상 Capsule 안전 여유와 TCP 허용 원 침범만을 로봇 간 충돌로 판단한다. "
    "검사는 adaptive sample 기반이며 연속시간 swept collision을 보증하지 않는다. "
    "실제 로봇 링크, 관절 "
    "자세, Z 방향 분리, 지그 및 환경 충돌은 반영하지 않는다. 형상 검증은 일정한 "
    "비드 폭과 layer 높이를 가정한 명목 기하 모델이며 열변형, 비드 형상 변화, "
    "용융풀 거동 및 공정 불안정성을 예측하지 않는다."
)
