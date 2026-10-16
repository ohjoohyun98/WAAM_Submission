# Multi-Robot WAAM Makespan Optimization

## Overview

세 대의 로봇이 수행하는 WAAM 작업 경로를 생성하고 전체 제조 완료 시간(Makespan)을 최소화하는 Solver입니다.

이 저장소에는 다음 제출물이 포함되어 있습니다.

- 실행 가능한 전체 Solver 소스 코드
- 공개 모델 6개의 `config.yaml`, `target.stl`, `trajectory.csv`
- 비공개 모델에서 `trajectory.csv`를 생성하는 실행 진입점
- 과제에서 제공된 공식 WAAM Validator
- 알고리즘 설명과 공개 모델 결과를 정리한 `REPORT.md`

본 Solver는 학습 모델이 아닌 결정론적 휴리스틱을 사용하므로 checkpoint 파일은 필요하지 않습니다.

## Algorithm

- **Algorithm:** Geometry-aware deterministic Large Neighborhood Search (LNS) scheduler
- **Objective:** 세 로봇의 전체 제조 Makespan 최소화
- **Constraints:** 공식 WAAM Validator의 Reach, Collision, Shape 및 trajectory 형식 요구사항
- **Default settings:** `balanced` mode, seed `11`

Solver는 입력 형상에서 레이어별 적층 작업을 생성하고, 로봇별 작업 배정과 순서를 구성한 뒤 유효성을 유지하면서 작업 병합·재배정·순서 변경 후보를 탐색합니다. 생성된 후보는 공식 Validator로 검사하며 PASS 가능한 해를 최종 출력합니다.

## Requirements

- Python 3.11
- Windows 10/11, macOS 또는 Linux
- 최초 설치 시 Python 패키지를 내려받기 위한 인터넷 연결

## Installation

저장소를 clone하거나 ZIP으로 내려받은 후 저장소 최상위 폴더에서 한 번만 실행합니다. 별도로 가상환경을 활성화할 필요는 없습니다.

### Windows PowerShell

```powershell
python setup_environment.py
```

### macOS / Linux

```bash
python3.11 setup_environment.py
```

설치 스크립트는 운영체제에 맞는 `WAAM_Validator/.venv`를 생성하고 `requirements.txt`에 연결된 고정 의존성 및 동봉된 공식 Validator를 설치합니다.

## Input

새 모델의 입력 폴더에는 교수님이 제공한 다음 두 파일이 있어야 합니다.

```text
input_model/
├── config.yaml
└── target.stl
```

`config.yaml`과 `target.stl`은 수정하지 않고 사용합니다. 출력 폴더는 실행 전에 존재하지 않는 새 경로여야 합니다.

## Execution

### Windows PowerShell

```powershell
python run_solver.py `
  --scenario "C:\path\to\input_model" `
  --output ".\results\input_model" `
  --mode balanced `
  --seed 11 `
  --validate
```

### macOS / Linux

```bash
python3.11 run_solver.py \
  --scenario "/path/to/input_model" \
  --output "./results/input_model" \
  --mode balanced \
  --seed 11 \
  --validate
```

성공하면 출력 폴더에 다음 핵심 파일이 생성됩니다.

- `config.yaml`: 입력 원본 복사본
- `target.stl`: 입력 원본 복사본
- `trajectory.csv`: Solver가 생성한 최종 경로
- `solver_result.json`: Solver 실행 결과
- `validator_check.log`, `validator_run.log`: 공식 Validator 검증 기록

## Results

공개 WAAM benchmark 6개 결과는 `public_models/01`부터 `public_models/06`에 있습니다. 각 폴더에는 제출에 필요한 세 파일만 포함되어 있습니다.

| Model | Makespan (s) | Collision events | Validator |
|---|---:|---:|---|
| 01 | 1193.38 | 0 | PASS |
| 02 | 3344.49 | 0 | PASS |
| 03 | 1872.36 | 0 | PASS |
| 04 | 52823.48 | 0 | PASS |
| 05 | 125547.45 | 0 | PASS |
| 06 | 6105.02 | 0 | PASS |

## Validation

공개 모델 6개를 공식 Validator의 `check`와 `run`으로 일괄 검증합니다.

### Windows PowerShell

```powershell
python verify_public.py
```

### macOS / Linux

```bash
python3.11 verify_public.py
```

빠른 입력 검사만 수행하려면 다음 옵션을 사용합니다.

```shell
python verify_public.py --check-only
```

개별 결과를 직접 검증할 수도 있습니다.

```shell
python validate.py check public_models/01
python validate.py run public_models/01 --output verification_results/model_01
```

`trajectory.csv`의 필수 헤더는 다음과 같습니다.

```csv
robot_id,time_s,x_mm,y_mm,z_mm,mode
```

## Repository Structure

```text
.
├── README.md
├── REPORT.md
├── requirements.txt
├── setup_environment.py
├── run_solver.py
├── validate.py
├── verify_public.py
├── public_models/
│   └── 01/ ... 06/
│       ├── config.yaml
│       ├── target.stl
│       └── trajectory.csv
├── solver/
│   └── Solver 전체 소스 코드
└── WAAM_Validator/
    └── 공식 Validator 소스 코드와 라이선스
```

알고리즘 의사코드, 상세 결과 및 평가 항목 대응은 `REPORT.md`에 정리되어 있습니다.
