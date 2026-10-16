# WAAM Solver 제출 보고서

## 제출 구성

본 제출물은 고정 seed 기반의 비학습 휴리스틱 Solver와 공개 모델 6개의 검증된 `trajectory.csv`로 구성됩니다. 학습 모델을 사용하지 않으므로 checkpoint 파일은 필요하지 않습니다. Core 및 Micro 탐색은 고정된 탐색 예산으로 수행하지만, 기준해 생성 과정에는 실행시간 기반 중단 조건이 포함될 수 있으므로 모든 실행에서 완전히 동일한 결과를 보장하지는 않습니다.

공개 모델별 제출 폴더에는 다음 세 파일만 포함합니다.

- `config.yaml`: 제공받은 원본 설정
- `target.stl`: 제공받은 원본 형상
- `trajectory.csv`: Solver가 생성한 최종 경로

최종 제출 Solver의 실행 경로는 `run_solver.py` → `solver/run_solver_release.py` → `waam_solver.competition_release.pipeline_reference.main()`입니다. 저장소에 포함된 다른 버전 모듈은 개발 및 비교용이며 최종 제출 진입점이 아닙니다.

## 알고리즘 의사코드

```text
입력: config.yaml, target.stl
설정: mode=balanced, seed=11

1. 설정과 형상을 읽고 레이어별 deposition task를 생성한다.
2. 로봇 도달 범위와 형상 조건을 만족하는 초기 작업 배정과 순서를 생성한다.
3. 이동, 대기, 적층, 주차 시간을 포함한 전체 스케줄을 계산한다.
4. 충돌과 형상 품질을 확인하면서 병합, 재배정, 순서 변경 후보를 탐색한다.
5. 제한된 탐색 예산 안에서 Makespan이 개선되는 유효 후보를 선택한다.
6. 후보별 trajectory.csv를 생성하고 공식 Validator check/run을 수행한다.
7. PASS 후보 중 최종 해를 선택한다. PASS 후보가 없으면 실패로 종료한다.
```

## 계산시간 측정 기준

Solver가 새 입력을 처리하면 출력 폴더의 `solver_result.json`에 실행시간 지표를 기록합니다.

- `T_to_final_csv`: 실행 시작부터 최종 후보 CSV가 선택되고 기록될 때까지의 경과시간
- `T_algorithm_excluding_validator`: 공식 Validator 실행시간을 제외한 알고리즘 실행시간
- `T_final_validation`: 최종 선택 후보의 공식 검증시간
- `T_total_with_validation`: 공식 검증을 포함한 전체 실행시간

알고리즘 계산시간은 `T_algorithm_excluding_validator`를 기준으로 보고하며, 최종 CSV 준비까지의 벽시계 시간을 사용할 경우에는 `T_to_final_csv`라고 명시합니다. 공개 모델 폴더에는 제출에 필요한 `config.yaml`, `target.stl`, `trajectory.csv`만 포함하므로 계산시간은 각 모델을 생성한 원래 실행 출력의 `solver_result.json`에서 확인합니다.

## 공개 모델 결과

| 모델 | Makespan (초) | Coverage | IoU | Collision events | Validator |
|---|---:|---:|---:|---:|---|
| 01 | 1193.38 | 0.920689 | 0.916413 | 0 | PASS |
| 02 | 3344.49 | 0.998987 | 0.990086 | 0 | PASS |
| 03 | 1872.36 | 0.945046 | 0.940280 | 0 | PASS |
| 04 | 52823.48 | 0.908038 | 0.907054 | 0 | PASS |
| 05 | 125547.45 | 0.971653 | 0.968417 | 0 | PASS |
| 06 | 6105.02 | 0.909708 | 0.907289 | 0 | PASS |

공개 모델 6개 모두 Reach, Collision, Shape 검증을 통과했으며 Collision event는 0건입니다.

## 일반화와 재현성

- 공개 모델별 하드코딩 대신 입력 형상과 설정에서 작업을 생성합니다.
- 비공개 입력도 동일한 `run_solver.py`와 동일한 기본 설정으로 처리합니다.
- 입력 형상의 특성에 따라 유효해를 찾지 못할 가능성은 있으므로 모든 임의 형상에 대한 PASS를 이론적으로 보장하지는 않습니다.
- 공개 모델은 저장소의 `verify_public.py`로 다시 검증할 수 있습니다.

## 평가 항목 대응

- Makespan: 제한된 후보 탐색과 스케줄 개선으로 최소화
- 계산 시간: 학습 없이 고정된 탐색 예산으로 실행
- 일반화: 새로운 `config.yaml`, `target.stl`을 직접 읽어 경로 생성
- 재현성: 고정된 mode와 seed 및 의존성으로 재현성을 높였으나, 기준해 생성의 실행시간 기반 중단 조건 때문에 모든 실행의 완전한 동일성을 보장하지는 않음
