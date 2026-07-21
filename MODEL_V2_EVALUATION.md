# model-v2 선발투수 challenger 평가

기준일: 2026-07-21  
상태: **승격 불가 / 추가 개선 필요**

## 목적

현재 `model-v1`의 적중률을 높이기 위해 팀 피처에 선발투수 정보를 추가했을 때의 효과를 먼저 분리 측정했다. `model-v1`과 운영 예측은 변경하지 않았다.

## 입력과 누수 경계

- 기존 12개 팀 단위 피처에 선발투수 5개 피처를 추가했다.
  - 홈-원정 K-BB 비율 차이
  - 홈 선발의 실점률 우위
  - 홈 선발의 피홈런율 우위
  - 원정/홈 선발의 전년도 MLB 이력 누락 표시
- 투수 성적 후보는 두 가지로 제한했다.
  - `prior_season`: 직전 정규시즌 전체 성적
  - `prior_plus_current`: 직전 시즌 성적과 목표 경기의 공식 날짜보다 앞선 당해 시즌 gameLog 누적
- `prior_plus_current`에 직전 1일·3일 불펜 투구량 차이를 더한 후보도 비교했다. 팀 전체 투구 수에서 해당 경기 선발의 투구 수를 빼 불펜 workload를 근사했고, 같은 공식 날짜 경기는 제외했다.
- 적은 표본은 200타자 상당의 고정 리그 사전분포로 축소했다.
- 2022~2024 7,278경기에서 두 피처 모드와 L2를 Accuracy 우선, Log Loss와 Brier 차순으로 선택했다.
- 선택된 후보는 `starter_logistic:prior_plus_current:l2=0.01`이다.
- 선발+불펜 후보의 선택 구간 Accuracy는 57.47%로 선발-only 57.65%보다 낮았고 Log Loss/Brier도 소폭 나빠 선택되지 않았다.
- 2025 2,426경기는 개발용 시간 외 비교 구간으로 사용했다.

중요한 제한이 있다. 과거 일정에서 읽은 선발 ID는 사후 응답이며, 경기 60분 전에 관측됐다는 타임스탬프가 없다. 따라서 이 평가는 **회고적 가능성 조사**이지 배포 가능한 검증이 아니다. 코드가 결과와 관계없이 `promotion_allowed=false`를 강제한다.

## 2025 결과

| 모델 | 맞힌 경기 | Accuracy | Log Loss | Brier |
|---|---:|---:|---:|---:|
| Elo | 1,362 | 56.14% | 0.687604 | 0.247117 |
| model-v1 Raw LR | 1,343 | 55.36% | 0.681555 | 0.244329 |
| model-v1 Platt | 1,341 | 55.28% | 0.681911 | 0.244495 |
| 선발투수 challenger | 1,344 | 55.40% | 0.681538 | 0.244363 |

선발투수 challenger는 `model-v1`보다 3경기를 더 맞혔고, Log Loss와 Brier도 소폭 낮았다. 그러나 Elo보다 18경기를 덜 맞혔다. 전년도-only 후보는 2025에서 55.85%였지만, 2025를 보기 전에 2022~2024 선택 규칙이 당해 시즌 누적형을 선택했으므로 사후에 후보를 교체하지 않았다.

## 시즌별 Accuracy

| 시즌 | Elo | model-v1 Raw | model-v1 Platt | 선발투수 challenger |
|---|---:|---:|---:|---:|
| 2022 | 57.60% | 58.05% | 57.84% | 60.15% |
| 2023 | 55.51% | 55.84% | 55.55% | 56.29% |
| 2024 | 55.52% | 55.28% | 54.99% | 56.51% |
| 2025 | 56.14% | 55.36% | 55.28% | 55.40% |

선택 구간 세 시즌에서는 Elo와 `model-v1`을 모두 넘었지만 2025에 개선이 재현되지 않았다. 이는 합산 선택 점수가 좋아도 새 시즌에서 효과가 약해질 수 있음을 보여준다.

## 날짜 블록 paired bootstrap

2025의 같은 날짜 경기를 한 블록으로 묶어 10,000회 재표집했다. 차이는 `challenger - model-v1`이며 낮을수록 challenger가 좋다.

| 지표 | 점 추정 차이 | 95% 구간 | 양측 p |
|---|---:|---:|---:|
| Error rate | -0.001237 | [-0.017902, 0.014944] | 0.9064 |
| Log Loss | -0.000373 | [-0.003363, 0.002718] | 0.8122 |
| Brier | -0.000132 | [-0.001624, 0.001356] | 0.8530 |

세 지표 모두 점 추정은 좋아졌지만 95% 구간이 0을 포함한다. 현재 차이를 안정적인 성능 향상으로 단정할 수 없다.

## 판정

- `model-v1` 대비 Accuracy·Log Loss·Brier 동시 개선: 통과
- Elo 대비 Accuracy 개선: 실패
- 과거 선발 ID의 경기 전 관측 증명: 실패
- 통계적 확실성: 미확립
- 배포 승격: 금지

## 다음 개선 순서

1. 선발의 최근 휴식일과 예상 이닝을 추가한다.
2. 불펜 전체 투구량 대신 핵심 고레버리지 투수별 가용성을 검증한다.
3. 확정 라인업과 상대 선발 손잡이 기준 타격 피처를 추가한다.
4. `model-v1 + starter`, Elo 혼합과 비선형 모델을 같은 시간 분할로 비교한다.
5. 모든 후보는 Accuracy가 Elo를 넘고, Log Loss와 Brier가 `model-v1`보다 나빠지지 않아야 한다.
6. 과거 회고 결과와 별개로 `pregame-pitching-snapshot-v2`의 실제 경기 전 자료에서 champion-challenger 결과를 계속 기록한다.

## 재현

```powershell
$env:PYTHONPATH = "src"
python -m mlb_predictor backtest-pitching-v2 `
  --features data\multi-year\features\pregame_features.parquet `
  --cache-dir data\pitching-backtest-cache `
  --output-dir data\pitching-v2-retrospective `
  --bootstrap-iterations 10000 `
  --seed 20260721
```

상세 경기별 확률은 생성 데이터 `data/pitching-v2-retrospective/predictions.csv`, 전체 지표와 신뢰구간은 `evaluation.json`에 저장된다.
