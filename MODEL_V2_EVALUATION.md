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
- 투수 성적은 각 경기 시즌의 **직전 정규시즌 전체 성적만** 사용했다. 현재 시즌이나 경기 결과는 사용하지 않았다.
- 적은 표본은 200타자 상당의 고정 리그 사전분포로 축소했다.
- 2022~2024 7,278경기에서 Accuracy 우선, Log Loss와 Brier 차순으로 L2를 선택했다.
- 선택된 후보는 `starter_logistic:l2=0.01`이다.
- 2025 2,426경기는 개발용 시간 외 비교 구간으로 사용했다.

중요한 제한이 있다. 과거 일정에서 읽은 선발 ID는 사후 응답이며, 경기 60분 전에 관측됐다는 타임스탬프가 없다. 따라서 이 평가는 **회고적 가능성 조사**이지 배포 가능한 검증이 아니다. 코드가 결과와 관계없이 `promotion_allowed=false`를 강제한다.

## 2025 결과

| 모델 | 맞힌 경기 | Accuracy | Log Loss | Brier |
|---|---:|---:|---:|---:|
| Elo | 1,362 | 56.14% | 0.687604 | 0.247117 |
| model-v1 Raw LR | 1,343 | 55.36% | 0.681555 | 0.244329 |
| model-v1 Platt | 1,341 | 55.28% | 0.681911 | 0.244495 |
| 선발투수 challenger | 1,355 | 55.85% | 0.681219 | 0.244193 |

선발투수 challenger는 `model-v1`보다 14경기를 더 맞혔고, Log Loss와 Brier도 소폭 낮았다. 그러나 Elo보다 7경기를 덜 맞혔다.

## 시즌별 Accuracy

| 시즌 | Elo | model-v1 Raw | model-v1 Platt | 선발투수 challenger |
|---|---:|---:|---:|---:|
| 2022 | 57.60% | 58.05% | 57.84% | 59.49% |
| 2023 | 55.51% | 55.84% | 55.55% | 55.43% |
| 2024 | 55.52% | 55.28% | 54.99% | 55.98% |
| 2025 | 56.14% | 55.36% | 55.28% | 55.85% |

개선 방향은 완전히 일관되지 않았다. challenger는 2023년에 `model-v1`보다도 소폭 낮았고, 2023·2025에는 Elo보다 낮았다.

## 날짜 블록 paired bootstrap

2025의 같은 날짜 경기를 한 블록으로 묶어 10,000회 재표집했다. 차이는 `challenger - model-v1`이며 낮을수록 challenger가 좋다.

| 지표 | 점 추정 차이 | 95% 구간 | 양측 p |
|---|---:|---:|---:|
| Error rate | -0.005771 | [-0.021356, 0.009106] | 0.4736 |
| Log Loss | -0.000692 | [-0.003391, 0.002095] | 0.6386 |
| Brier | -0.000302 | [-0.001627, 0.001054] | 0.6522 |

세 지표 모두 점 추정은 좋아졌지만 95% 구간이 0을 포함한다. 현재 차이를 안정적인 성능 향상으로 단정할 수 없다.

## 판정

- `model-v1` 대비 Accuracy·Log Loss·Brier 동시 개선: 통과
- Elo 대비 Accuracy 개선: 실패
- 과거 선발 ID의 경기 전 관측 증명: 실패
- 통계적 확실성: 미확립
- 배포 승격: 금지

## 다음 개선 순서

1. 당해 시즌의 경기 전 누적 선발 성적과 최근 휴식일을 추가한다.
2. 이미 봉인 중인 직전 1·3일 불펜 투구량을 challenger에 추가한다.
3. `model-v1 + starter`, `model-v1 + starter + bullpen`, Elo 혼합을 같은 시간 분할로 비교한다.
4. 모든 후보는 Accuracy가 Elo를 넘고, Log Loss와 Brier가 `model-v1`보다 나빠지지 않아야 한다.
5. 과거 회고 결과와 별개로 `pregame-pitching-snapshot-v2`의 실제 경기 전 자료에서 champion-challenger 결과를 계속 기록한다.

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
