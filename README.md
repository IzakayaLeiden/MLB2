# MLB2 검증형 공개 베타

MLB 정규시즌 홈 승리 확률을 경기 전 데이터만으로 생성하고, 역사 검증과 미래 섀도 검증 상태를 함께 공개하기 위한 프로젝트입니다. MLB Stats API 일정 응답을 원본 그대로 캐시하고, 완료 경기와 예정·연기·취소 경기를 분리해 정규화하며, **경기 당일 결과를 전혀 사용하지 않는** 피처를 생성합니다. 배당·베팅 추천·ROI·수익성 및 포스트시즌은 범위에서 제외합니다.

공개 저장소: [IzakayaLeiden/MLB2](https://github.com/IzakayaLeiden/MLB2)

외부 검토를 위한 모델 작동 방식, 수식, 검증 결과 및 한계는 [`MODEL_CARD.md`](MODEL_CARD.md)에 정리되어 있습니다.
선발투수 challenger의 회고 평가와 승격 보류 사유는 [`MODEL_V2_EVALUATION.md`](MODEL_V2_EVALUATION.md)에 정리되어 있습니다.
정확한 경기별 예측·입력 snapshot·제외 목록·10,000회 날짜 블록 bootstrap 결과는 [model-v1 감사 Release](https://github.com/IzakayaLeiden/MLB2/releases/tag/model-v1-audit-2026-07-21)에서 받을 수 있습니다.

## 현재 검증 상태 (2026-07-21)

- 2018-03-29~2026-07-20 완료 정규시즌 19,379경기 검증 완료(2020 단축 시즌 포함)
- 2022~2024 워크포워드 선택: `logistic_platt:l2=0.01`
- 봉인된 2025 홀드아웃 2,426경기 단회 평가 통과
  - 선택 모델: Log Loss `0.68191`, Brier `0.24450`
  - Elo: Log Loss `0.68760`, Brier `0.24712`
  - 상수 기준선: Log Loss `0.68985`, Brier `0.24835`
- `model-v1`은 2026-07-20까지의 데이터로 동결됐으며 섀도 기간에는 재학습하지 않음
- 역사 검증을 통과한 `model-v1`의 당일 예측을 운영 안전 검사 통과 시 즉시 제공
- 미래 검증 상태는 아직 `false`
  - 30일 경과 및 300경기 채점, 95% 커버리지 등 실제 미래 성능을 별도로 추적 중
  - 미래 검증 진행 여부는 상태 배지로 표시하며, 당일 예측 자체를 지연시키지 않음

동결 모델과 검증 요약은 `artifacts/model-v1/`에 있습니다. 생성 데이터와 일별 섀도 자산은 Git에 커밋하지 않습니다.

## 현재 범위

- 정규시즌 완료 경기 수집 및 날짜 구간별 원본 JSON 캐시
- 취소·연기·진행 중·무승부·점수 누락·요청 공식 날짜 범위 밖 경기 제외와 사유 기록
- 중단 후 나중에 재개된 경기는 완료 시점 처리가 추가되기 전까지 보수적으로 제외
- 경기 ID, 팀, 선발 예정 투수, 구장, 시작 시각, 결과 정규화
- Elo, 시즌 누적 승률, 최근 경기 승률·득점·실점, 휴식일 피처
- 동일 날짜 경기 전체를 같은 사전 스냅샷으로 계산해 더블헤더 누수 차단
- 필수 필드, 중복, 결과 일관성, 확률 범위, 피처 시점 자동 검사
- CSV와 Parquet 동시 출력, 실행 매니페스트와 SHA-256 체크섬 생성

이 단계의 선발투수 값은 일정 응답에 기록된 `probablePitcher` 메타데이터입니다. 과거 경기를 사후 조회한 값은 당시 예측 시점의 선발 정보였다고 보장할 수 없으므로 학습 피처로 사용하지 않으며, 투수 성적 피처와 라인업 피처도 아직 포함하지 않습니다.

`model-v2` challenger를 위해서는 별도의 `pregame-pitching-snapshot-v2`를 수집합니다. 이 스냅샷은 API 수집 UTC 시각·원본 URL·응답 SHA-256을 보존하고, 경기 60분 전 이후에 수집된 자료나 당일 경기 결과가 섞인 자료를 거부합니다. 선발투수는 직전 365일 정규시즌의 전날까지 성적만 사용하며, 불펜은 직전 3일 완료 경기의 투구 수와 상대 타자 수만 집계합니다. 이 피처는 아직 동결 `model-v1` 확률에 반영되지 않습니다.

## 실행

현재 환경에 의존성이 이미 있다면 설치 없이 다음처럼 실행할 수 있습니다.

```powershell
$env:PYTHONPATH = "src"
python -m mlb_predictor build --start-date 2025-03-27 --end-date 2025-04-05 --output-dir data\sample
```

다년 검증, 당일 예측, 결과 연결, 공개 게이트는 다음 CLI를 사용합니다.

```powershell
python -m mlb_predictor backtest --features data\multi-year\features\pregame_features.parquet --output-dir data\model-v1 --cutoff-date 2026-07-20
python -m mlb_predictor forecast --history-games data\runtime\processed\history_games.parquet --model artifacts\model-v1\model-v1.json --target-date 2026-07-21 --output-dir data\shadow
python -m mlb_predictor grade --feed data\shadow\prediction.json --completed-games data\runtime\processed\games.parquet --output data\shadow\grade.json
python -m mlb_predictor gate --feeds-dir data\shadow --grades-dir data\shadow --model artifacts\model-v1\model-v1.json --as-of-date 2026-08-21 --output data\shadow\status.json
python -m mlb_predictor snapshot-pitching --target-date 2026-07-21 --cache-dir data\pitching-cache --output-dir data\pitching-snapshots
```

`backtest`의 2025 홀드아웃은 상태 파일로 단회 사용을 강제합니다. 이미 사용된 출력 디렉터리에 다시 실행하면 실패합니다.

## Sites 앱

`sites-app/`은 공식 Sites vinext worker starter 기반 React·TypeScript 앱입니다. GitHub Pages의 `prediction-feed-v1`을 서버 경로에서 검증하며, 피드가 26시간 이상 오래됐거나 날짜·스키마·품질 검사가 실패하면 확률을 숨기고 `예측 사용 불가`를 표시합니다.

```powershell
cd sites-app
npm ci
npm run typecheck
npm run lint
npm test
```

호스팅 값 `PAGES_FEED_URL`은 Sites 환경 변수로 설정합니다. `SITE_DEMO_FEED=1`은 로컬 시각 QA 전용이며 배포 환경에서는 사용하지 않습니다.

## 자동화

- `.github/workflows/daily-shadow.yml`: 매일 `08:17 America/New_York`, 당일 봉인·전일 채점·draft release 업로드·익명 다운로드 차단 확인
- 같은 daily workflow가 날짜별 `pitching-v2-YYYY-MM-DD.zip`도 draft release에 한 번만 봉인해 challenger 학습 자료를 축적
- `.github/workflows/pages.yml`: 역사 검증·동결 모델·커버리지·오류·봉인 시각 안전 검사를 통과한 `latest.json`, 날짜별 아카이브, 모델 요약, 상태를 GitHub Pages에 배포
- 미래 30일·300경기 성능 검증은 배포와 동시에 계속 누적하며 `status.json`에 기록
- 운영 안전 검사나 Pages 작업이 실패하면 마지막 정상 공개 피드를 교체하지 않음

패키지로 설치한 뒤에는 다음 명령도 사용할 수 있습니다.

```powershell
python -m pip install -e .
mlb-dataset build --start-date 2025-03-27 --end-date 2025-04-05 --output-dir data\sample
```

캐시된 원본을 무시하고 다시 내려받으려면 `--refresh`를 추가합니다. 기본 수집 단위는 7일이며 `--chunk-days`로 조정할 수 있습니다.

피처 워밍업은 기본적으로 `start-date`가 속한 해의 1월 1일부터 정규시즌 경기를 수집해 계산합니다. 더 긴 Elo 이력이 필요하면 `--history-start-date`를 명시하세요. 이력 구간은 `history_games`로 별도 저장되고, 최종 `games`와 `pregame_features`는 요청한 `start-date`부터만 포함합니다.

## 모델 학습과 평가

학습은 `official_date` 순서의 시간 기반 분할을 사용합니다. 경계가 한 날짜의 경기 중간을 가르지 않도록 같은 공식 날짜의 모든 경기를 하나의 단위로 배치하는 **date-atomic split**이므로, 동일 날짜가 학습·검증·테스트에 나뉘지 않습니다. 다음 명령은 기본 비율 60%/20%/20%로 모델 산출물을 만듭니다.

```powershell
$env:PYTHONPATH = "src"
python -m mlb_predictor train --features data\sample\features\pregame_features.parquet --output-dir data\sample-model --train-fraction 0.6 --validation-fraction 0.2 --l2 1.0 --calibration-bins 10
```

평가 보고서는 상수 확률 기준선, Elo 기준선, 원시 로지스틱 회귀(LR), 검증 구간으로 학습한 Platt 보정 LR을 비교하며 **Log Loss**와 **Brier score**를 기록합니다. 두 지표 모두 낮을수록 좋습니다. 출력은 다음 세 파일입니다.

Platt 보정은 비교 대상으로만 기록하며 자동으로 기본 예측이 되지 않습니다. 독립적인 시간 구간에서 보정 개선이 확인되기 전까지 `model.json`의 기본 출력은 원시 로지스틱 확률입니다.

- `model.json`: Python/pickle 런타임 없이 ChatGPT Sites에서 읽을 수 있는 이식 가능한 모델 계수·전처리 계약
- `evaluation.json`: 분할별 기준선·LR·Platt 지표와 보정 구간
- `manifest.json`: 입력 해시, 날짜 경계, 학습 설정, 산출물 체크섬과 재현성 메타데이터

현재 포함된 짧은 샘플로 만든 매니페스트에는 `sample_only_not_performance_evidence`가 기록됩니다. 이는 배선과 재현성 확인용일 뿐이며, 성능·수익성·운영 준비의 근거가 아닙니다. 실제 판단 전에는 다년 데이터와 별도 미래 홀드아웃을 사용해야 합니다.

## 출력 구조

```text
data/sample/
├── raw/                         # API 원본 JSON
├── processed/
│   ├── history_games.csv
│   ├── history_games.parquet
│   ├── games.csv
│   └── games.parquet
├── features/
│   ├── pregame_features.csv
│   └── pregame_features.parquet
├── reports/
│   ├── skipped_games.csv
│   └── quality.json
└── manifest.json
```

같은 출력 폴더에서 다시 실행하면 기존 정식 산출물은 `previous_runs/<run_id>/`로 이동합니다. 새 실행이 실패하면 정식 `manifest.json`은 `build_status=failed`, `artifacts_valid=false`로 기록되고 부분 데이터 파일은 `failed_runs/<run_id>/`로 격리됩니다.

`pregame_features`에는 최종 점수나 승자 플래그가 들어가지 않습니다. 학습 타깃은 `home_win` 하나이며, `home_history_through_date`와 `away_history_through_date`는 항상 현재 경기의 `official_date`보다 이릅니다.

## 검증

```powershell
python -m pytest
python -m compileall -q src tests
```

기존 산출물을 다시 검사하려면 다음을 사용합니다.

```powershell
$env:PYTHONPATH = "src"
python -m mlb_predictor validate --games data\sample\processed\games.parquet --features data\sample\features\pregame_features.parquet
```

## 시점 정책

첫 버전은 보수적으로 `prior_official_date_only` 정책을 사용합니다. 같은 공식 날짜에 열린 경기 결과는 시작 시각이 더 빠르더라도 그날의 다른 경기 피처에 반영하지 않습니다. 따라서 더블헤더와 같은 날짜의 일정 변경에서 결과 누수가 발생하지 않습니다. 다음 날의 Elo에는 그날 완료된 모든 경기의 변화량이 일괄 반영됩니다.

## 주의

MLB Stats API 응답 구조는 공식적으로 안정성이 보장된 개발자 계약이 아닙니다. 원본 캐시, 명시적 정규화, 실패 보고서를 유지해야 하며 공개·상용 서비스 전에는 데이터 사용 권한과 라이선스를 별도로 확인해야 합니다.
