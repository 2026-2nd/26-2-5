# CONTRIBUTING.MD(협업 가이드)

팀원 모두가 같은 방식으로 작업하기 위한 규칙입니다. 작업 전에 한 번 읽어 주세요!

## 1. 브랜치

- `main`에는 직접 push하지 않습니다. 모든 변경은 브랜치를 만들어 PR로 합칩니다.
- 브랜치는 항상 최신 `main`에서 만듭니다.
- 이름 규칙: `<종류>/<작업-내용>` (소문자, 단어는 `-`로 연결)

| 종류 | 용도 | 예시 |
|---|---|---|
| `feature/` | 기능, 모델, 실험 추가 | `feature/model-lightgbm`, `feature/evaluation` |
| `fix/` | 버그 수정 | `fix/precision-at-k-ties` |
| `docs/` | 문서만 수정 | `docs/contributing` |

```bash
git switch main
git pull
git switch -c feature/model-ridge
```

- 브랜치를 오래 두면 `main`과 멀어져 충돌이 커집니다. 작업이 며칠 이상 걸리면 중간에 `main`을 자기 브랜치에 합치고, 가능하면 작은 단위로 PR을 올립니다.
- 머지한 브랜치는 GitHub에서 삭제하고(PR 화면의 Delete branch), 내 컴퓨터의 브랜치도 `git branch -d <브랜치>`로 지웁니다.

## 2. 커밋 메시지

형식: `<접두사>: <변경 내용 요약>`

- 요약은 영어 한 줄, 소문자 시작, 마침표 없이 씁니다.
- 한 커밋에는 한 가지 일만 담습니다.

| 접두사 | 용도 |
|---|---|
| `feat` | 새 기능, 새 스크립트, 새 모델 |
| `fix` | 버그 수정 |
| `exp` | 실험 결과 추가 (`results/` 지표 파일 등) |
| `test` | 테스트 추가·수정 |
| `refactor` | 동작은 그대로인 코드 정리 |
| `docs` | README 등 문서 수정 |
| `chore` | 설정, 의존성(`requirements.txt`) 등 잡일 |

```
feat: add common evaluation code and tests
fix: make Precision@k tie handling independent of row order
exp: add last-week baseline scores (val, test)
```

## 3. Pull Request

1. 브랜치를 push하고 GitHub에서 `main`을 대상으로 PR을 만듭니다.
2. PR 제목은 커밋 메시지와 같은 형식으로 씁니다. 예: `feat: add Ridge regression training script`
3. PR 본문에는 무엇을, 왜 바꿨는지 적습니다. 실험 PR이면 val/test 점수도 적습니다.
4. 팀원 1명 이상이 확인(approve)한 뒤 머지합니다. 본인 PR은 본인이 머지하지 않습니다.
5. 머지 방식은 **Merge commit**(기본값)을 씁니다. Squash, Rebase는 쓰지 않습니다.
6. 충돌이 나면 PR 작성자가 `main`을 자기 브랜치에 먼저 합쳐서(`git merge main`) 해결합니다.

## 4. 실험 기록

모든 모델은 같은 평가 코드(`src/evaluate.py`)와 같은 분할(`train.csv`, `val.csv`, `test.csv`)을 씁니다.

### 평가 규칙

- 하이퍼파라미터 선택과 모델 비교는 검증(val) 점수로 합니다.
- **테스트 점수는 최종 모델 하나에 대해서만, 마지막에 한 번 계산합니다.** 테스트 점수를 보고 모델을 고치지 않습니다.
- 분할은 날짜 기준 시간순입니다. 랜덤 분할이나 셔플은 쓰지 않습니다.
- 랜덤 시드는 코드에 고정하고, 튜닝 범위와 선택한 값을 결과 파일이나 PR 본문에 남깁니다.

### 결과 파일

- 점수는 `src/evaluate.py`의 `evaluate()`로 `results/metrics/`에 CSV로 저장합니다. 직접 CSV를 만들지 않습니다.
- 파일 이름: `<모델>_<설명>_<val|test>.csv` (예: `LightGBM_tuning_val.csv`, `baseline_last_week_test.csv`)
- 컬럼은 아래로 통일합니다. 튜닝 기록은 이 컬럼 앞에 하이퍼파라미터 컬럼(예: `num_leaves`, `learning_rate`)을 붙입니다.

| 컬럼 | 의미 |
|---|---|
| `model` | 파일 이름과 같은 모델 이름 |
| `n_rows` | 평가한 행 수 |
| `RMSE`, `MAE`, `R2` | 회귀 지표 |
| `P@10_shortage`, `P@20_shortage`, `P@50_shortage` | 부족 대여소 상위 k개 Precision@k |
| `P@10_surplus`, `P@20_surplus`, `P@50_surplus` | 과잉 대여소 상위 k개 Precision@k |

### 분할 파일 공유

`data/processed/`는 git에 올라가지 않으므로 분할 파일은 아래 방법으로 받습니다.

- 받는 곳: 카카오톡 or 드라이브 링크
- 직접 만들 때는 README 4장 순서로 `src/01`~`06`을 실행합니다.
- 분할 파일이나 전처리를 바꿔야 하면 혼자 바꾸지 말고 PR로 올려, 팀 전체가 같은 파일을 다시 받습니다.
- 같은 분할인지는 `n_rows`로 확인합니다. val은 733,800행, test는 747,120행이어야 합니다. 다르면 점수를 비교하지 말고 먼저 팀에 알립니다.

## 5. 폴더와 파일

| 폴더 | 넣는 것 |
|---|---|
| `src/` | 전처리 스크립트(`01`~`06`), 공통 코드(`evaluate.py`) |
| `src/models/` | 모델 학습 스크립트 `train_<모델>.py` |
| `tests/` | `test_<대상>.py` |
| `results/metrics/` | 점수 CSV (모델 파일, 큰 예측 파일은 넣지 않음) |
| `results/figures/` | 그림 (필요할 때 만듭니다) |
| `notebooks/` | EDA, 시각화. `NN_내용.ipynb` 형식 (필요할 때 만듭니다) |
| `data/reference/` | 작은 보조 파일(공휴일 목록 등) |
| `data/raw/`, `data/processed/` | 원본과 집계 데이터 (git에 올리지 않음) |

- 데이터 파일과 인증키가 든 `.env`는 커밋하지 않습니다. 새 환경 변수는 `.env.example`에 형식만 추가합니다.
- 학습된 모델 파일(`*.pkl`, `*.joblib` 등)과 100MB 넘는 파일은 커밋하지 않습니다. GitHub가 거부하고, 한 번 커밋하면 기록에서 지우기 번거롭습니다.
- 패키지를 추가하면 `requirements.txt`도 같은 PR에서 수정합니다.
- `src/evaluate.py` 같은 공통 코드를 바꿀 때는 `pytest tests`로 통과를 확인하고, PR 본문에 공통 코드를 바꿨다고 분명히 적습니다.
- 마감 후에는 `main`을 수정하지 않습니다.
