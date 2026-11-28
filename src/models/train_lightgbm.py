"""LightGBM으로 순유출(net_out)을 예측한다.

사용법 (저장소 폴더에서):
    python src/models/train_lightgbm.py                    # 기본 설정 1회 학습, val로 평가
    python src/models/train_lightgbm.py --drop month       # 피처 빼고 학습
    python src/models/train_lightgbm.py --station          # 대여소 ID를 범주형 피처로 추가해 val 비교
    python src/models/train_lightgbm.py --tune             # 하이퍼파라미터 조합 비교 (val)
    python src/models/train_lightgbm.py --final            # 최종 설정으로 학습 후 test 평가 (설정을 다 정한 뒤 1회만)

- train으로 학습하고 val로 평가한다. test는 설정을 다 정한 뒤 최종 평가 때만 쓴다.
- 조기 종료(early stopping): val 점수가 50번 연속 좋아지지 않으면 학습을 멈춘다.
- 결과
  - 1회 학습: results/metrics/LightGBM_<이름>_val.csv
  - --tune:  results/metrics/LightGBM_tuning_val.csv (조합별 점수 표)
  - --final: results/metrics/LightGBM_final_val.csv, LightGBM_final_test.csv
  - --station --final: results/metrics/LightGBM_station_final_val.csv, LightGBM_station_final_test.csv (최종 모델)
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # src/ (evaluate.py 위치)
from evaluate import DEFAULT_DATA, DEFAULT_OUT, TARGET, evaluate  # noqa: E402

FEATURES = ["block", "weekday", "month", "is_holiday", "temp_c", "rain_mm", "wind_ms",
            "lat", "lon", "docks", "net_prev_block", "net_prev_day"]
CATEGORICAL = ["block", "weekday"]   # 순서보다 "어느 칸이냐"가 중요한 피처
DEFAULT_DROP = ["month"]             # train에 4~6월이 없어 val/test에서 처음 보는 값이 됨 (1단계 비교 결과 점수 차이 없음)
STATION = "station_code"             # --station일 때만 쓰는 대여소 번호 (범주형)

PARAMS = {
    "objective": "regression",   # 제곱오차(L2) 최소화 = RMSE 기준
    "learning_rate": 0.1,
    "num_leaves": 255,           # --tune 결과 val RMSE 1위 (4.518). 표 전체: results/metrics/LightGBM_tuning_val.csv
    "min_child_samples": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    "verbose": -1,
}

# --tune에서 비교할 조합: 잎 개수(모델 복잡도) x 잎 하나의 최소 샘플 수(과적합 방지)
GRID = {
    "num_leaves": [63, 127, 255],
    "min_child_samples": [100, 500],
}


def load(name, data_dir):
    return pd.read_csv(Path(data_dir) / f"{name}.csv", parse_dates=["date"], dtype={"station_id": str})


def add_station_code(train, *others):
    """station_id를 train 기준 번호로 바꾼 컬럼을 추가한다. train에 없던 대여소는 -1(LightGBM이 결측으로 처리)."""
    ids = pd.Index(sorted(train["station_id"].unique()))
    for df in (train, *others):
        df[STATION] = ids.get_indexer(df["station_id"]).astype("int32")


def fit(params, train, val, feats, rounds=3000, log_every=0):
    cats = [c for c in CATEGORICAL + [STATION] if c in feats]
    dtrain = lgb.Dataset(train[feats], train[TARGET], categorical_feature=cats, free_raw_data=False)
    dval = lgb.Dataset(val[feats], val[TARGET], categorical_feature=cats, reference=dtrain)
    callbacks = [lgb.early_stopping(50, verbose=False)]
    if log_every:
        callbacks.append(lgb.log_evaluation(log_every))
    t = time.time()
    model = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dval], valid_names=["val"],
                      callbacks=callbacks)
    return model, time.time() - t


def run_once(a, feats, train, val):
    model, sec = fit(PARAMS, train, val, feats, log_every=200)
    print(f"학습 {sec:.0f}초, 최적 반복 {model.best_iteration}회")
    pred = model.predict(val[feats], num_iteration=model.best_iteration)
    name = a.name or ("station" if a.station else "all" if not a.drop else "no_" + "_".join(a.drop))
    evaluate(val, pred, model_name=f"LightGBM_{name}_val")

    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("\n피처 중요도 (gain 비율)")
    print((imp / imp.sum()).round(3).to_string())


def run_final(feats, train, val, data_dir):
    """최종 설정으로 학습하고 val과 test를 평가한다. test는 이 함수에서만 쓴다."""
    model, sec = fit(PARAMS, train, val, feats, log_every=500)
    print(f"학습 {sec:.0f}초, 최적 반복 {model.best_iteration}회 (상한 3000)")
    prefix = "LightGBM_station_final" if STATION in feats else "LightGBM_final"
    evaluate(val, model.predict(val[feats], num_iteration=model.best_iteration), model_name=f"{prefix}_val")
    test = load("test", data_dir)
    if STATION in feats:
        add_station_code(train, test)
        print(f"test에서 train에 없던 대여소 {int((test[STATION] == -1).sum())}행")
    evaluate(test, model.predict(test[feats], num_iteration=model.best_iteration), model_name=f"{prefix}_test")

    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("\n피처 중요도 (gain 비율)")
    print((imp / imp.sum()).round(3).to_string())


def run_tune(feats, train, val):
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    out = DEFAULT_OUT / "LightGBM_tuning_val.csv"
    DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, c in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}] {c} 학습 시작", flush=True)
        params = {**PARAMS, **c}
        model, sec = fit(params, train, val, feats, log_every=500)
        pred = model.predict(val[feats], num_iteration=model.best_iteration)
        res = evaluate(val, pred, model_name=f"tune{i}", save=False)
        rows.append({**c, "learning_rate": params["learning_rate"], "best_iter": model.best_iteration,
                     "sec": round(sec), **{k: v for k, v in res.items() if k not in ("model", "n_rows")}})
        print(f"[{i}/{len(combos)}] {c} -> RMSE {res['RMSE']:.4f}, 반복 {model.best_iteration}, {sec:.0f}초", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)   # 조합마다 저장 (중간에 멈춰도 결과가 남음)

    table = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    table.to_csv(out, index=False)
    print("\n조합별 val 점수 (RMSE 순)")
    print(table.round(4).to_string(index=False))
    print("저장:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--drop", nargs="*", default=DEFAULT_DROP, help="뺄 피처 (기본: month). 다 쓰려면 --drop 만 쓴다")
    ap.add_argument("--name", default=None, help="결과 파일 이름 (기본: 뺀 피처로 자동)")
    ap.add_argument("--tune", action="store_true", help="GRID의 조합을 모두 학습해 val 점수 비교")
    ap.add_argument("--final", action="store_true", help="최종 설정으로 학습 후 test까지 평가 (최종 보고용 1회)")
    ap.add_argument("--station", action="store_true", help="대여소 ID를 범주형 피처로 추가")
    a = ap.parse_args()
    if a.station and a.tune:
        ap.error("--station은 --tune과 함께 쓸 수 없다 (기존 튜닝표 덮어쓰기 방지)")

    feats = [f for f in FEATURES if f not in a.drop]
    if a.station:
        feats.append(STATION)
    print(f"피처 {len(feats)}개: {feats}")
    train, val = load("train", a.data_dir), load("val", a.data_dir)
    if a.station:
        add_station_code(train, val)
        print(f"val에서 train에 없던 대여소 {int((val[STATION] == -1).sum())}행")
    if a.tune:
        run_tune(feats, train, val)
    elif a.final:
        run_final(feats, train, val, a.data_dir)
    else:
        run_once(a, feats, train, val)


if __name__ == "__main__":
    main()
