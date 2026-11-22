"""모든 모델이 같이 쓰는 평가 코드.

사용법 (모델 코드에서):
    from evaluate import evaluate
    evaluate(test_df, y_pred, model_name="LightGBM")

- test_df: 06_split_time.py가 만든 test.csv (또는 val.csv)를 읽은 표. station_id, date, block, net_out 컬럼이 필요하다.
- y_pred: test_df와 같은 행 순서의 예측값 (길이가 같아야 한다).
- 결과는 화면에 출력하고 results/metrics/<model_name>.csv 로 저장한다.

지표
1. RMSE, MAE, R²: 전체 행 기준 회귀 지표
2. Precision@k: 날짜x시간 블록마다 "예측 상위 k개 대여소 중 실제 상위 k개에 든 비율"을 구해 평균한다.
   - shortage(부족): 순유출이 큰 쪽 = 곧 빌 대여소
   - surplus(과잉): 순유출이 작은(음수) 쪽 = 곧 넘칠 대여소
   - k = 10, 20, 50

기준선: baseline_last_week()로 "지난주 같은 요일·같은 블록의 실제 순유출"을 예측값으로 만든다.
    python src/evaluate.py --data-dir C:/bike_data/processed --split test
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

KEYS = ["station_id", "date", "block"]
TARGET = "net_out"
KS = (10, 20, 50)
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "results" / "metrics"


# ① 회귀 지표 ---------------------------------------------------------------
def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


# ② 기준선 ------------------------------------------------------------------
def baseline_last_week(target_df, history_df):
    """target_df 각 행에 대해 7일 전 같은 대여소·같은 블록의 실제 net_out을 예측값으로 돌려준다.

    history_df에는 target_df보다 앞선 기간이 포함되어야 한다 (예: train+val+test를 합친 표).
    과거 값만 보므로 미래 정보는 쓰지 않는다.
    7일 전 기록이 없는 행(대여소가 새로 생긴 경우 등)은 0으로 채우고 개수를 출력한다.
    """
    past = history_df[KEYS + [TARGET]].copy()
    past["date"] = pd.to_datetime(past["date"]) + pd.Timedelta(days=7)   # 7일 뒤 행에 붙도록 날짜를 민다
    past = past.rename(columns={TARGET: "pred"})
    t = target_df[KEYS].copy()
    t["date"] = pd.to_datetime(t["date"])
    t["station_id"] = t["station_id"].astype(str)
    past["station_id"] = past["station_id"].astype(str)
    pred = t.merge(past, on=KEYS, how="left")["pred"]
    n_missing = int(pred.isna().sum())
    if n_missing:
        print(f"[기준선] 7일 전 기록이 없는 {n_missing}행({n_missing / len(pred):.2%})은 0으로 채움")
    return pred.fillna(0).to_numpy()


# ③ Precision@k --------------------------------------------------------------
def precision_at_k(df, y_pred, ks=KS):
    """날짜x블록마다 예측 상위 k개와 실제 상위 k개가 겹치는 비율을 구해 평균한다.

    순위가 같은 값(동점)은 표의 행 순서대로 앞선 것을 높게 본다 (method="first").
    모든 모델이 같은 test 표를 쓰므로 동점 처리 기준도 같다.
    """
    d = df[["date", "block", TARGET]].copy().reset_index(drop=True)
    d["pred"] = np.asarray(y_pred, dtype=float)
    g = d.groupby(["date", "block"])
    out = {}
    for side, ascending in [("shortage", False), ("surplus", True)]:
        r_true = g[TARGET].rank(method="first", ascending=ascending)
        r_pred = g["pred"].rank(method="first", ascending=ascending)
        for k in ks:
            hit = (r_true <= k) & (r_pred <= k)
            per_block = hit.groupby([d["date"], d["block"]]).sum() / k
            out[f"P@{k}_{side}"] = float(per_block.mean())
    return out


# 한 번에 평가·저장 ---------------------------------------------------------
def evaluate(df, y_pred, model_name, ks=KS, out_dir=DEFAULT_OUT, save=True):
    if len(df) != len(y_pred):
        raise ValueError(f"행 수가 다름: df {len(df)}, y_pred {len(y_pred)}")
    res = {"model": model_name, "n_rows": len(df)}
    res.update(regression_metrics(df[TARGET], y_pred))
    res.update(precision_at_k(df, y_pred, ks))
    print(f"\n[{model_name}]")
    for k, v in res.items():
        if k != "model":
            print(f"  {k:16s} {v:.4f}" if isinstance(v, float) else f"  {k:16s} {v}")
    if save:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([res]).to_csv(out_dir / f"{model_name}.csv", index=False)
        print("  저장:", out_dir / f"{model_name}.csv")
    return res


def main():
    """기준선 점수를 계산한다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="train.csv, val.csv, test.csv가 있는 폴더")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    read = lambda n: pd.read_csv(Path(a.data_dir) / f"{n}.csv", parse_dates=["date"], dtype={"station_id": str})
    parts = {n: read(n) for n in ["train", "val", "test"]}
    history = pd.concat(parts.values(), ignore_index=True)
    target = parts[a.split]
    pred = baseline_last_week(target, history)
    evaluate(target, pred, model_name=f"baseline_last_week_{a.split}")


if __name__ == "__main__":
    main()
