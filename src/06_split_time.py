"""최종 표(table_final.csv)를 날짜 순서대로 학습·검증·테스트로 나눠 저장한다.

사용법 (저장소 폴더에서):
    python src/06_split_time.py --table C:/bike_data/processed/table_final.csv --out-dir C:/bike_data/processed

- 랜덤 분할이 아니라 날짜 순서로 나눈다 (미래 정보가 학습에 새어 들어가는 것을 막기 위해).
- 학습 70%, 검증 15%, 테스트 15% (날짜 기준). 같은 날짜는 한 곳에만 들어간다.
- 타깃은 net_out. rentals, returns는 타깃을 이루는 값이므로 입력 피처로 쓰지 않는다.
"""
import argparse
from pathlib import Path

import pandas as pd

TARGET = "net_out"
FEATURES = ["block", "weekday", "month", "is_holiday", "temp_c", "rain_mm", "wind_ms",
            "lat", "lon", "docks", "net_prev_block", "net_prev_day"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    a = ap.parse_args()

    df = pd.read_csv(a.table, parse_dates=["date"], dtype={"station_id": str})
    dates = sorted(df["date"].unique())
    n = len(dates)
    i1, i2 = int(n * a.train), int(n * (a.train + a.val))
    d_train_end, d_val_end = dates[i1 - 1], dates[i2 - 1]
    parts = {
        "train": df[df["date"] <= d_train_end],
        "val": df[(df["date"] > d_train_end) & (df["date"] <= d_val_end)],
        "test": df[df["date"] > d_val_end],
    }
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"전체 {len(df)}행, {n}일, 피처 {len(FEATURES)}개: {FEATURES}")
    for name, p in parts.items():
        p.to_csv(out / f"{name}.csv", index=False)
        print(f"{name:5s} {p['date'].min().date()} ~ {p['date'].max().date()} | {p['date'].nunique()}일 {len(p)}행 | "
              f"{TARGET} 평균 {p[TARGET].mean():.3f} 표준편차 {p[TARGET].std():.3f}")
    assert sum(len(p) for p in parts.values()) == len(df)
    assert parts["train"]["date"].max() < parts["val"]["date"].min() <= parts["val"]["date"].max() < parts["test"]["date"].min()
    print("저장:", out)


if __name__ == "__main__":
    main()
