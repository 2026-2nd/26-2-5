"""기본 표(table_stations.csv)에 기상 피처(기온, 강수량, 풍속)를 붙인다.

사용법 (저장소 폴더에서):
    python src/05_merge_weather.py --table C:/bike_data/processed/table_stations.csv \
        --weather C:/bike_data/기상파일이름.csv --out C:/bike_data/processed/table_final.csv

- 기상 자료: 기상청 기상자료개방포털 종관기상관측(ASOS) 시간 자료, 서울(108). 열: 일시, 기온(°C), 강수량(mm), 풍속(m/s).
- 시간별 값을 시간 블록(0~6, 6~10, 10~16, 16~20, 20~24시)으로 묶는다: 기온·풍속은 평균, 강수량은 합계.
- 강수량 칸이 비어 있는 시간은 강수가 없었던 것으로 보고 0으로 처리한다 (기상청 시간 자료에서 무강수는 공백으로 표시됨).
"""
import argparse
from pathlib import Path

import pandas as pd

BINS = [0, 6, 10, 16, 20, 24]


def read_any(path):
    try:
        return pd.read_csv(path, encoding="cp949")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def find_col(df, key):
    cols = [c for c in df.columns if key in c]
    if len(cols) != 1:
        raise SystemExit(f"'{key}'가 들어간 열을 정확히 하나 찾지 못했습니다. 실제 열: {list(df.columns)}")
    return cols[0]


def block_weather(path):
    w = read_any(path)
    t, temp, rain, wind = (find_col(w, k) for k in ["일시", "기온", "강수", "풍속"])
    w["time"] = pd.to_datetime(w[t])
    print(f"기상 자료: {len(w)}행, {w['time'].min()} ~ {w['time'].max()}")
    print(f"강수량 빈 칸: {w[rain].isna().sum()}행 -> 0으로 처리 (기온 빈 칸 {w[temp].isna().sum()}, 풍속 빈 칸 {w[wind].isna().sum()})")
    w["rain_mm"] = w[rain].fillna(0)
    w["date"] = w["time"].dt.normalize()
    w["block"] = pd.cut(w["time"].dt.hour, bins=BINS, labels=False, right=False).astype(int)
    g = (w.groupby(["date", "block"])
          .agg(temp_c=(temp, "mean"), rain_mm=("rain_mm", "sum"), wind_ms=(wind, "mean"), n_hours=(temp, "size"))
          .reset_index())
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--weather", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    bw = block_weather(a.weather)
    tab = pd.read_csv(a.table, parse_dates=["date"], dtype={"station_id": str})
    n0 = len(tab)
    merged = tab.merge(bw.drop(columns="n_hours"), on=["date", "block"], how="left")
    miss = merged[["temp_c", "rain_mm", "wind_ms"]].isna().any(axis=1)
    print(f"기상이 붙지 않은 행: {int(miss.sum())} / {n0}")
    if miss.any():
        print("붙지 않은 날짜 예:", sorted(merged.loc[miss, "date"].dt.date.unique())[:5])
    merged = merged[~miss]
    short = bw[bw["n_hours"] < 3]
    if len(short):
        print(f"주의: 시간 수가 3개 미만인 블록 {len(short)}개 (기상 자료에 빠진 시간이 있음)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(a.out, index=False)
    print("저장:", a.out, merged.shape)
    print(merged[["temp_c", "rain_mm", "wind_ms"]].describe().round(2))


if __name__ == "__main__":
    main()
