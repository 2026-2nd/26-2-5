"""기본 표(base_table.csv)에 대여소 속성(거치대 수, 위도, 경도)을 붙인다.

사용법 (저장소 폴더에서):
    python src/04_merge_stations.py --base C:/bike_data/processed/base_table.csv --out C:/bike_data/processed/table_stations.csv

- 대여소 속성은 data/reference/stations.csv (01_fetch_stations.py가 만든 파일, 호출 시점의 스냅샷)에서 가져온다.
- 대여이력에는 있는데 stations.csv에는 없는 대여소(폐쇄·이전 등)는 속성을 알 수 없으므로 제외하고,
  제외한 대여소와 행 수를 출력·저장해 보고서에 쓸 수 있게 한다.
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATIONS = ROOT / "data" / "reference" / "stations.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stations", default=str(DEFAULT_STATIONS))
    a = ap.parse_args()

    base = pd.read_csv(a.base, parse_dates=["date"], dtype={"station_id": str})
    st = pd.read_csv(a.stations, dtype={"station_id": str})
    base["station_id"] = base["station_id"].str.strip()
    st["station_id"] = st["station_id"].str.strip()
    st = st[["station_id", "docks", "lat", "lon"]].drop_duplicates("station_id")

    ids_base = set(base["station_id"].unique())
    ids_st = set(st["station_id"])
    matched = ids_base & ids_st
    print(f"기본 표의 대여소 {len(ids_base)}개 중 stations.csv와 매칭 {len(matched)}개 ({len(matched) / len(ids_base):.1%})")
    print(f"stations.csv에만 있는 대여소: {len(ids_st - ids_base)}개")

    merged = base.merge(st, on="station_id", how="left")
    miss = merged["docks"].isna() | merged["lat"].isna() | merged["lon"].isna()
    print(f"속성이 없어 제외할 행: {int(miss.sum())} / {len(merged)} ({miss.mean():.2%})")
    if miss.any():
        un = (merged[miss].groupby("station_id").agg(rows=("net_out", "size"), rentals=("rentals", "sum"))
              .sort_values("rows", ascending=False))
        print(f"제외 대상 대여소 {len(un)}개, 대여 합계가 큰 상위 10개:")
        print(un.sort_values("rentals", ascending=False).head(10))
        out_dir = Path(a.out).parent
        un.to_csv(out_dir / "unmatched_stations.csv")
    merged = merged[~miss]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(a.out, index=False)
    print("저장:", a.out, merged.shape)
    print("컬럼:", list(merged.columns))


if __name__ == "__main__":
    main()
