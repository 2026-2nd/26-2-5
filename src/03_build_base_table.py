"""집계 결과(counts_all.csv)로 모델링용 기본 표를 만든다.

사용법 (저장소 폴더에서):
    python src/03_build_base_table.py --counts C:/bike_data/processed/counts_all.csv --out C:/bike_data/processed/base_table.csv

하는 일
1. 각 대여소의 활동 기간(첫 기록일~마지막 기록일) 동안 모든 날짜 x 시간 블록 칸을 만들고, 기록이 없는 칸은 0으로 채운다.
2. 순유출(net_out = rentals - returns)을 계산한다. 이것이 예측 목표(타깃)이다.
3. 직전 블록 순유출(net_prev_block), 전일 같은 블록 순유출(net_prev_day)을 만든다.
4. 요일, 월, 공휴일 피처를 붙인다.
5. 분석 구간(--start ~ --end)만 남긴다. 첫 이틀은 6월 말에 빌린 건의 반납이 빠져 있어 제외한다.

주의: rentals, returns는 타깃(net_out)을 이루는 값이므로 같은 블록의 rentals/returns를 입력 피처로 쓰면 안 된다.
"""
import argparse
from pathlib import Path

import pandas as pd

BLOCKS_PER_DAY = 5
DEFAULT_HOLIDAYS = Path(__file__).resolve().parent.parent / "data" / "reference" / "holidays.csv"


def build_grid(counts):
    """대여소별 활동 기간 안의 모든 날짜 x 블록 칸을 만들고 건수를 붙인다 (없으면 0)."""
    span = counts.groupby("station_id")["date"].agg(["min", "max"])
    all_dates = pd.date_range(counts["date"].min(), counts["date"].max())
    idx = pd.MultiIndex.from_product(
        [span.index, all_dates, range(BLOCKS_PER_DAY)], names=["station_id", "date", "block"])
    grid = idx.to_frame(index=False)
    grid = grid.merge(span, left_on="station_id", right_index=True)
    grid = grid[(grid["date"] >= grid["min"]) & (grid["date"] <= grid["max"])].drop(columns=["min", "max"])
    grid = grid.merge(counts, on=["station_id", "date", "block"], how="left")
    grid[["rentals", "returns"]] = grid[["rentals", "returns"]].fillna(0).astype("int64")
    return grid


def add_features(grid, holidays):
    grid = grid.sort_values(["station_id", "date", "block"]).reset_index(drop=True)
    grid["net_out"] = grid["rentals"] - grid["returns"]
    g = grid.groupby("station_id")["net_out"]
    grid["net_prev_block"] = g.shift(1)                  # 직전 블록
    grid["net_prev_day"] = g.shift(BLOCKS_PER_DAY)       # 전일 같은 블록
    grid["weekday"] = grid["date"].dt.weekday            # 0=월 ... 6=일
    grid["month"] = grid["date"].dt.month
    grid["is_holiday"] = grid["date"].isin(holidays).astype(int)
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holidays", default=str(DEFAULT_HOLIDAYS))
    ap.add_argument("--start", default="2025-07-03")
    ap.add_argument("--end", default="2026-06-30")
    a = ap.parse_args()

    counts = pd.read_csv(a.counts, parse_dates=["date"])
    holidays = pd.to_datetime(pd.read_csv(a.holidays)["date"])
    print("집계 입력:", counts.shape)

    grid = build_grid(counts)
    print("빈 칸을 채운 뒤:", grid.shape)
    grid = add_features(grid, holidays)

    grid = grid[(grid["date"] >= a.start) & (grid["date"] <= a.end)]
    before = len(grid)
    grid = grid.dropna(subset=["net_prev_block", "net_prev_day"])
    print(f"분석 구간 {a.start}~{a.end}: {before}행, 직전/전일 값이 없는 {before - len(grid)}행 제외 -> {len(grid)}행")
    for c in ["net_prev_block", "net_prev_day"]:
        grid[c] = grid[c].astype("int64")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(a.out, index=False)
    print("저장:", a.out)
    print(grid["net_out"].describe())


if __name__ == "__main__":
    main()
