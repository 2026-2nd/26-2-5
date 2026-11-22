"""월별 따릉이 대여이력 CSV를 읽어 (대여소, 날짜, 시간 블록)별 대여·반납 건수로 집계한다.

사용법 (저장소 폴더에서):
    python src/02_aggregate_months.py --data-dir C:\\bike_data --out-dir C:\\bike_data\\processed

- 생년·성별·이용자종류 등은 읽지 않는다 (필요한 6개 컬럼만 사용).
- 제외: 같은 대여소 반납 + 이동거리 0m, 반납 시각이 대여 시각보다 이른 건, 대여 정보 결측.
- 반납 대여소ID만 비어 있는 건은 대여로만 센다(반납은 대여소를 알 수 없어 세지 않음).
- 월별 결과는 out-dir/monthly/ 에, 합친 결과는 out-dir/counts_all.csv 에 저장한다.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

COLS = ["대여일시", "반납일시", "이용시간(분)", "이용거리(M)", "대여대여소ID", "반납대여소ID"]
RENAME = {"대여일시": "rent_time", "반납일시": "ret_time", "이용시간(분)": "minutes",
          "이용거리(M)": "meters", "대여대여소ID": "rent_id", "반납대여소ID": "ret_id"}
# 시간 블록: 0=0~6시, 1=6~10시, 2=10~16시, 3=16~20시, 4=20~24시
BINS = [0, 6, 10, 16, 20, 24]


def load_month(path):
    kw = dict(usecols=COLS, dtype={"대여대여소ID": str, "반납대여소ID": str})
    try:
        df = pd.read_csv(path, encoding="cp949", **kw)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="utf-8", **kw)
    df = df.rename(columns=RENAME)
    df["rent_time"] = pd.to_datetime(df["rent_time"], errors="coerce")
    df["ret_time"] = pd.to_datetime(df["ret_time"], errors="coerce")
    return df


def clean(df):
    log = {"rows_raw": int(len(df))}
    # 반납 대여소ID가 비어 있는 건(2026년 파일에서 나타남)은 버리지 않는다.
    # 자전거가 대여소를 떠난 것은 사실이므로 대여로는 세고, 반납은 대여소를 알 수 없어 세지 않는다.
    df = df.dropna(subset=["rent_time", "ret_time", "rent_id"])
    log["after_dropna"] = int(len(df))
    log["ret_id_missing"] = int(df["ret_id"].isna().sum())
    same_zero = (df["rent_id"] == df["ret_id"]) & (df["meters"] == 0)
    log["drop_same_station_0m"] = int(same_zero.sum())
    df = df[~same_zero]
    reversed_time = df["ret_time"] < df["rent_time"]
    log["drop_return_before_rent"] = int(reversed_time.sum())
    df = df[~reversed_time]
    log["rows_clean"] = int(len(df))
    return df, log


def block_of(ts):
    return pd.cut(ts.dt.hour, bins=BINS, labels=False, right=False).astype(int)


def count_events(ids, times, name):
    t = pd.DataFrame({"station_id": ids.values,
                      "date": times.dt.normalize().values,
                      "block": block_of(times).values})
    return t.groupby(["station_id", "date", "block"]).size().rename(name)


def aggregate(df):
    rentals = count_events(df["rent_id"], df["rent_time"], "rentals")
    with_ret = df.dropna(subset=["ret_id"])
    returns = count_events(with_ret["ret_id"], with_ret["ret_time"], "returns")
    return pd.concat([rentals, returns], axis=1).fillna(0).astype("int64").reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    out = Path(a.out_dir).resolve()
    # 출력 폴더가 데이터 폴더 안에 있어도 이전 결과 파일을 입력으로 읽지 않도록 제외한다
    files = [p for p in sorted(Path(a.data_dir).rglob("*.csv"))
             if out not in p.resolve().parents]
    (out / "monthly").mkdir(parents=True, exist_ok=True)
    print(f"CSV {len(files)}개 발견")

    logs, parts = {}, []
    for p in files:
        print("처리 중:", p.name, flush=True)
        df, log = clean(load_month(p))
        agg = aggregate(df)
        agg.to_csv(out / "monthly" / f"{p.stem}.csv", index=False)
        logs[p.name] = log
        parts.append(agg)
        print("  ", log, flush=True)

    allc = (pd.concat(parts)
            .groupby(["station_id", "date", "block"], as_index=False)[["rentals", "returns"]].sum())
    allc.to_csv(out / "counts_all.csv", index=False)
    (out / "cleaning_log.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    print("완료:", out / "counts_all.csv", allc.shape)


if __name__ == "__main__":
    main()
