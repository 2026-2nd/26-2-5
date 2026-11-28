"""evaluate.py 검증용 테스트. 손으로 계산한 답과 비교한다.

실행 (저장소 폴더에서):
    python tests/test_evaluate.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluate import baseline_last_week, precision_at_k, regression_metrics  # noqa: E402


def close(a, b):
    return abs(a - b) < 1e-9


def test_regression_metrics():
    # 오차 = [0, 0, 2]
    # MAE = 2/3, RMSE = sqrt(4/3), R² = 1 - 잔차제곱합 4 / 총제곱합 2 = -1
    m = regression_metrics([1, 2, 3], [1, 2, 5])
    assert close(m["MAE"], 2 / 3), m
    assert close(m["RMSE"], np.sqrt(4 / 3)), m
    assert close(m["R2"], -1.0), m
    # 완벽한 예측이면 오차 0, R² 1
    m = regression_metrics([1, 2, 3], [1, 2, 3])
    assert m["RMSE"] == 0 and m["MAE"] == 0 and close(m["R2"], 1.0), m


def test_baseline_last_week():
    # 대여소 A: 7/1~7/8, net_out = 그날의 일(day) 숫자
    # 대여소 B: 7/5부터만 있음 → 7/8의 7일 전(7/1) 기록이 없음
    days = pd.date_range("2025-07-01", "2025-07-08")
    a = pd.DataFrame({"station_id": "A", "date": days, "block": 0, "net_out": days.day})
    b = pd.DataFrame({"station_id": "B", "date": days[4:], "block": 0, "net_out": 100})
    history = pd.concat([a, b], ignore_index=True)
    target = history[history["date"] == "2025-07-08"].reset_index(drop=True)
    pred = baseline_last_week(target, history)
    # A의 7/8 예측 = 7/1 값 = 1, B는 기록 없음 → 0
    assert list(pred) == [1, 0], pred


def test_baseline_matches_block():
    # 같은 날 다른 블록 값이 섞이지 않는지 확인
    days = pd.date_range("2025-07-01", "2025-07-08")
    rows = [{"station_id": "A", "date": d, "block": blk, "net_out": blk * 10}
            for d in days for blk in range(5)]
    history = pd.DataFrame(rows)
    target = history[history["date"] == "2025-07-08"].reset_index(drop=True)
    assert list(baseline_last_week(target, history)) == [0, 10, 20, 30, 40]


def one_block(true, pred, date="2025-07-01", block=0):
    df = pd.DataFrame({"station_id": [f"s{i}" for i in range(len(true))],
                       "date": pd.Timestamp(date), "block": block, "net_out": true})
    return df, np.array(pred, dtype=float)


def test_precision_perfect_and_reversed():
    df, pred = one_block([5, 4, 3, 2, 1], [5, 4, 3, 2, 1])
    p = precision_at_k(df, pred, ks=(2,))
    assert p == {"P@2_shortage": 1.0, "P@2_surplus": 1.0}, p
    # 순서를 완전히 뒤집으면 상위 2개와 하위 2개가 하나도 안 맞음
    df, pred = one_block([5, 4, 3, 2, 1], [1, 2, 3, 4, 5])
    p = precision_at_k(df, pred, ks=(2,))
    assert p == {"P@2_shortage": 0.0, "P@2_surplus": 0.0}, p


def test_precision_partial():
    # 실제 상위2 {s0,s1}, 예측 상위2 {s0,s1} → 2/2 = 1.0
    # 실제 하위2 {s4,s3}, 예측 하위2 {s2,s3} → 1/2 = 0.5
    df, pred = one_block([5, 4, 3, 2, 1], [5, 4, 1, 2, 3])
    p = precision_at_k(df, pred, ks=(2,))
    assert p == {"P@2_shortage": 1.0, "P@2_surplus": 0.5}, p


def test_precision_averages_over_blocks():
    # 블록 0은 완벽(1.0), 블록 1은 뒤집힘(0.0) → 평균 0.5
    df0, p0 = one_block([5, 4, 3, 2, 1], [5, 4, 3, 2, 1], block=0)
    df1, p1 = one_block([5, 4, 3, 2, 1], [1, 2, 3, 4, 5], block=1)
    df = pd.concat([df0, df1], ignore_index=True)
    p = precision_at_k(df, np.concatenate([p0, p1]), ks=(2,))
    assert p["P@2_shortage"] == 0.5 and p["P@2_surplus"] == 0.5, p


def test_precision_ignores_index():
    # test 표의 인덱스가 0부터가 아니어도(필터링 후 등) 결과가 같아야 함
    df, pred = one_block([5, 4, 3, 2, 1], [5, 4, 1, 2, 3])
    df.index = [10, 30, 20, 50, 40]
    p = precision_at_k(df, pred, ks=(2,))
    assert p == {"P@2_shortage": 1.0, "P@2_surplus": 0.5}, p


def test_precision_ties():
    # 실제 [5, 3, 3, 3, 1], k=2: 1위는 s0 확정, 남은 1자리를 s1~s3(값 3) 셋이 나눔 → 각 1/3
    # 예측 상위2 = {s0, s1} → 기대 일치 = (1 + 1/3) / 2 = 2/3
    df, pred = one_block([5, 3, 3, 3, 1], [5, 4, 2, 1, 0])
    p = precision_at_k(df, pred, ks=(2,))
    assert close(p["P@2_shortage"], 2 / 3), p
    # 예측이 전부 같은 값이면 무작위로 찍는 것과 같음 → 기대 일치 = k/n = 2/5
    df, pred = one_block([5, 4, 3, 2, 1], [0, 0, 0, 0, 0])
    p = precision_at_k(df, pred, ks=(2,))
    assert close(p["P@2_shortage"], 2 / 5) and close(p["P@2_surplus"], 2 / 5), p


def test_precision_row_order_independent():
    # 행 순서를 섞어도 결과가 같아야 함 (동점이 많은 경우)
    rng = np.random.default_rng(0)
    true = rng.integers(-3, 4, 200)
    pred = rng.integers(-3, 4, 200)
    df, pred = one_block(true, pred)
    a = precision_at_k(df, pred, ks=(10,))
    perm = rng.permutation(200)
    b = precision_at_k(df.iloc[perm], pred[perm], ks=(10,))
    assert all(close(a[k], b[k]) for k in a), (a, b)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"통과  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"실패  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} 통과")
    sys.exit(1 if failed else 0)
