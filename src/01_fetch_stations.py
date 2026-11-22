"""서울 열린데이터광장 '따릉이 실시간 대여정보' API(OA-15493)에서 대여소 속성(거치대 수, 위경도)을 받아 CSV로 저장한다.

사용법 (저장소 폴더에서):
    python src/01_fetch_stations.py            # 또는 노트북에서  %run ../src/01_fetch_stations.py

- 인증키는 저장소 최상위의 .env 파일에서 읽는다:  SEOUL_API_KEY=발급받은키   (.env는 git에 올라가지 않는다)
- 이 API는 호출 시점의 스냅샷이다. 대여소 속성(거치대 수, 위경도)만 저장하고,
  현재 보유 대수·거치율은 저장하지 않는다 (과거 순유출 예측에 쓸 수 없기 때문).
- 인증키는 화면에 출력하지 않는다.
"""
import argparse
import os
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
FIELDS = {  # API 응답 키 -> 저장할 컬럼 이름
    "stationId": "station_id",
    "stationName": "station_name",
    "rackTotCnt": "docks",
    "stationLatitude": "lat",
    "stationLongitude": "lon",
}


def load_key():
    key = os.environ.get("SEOUL_API_KEY")
    if not key:
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
        key = os.environ.get("SEOUL_API_KEY")
    if not key:
        raise SystemExit("SEOUL_API_KEY를 찾지 못했습니다. 저장소 최상위의 .env 파일에 SEOUL_API_KEY=키 를 적고 저장하세요.")
    return key


def fetch_page(base_url, key, service, start, end):
    r = requests.get(f"{base_url}/{key}/json/{service}/{start}/{end}/", timeout=30)
    r.raise_for_status()
    data = r.json()
    # 응답의 최상위 키는 서비스 이름과 다를 수 있으므로(예: rentBikeStatus), row를 가진 블록을 찾는다
    block = next((v for v in data.values() if isinstance(v, dict) and "row" in v), None)
    if block is None:  # 오류 응답: 키 없이 코드와 메시지만 보여 준다
        raise SystemExit(f"API 오류 응답: {str(data.get('RESULT', list(data))[:200] if isinstance(data.get('RESULT', ''), str) else data.get('RESULT'))}")
    code = block.get("RESULT", {}).get("CODE", "")
    if code and code != "INFO-000":
        raise SystemExit(f"API 오류: {block['RESULT']}")
    return block.get("row", []), int(block.get("list_total_count", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--service", default="bikeList")
    ap.add_argument("--base-url", default="http://openapi.seoul.go.kr:8088")
    ap.add_argument("--out", default=str(ROOT / "data" / "reference" / "stations.csv"))
    a = ap.parse_args()

    key = load_key()
    rows, start, step = [], 1, 1000
    while True:
        page, total = fetch_page(a.base_url, key, a.service, start, start + step - 1)
        rows.extend(page)
        print(f"받은 행: {len(rows)}" + (f" / 전체 {total}" if total else ""), flush=True)
        if len(page) < step:
            break
        start += step

    df = pd.DataFrame(rows)
    missing = [k for k in FIELDS if k not in df.columns]
    if missing:
        raise SystemExit(f"예상한 항목이 없습니다: {missing}. 실제 항목: {list(df.columns)}")
    out = df[list(FIELDS)].rename(columns=FIELDS)
    out["docks"] = pd.to_numeric(out["docks"], errors="coerce")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["fetched"] = date.today().isoformat()
    out = out.drop_duplicates("station_id")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False, encoding="utf-8")
    print("저장:", a.out, out.shape)
    print(out[["docks", "lat", "lon"]].describe().round(3))


if __name__ == "__main__":
    main()
