"""MLP(다층 퍼셉트론, PyTorch)로 순유출(net_out)을 예측한다. (가산점용)

사용법 (저장소 폴더에서):
    python src/models/train_mlp.py                 # 기본: LightGBM과 같은 피처 11개
    python src/models/train_mlp.py --embed 8       # 2단계: 대여소 임베딩(8차원) 추가

- train으로 학습하고 val로 평가한다. test는 설정을 다 정한 뒤 최종 평가 때만 쓴다.
- 조기 종료: 에폭마다 val RMSE를 보고, 3번 연속 좋아지지 않으면 멈춘 뒤 가장 좋았던 가중치로 되돌린다.
- GPU가 있으면 자동으로 쓴다 (Colab에서도 코드 수정 없이 실행).
- 결과: results/metrics/MLP_<이름>_val.csv

전처리 (신경망은 입력 크기·형태에 민감하다)
- block, weekday: 원-핫 인코딩 (숫자 크기로 읽지 않도록 칸을 펼침)
- rain_mm: 대부분 0이고 가끔 큰 값 → log1p 후 표준화
- 나머지 숫자 피처: 표준화 (평균 0, 표준편차 1)
- is_holiday: 0/1 그대로
- 표준화 기준(평균·표준편차)은 train에서만 계산한다 (val/test 정보가 새지 않도록).
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # src/ (evaluate.py 위치)
from evaluate import DEFAULT_DATA, TARGET, evaluate  # noqa: E402

ONEHOT = {"block": range(5), "weekday": range(7)}
LOG_THEN_SCALE = ["rain_mm"]
SCALE = ["temp_c", "wind_ms", "lat", "lon", "docks", "net_prev_block", "net_prev_day"]
AS_IS = ["is_holiday"]
# month는 LightGBM과 같은 이유로 뺀다 (train에 4~6월이 없음)


class Preprocessor:
    """train으로 fit하고, train/val/test에 똑같이 transform한다."""

    def fit(self, df):
        num = self._numeric(df)
        self.mean, self.std = num.mean(), num.std().replace(0, 1)
        return self

    def _numeric(self, df):
        num = df[SCALE + LOG_THEN_SCALE].astype("float64").copy()
        num[LOG_THEN_SCALE] = np.log1p(num[LOG_THEN_SCALE].clip(lower=0))
        return num

    def transform(self, df):
        num = (self._numeric(df) - self.mean) / self.std
        onehot = [pd.get_dummies(pd.Categorical(df[c], categories=list(cats)), prefix=c, dtype="float32")
                  for c, cats in ONEHOT.items()]
        x = pd.concat([num.reset_index(drop=True), df[AS_IS].reset_index(drop=True)] + onehot, axis=1)
        self.columns = list(x.columns)
        return torch.tensor(x.to_numpy(dtype="float32"))


class StationIndex:
    """station_id → 정수 번호. train에 없던 대여소는 0번(모름)으로 보낸다."""

    def fit(self, ids):
        self.map = {s: i + 1 for i, s in enumerate(sorted(ids.unique()))}
        return self

    def transform(self, ids):
        return torch.tensor(ids.map(self.map).fillna(0).astype("int64").to_numpy())

    def __len__(self):
        return len(self.map) + 1


class MLP(nn.Module):
    def __init__(self, n_in, hidden=(128, 64), dropout=0.1, n_stations=0, embed_dim=0):
        super().__init__()
        self.embed = nn.Embedding(n_stations, embed_dim) if embed_dim else None
        layers, d = [], n_in + embed_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, station=None):
        if self.embed is not None:
            x = torch.cat([x, self.embed(station)], dim=1)
        return self.net(x).squeeze(1)


def predict(model, x, s, device, batch=65536):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(model(x[i:i + batch].to(device), s[i:i + batch].to(device)).cpu())
    return torch.cat(out).numpy()


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - p) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--embed", type=int, default=0, help="대여소 임베딩 차원 (0이면 사용 안 함)")
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=30, help="최대 에폭 (조기 종료로 보통 더 일찍 멈춤)")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = a.name or (f"embed{a.embed}" if a.embed else "base")
    print(f"장치: {device}, 설정: {vars(a)}")

    read = lambda n: pd.read_csv(Path(a.data_dir) / f"{n}.csv", parse_dates=["date"], dtype={"station_id": str})
    train, val = read("train"), read("val")

    prep = Preprocessor().fit(train)
    x_tr, x_va = prep.transform(train), prep.transform(val)
    st = StationIndex().fit(train["station_id"])
    s_tr, s_va = st.transform(train["station_id"]), st.transform(val["station_id"])
    y_tr = torch.tensor(train[TARGET].to_numpy(dtype="float32"))
    print(f"입력 {x_tr.shape[1]}칸: {prep.columns}")
    if a.embed:
        print(f"대여소 임베딩: {len(st) - 1}개 대여소 x {a.embed}차원, val에서 train에 없던 대여소 {int((s_va == 0).sum())}행")

    model = MLP(x_tr.shape[1], a.hidden, a.dropout, len(st), a.embed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.MSELoss()

    best, best_state, bad = float("inf"), None, 0
    for epoch in range(1, a.epochs + 1):
        t = time.time()
        model.train()
        perm = torch.randperm(len(x_tr))
        total = 0.0
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            xb, sb, yb = x_tr[idx].to(device), s_tr[idx].to(device), y_tr[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb, sb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        val_rmse = rmse(val[TARGET], predict(model, x_va, s_va, device))
        mark = ""
        if val_rmse < best - 1e-4:
            best, best_state, bad, mark = val_rmse, copy.deepcopy(model.state_dict()), 0, " *"
        else:
            bad += 1
        print(f"에폭 {epoch:2d} | train RMSE {np.sqrt(total / len(perm)):.4f} | val RMSE {val_rmse:.4f}{mark} | {time.time() - t:.0f}초",
              flush=True)
        if bad >= a.patience:
            print(f"val이 {a.patience}번 연속 좋아지지 않아 멈춤")
            break

    model.load_state_dict(best_state)
    evaluate(val, predict(model, x_va, s_va, device), model_name=f"MLP_{name}_val")


if __name__ == "__main__":
    main()
