"""Random Forest for station-block net outflow.

Select the setting on validation RMSE (test is evaluated once, with --final).
Search space: min_samples_leaf x max_features. Depth is unlimited and each tree sees 20% of the
training rows (bootstrap sample); a depth-limited setting (depth 14/20, 200k rows per tree) was
tried first and gave a higher validation RMSE (5.321), so the search space was widened.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import DEFAULT_DATA, DEFAULT_OUT, evaluate

FEATURES = ["block", "weekday", "is_holiday", "temp_c", "rain_mm", "wind_ms",
            "lat", "lon", "docks", "net_prev_block", "net_prev_day"]
N_TREES = 60
MAX_SAMPLES = 0.2                                                    # share of training rows each tree sees
GRID = [(leaf, feat) for leaf in (5, 20, 50) for feat in (0.5, 1.0)]   # (min_samples_leaf, max_features)


def load(directory, split):
    return pd.read_csv(directory / f"{split}.csv",
                       usecols=FEATURES + ["station_id", "date", "net_out"],
                       dtype={c: "float32" for c in FEATURES})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--final", action="store_true", help="Evaluate the selected forest on test")
    args = parser.parse_args()
    train = load(args.data_dir, "train")
    val = load(args.data_dir, "val")
    X, X_val = train[FEATURES].to_numpy(), val[FEATURES].to_numpy()
    best_rmse = float("inf")
    tuning = []
    for leaf, feat in GRID:
        forest = RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=leaf, max_features=feat,
                                       max_samples=MAX_SAMPLES, random_state=42, n_jobs=args.jobs)
        forest.fit(X, train.net_out)
        pred = forest.predict(X_val)
        scores = evaluate(val, pred, f"RandomForest_leaf{leaf}_feat{feat}_val", save=False)
        tuning.append(dict(min_samples_leaf=leaf, max_features=feat, **scores))
        if scores["RMSE"] < best_rmse:
            best_rmse, best, val_pred = scores["RMSE"], forest, pred
        else:
            del forest
    evaluate(val, val_pred, "RandomForest_final_val")
    pd.DataFrame(tuning).to_csv(DEFAULT_OUT / "RandomForest_tuning_val.csv", index=False)
    folder = DEFAULT_OUT.parent / "eda"
    folder.mkdir(exist_ok=True)
    pd.Series(best.feature_importances_, index=FEATURES, name="importance").rename_axis(
        "feature").to_csv(folder / "rf_importance.csv")
    val.assign(error=abs(val.net_out - val_pred)).groupby("block").error.mean().rename("MAE").to_csv(
        folder / "rf_val_block_mae.csv")
    if args.final:
        test = load(args.data_dir, "test")
        evaluate(test, best.predict(test[FEATURES].to_numpy()), "RandomForest_final_test")


if __name__ == "__main__":
    main()
