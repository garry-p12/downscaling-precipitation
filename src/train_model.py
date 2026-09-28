"""Phase 3: gradient-boosted (XGBoost) / random-forest training for both stages."""
from __future__ import annotations

import pickle
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .feature_engineering import (
    coarse_feature_names,
    coarse_features_10km,
    fine_feature_names,
    fine_features_1km,
    sample_fine_training_data,
)
from .utils import LOG, GridPair, open_dataset, save_json

DEFAULT_XGB = dict(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8,
                   colsample_bytree=0.8, min_child_weight=5)
DEFAULT_RF = dict(n_estimators=200, max_depth=12, min_samples_split=20)


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def make_model(model_type: str = "xgb", hyperparams: dict | None = None, n_jobs: int = -1,
               early_stopping_rounds: int | None = None):
    if model_type == "xgb":
        import xgboost as xgb

        params = {**DEFAULT_XGB, **(hyperparams or {})}
        return xgb.XGBRegressor(
            tree_method="hist", n_jobs=n_jobs, objective=params.pop("objective", "reg:squarederror"),
            early_stopping_rounds=early_stopping_rounds, eval_metric="rmse", random_state=42, **params,
        )
    if model_type == "lgbm":
        import lightgbm as lgb

        params = {**DEFAULT_XGB, **(hyperparams or {})}
        params.pop("min_child_weight", None)
        return lgb.LGBMRegressor(n_jobs=n_jobs, random_state=42, verbose=-1, **params)
    if model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor

        params = {**DEFAULT_RF, **(hyperparams or {})}
        return RandomForestRegressor(n_jobs=n_jobs, random_state=42, **params)
    raise ValueError(f"Unknown model_type {model_type}")


def train_model(X_train: np.ndarray, y_train: np.ndarray, model_type: str = "xgb", hyperparams: dict | None = None,
                X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
                early_stopping_rounds: int | None = None, n_jobs: int = -1):
    """Fit a regressor; XGBoost uses ``(X_val, y_val)`` for early stopping."""
    t0 = _time.time()
    use_es = model_type in ("xgb", "lgbm") and X_val is not None and early_stopping_rounds
    model = make_model(model_type, hyperparams, n_jobs, early_stopping_rounds if use_es else None)
    if model_type == "xgb":
        if use_es:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train, verbose=False)
    elif model_type == "lgbm":
        if use_es:
            import lightgbm as lgb

            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
        else:
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    LOG.info("Trained %s on %d rows in %.1fs", model_type, len(y_train), _time.time() - t0)
    return model


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def get_feature_importance(model, feature_names: list[str], X: np.ndarray | None = None,
                           y: np.ndarray | None = None, permutation: bool = False,
                           max_rows: int = 100_000) -> pd.DataFrame:
    """Gain-based (or impurity) importance, optionally with permutation importance."""
    if hasattr(model, "get_booster"):
        score = model.get_booster().get_score(importance_type="gain")
        imp = np.array([score.get(f"f{i}", 0.0) for i in range(len(feature_names))])
    elif hasattr(model, "booster_"):
        imp = model.booster_.feature_importance(importance_type="gain").astype(float)
    else:
        imp = np.asarray(model.feature_importances_, dtype=float)
    imp = imp / imp.sum() if imp.sum() > 0 else imp
    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    if permutation and X is not None and y is not None:
        from sklearn.inspection import permutation_importance

        rng = np.random.default_rng(0)
        idx = rng.choice(len(y), size=min(max_rows, len(y)), replace=False)
        r = permutation_importance(model, X[idx], y[idx], n_repeats=3, random_state=0, n_jobs=1,
                                   scoring="neg_root_mean_squared_error")
        df["permutation_importance"] = r.importances_mean
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def save_model(model, path: str | Path, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "metadata": metadata or {}}, f, protocol=pickle.HIGHEST_PROTOCOL)
    LOG.info("Saved model -> %s (%.1f MB)", path, path.stat().st_size / 1e6)


def load_model(path: str | Path):
    with open(path, "rb") as f:
        blob = pickle.load(f)
    return blob["model"], blob.get("metadata", {})


def _split_tail(idx_time: np.ndarray, frac: float) -> np.ndarray:
    """Boolean mask selecting the last ``frac`` of *days* (temporal hold-out)."""
    nt = idx_time.max() + 1
    cut = int(np.floor(nt * (1 - frac)))
    return idx_time >= cut


# --------------------------------------------------------------------------- #
# Stage 1: 10 km model
# --------------------------------------------------------------------------- #
def train_coarse_stage(cfg: dict, grids: GridPair) -> dict:
    proc = Path(cfg["paths"]["processed"])
    mcfg = cfg["model"]["coarse"]
    X_train = np.load(proc / "X_train.npy")
    y_train = np.load(proc / "y_train.npy")
    X_val = np.load(proc / "X_val.npy")
    y_val = np.load(proc / "y_val.npy")
    idx_train = np.load(proc / "idx_train.npz")
    names = coarse_feature_names(cfg)
    model_type = mcfg.get("type", "xgb")
    params = mcfg.get("params" if model_type != "rf" else "rf_params", {})

    # Early stopping on the tail of the training period keeps 2019-2020 untouched.
    es_source = mcfg.get("early_stopping_split", "train_tail")
    es_rounds = mcfg.get("early_stopping_rounds", 25)
    if model_type == "rf" or not es_rounds:
        X_fit, y_fit, X_es, y_es = X_train, y_train, None, None
    elif es_source == "val":
        X_fit, y_fit, X_es, y_es = X_train, y_train, X_val, y_val
    else:
        tail = _split_tail(idx_train["time_index"], mcfg.get("early_stopping_frac", 0.15))
        X_fit, y_fit, X_es, y_es = X_train[~tail], y_train[~tail], X_train[tail], y_train[tail]

    model = train_model(X_fit, y_fit, model_type, params, X_es, y_es, es_rounds)
    if es_source == "train_tail" and X_es is not None and model_type == "xgb":
        # Refit on the full training period with the tuned number of trees.
        best = int(getattr(model, "best_iteration", params.get("n_estimators", 200)) or params.get("n_estimators", 200)) + 1
        LOG.info("Refitting on full training period with %d trees", best)
        model = train_model(X_train, y_train, model_type, {**params, "n_estimators": best}, None, None, None)

    pred_val = np.clip(model.predict(X_val), 0, None)
    pred_tr = np.clip(model.predict(X_train), 0, None)
    imerg_col = names.index("imerg")
    metrics = {
        "train_rmse": rmse(pred_tr, y_train),
        "val_rmse": rmse(pred_val, y_val),
        "val_rmse_imerg_raw": rmse(X_val[:, imerg_col], y_val),
        "val_bias": float(np.mean(pred_val - y_val)),
        "val_corr": float(np.corrcoef(pred_val, y_val)[0, 1]),
        "val_corr_imerg_raw": float(np.corrcoef(X_val[:, imerg_col], y_val)[0, 1]),
    }
    LOG.info("Stage 1 val RMSE %.3f (raw IMERG %.3f), corr %.3f (raw %.3f)", metrics["val_rmse"],
             metrics["val_rmse_imerg_raw"], metrics["val_corr"], metrics["val_corr_imerg_raw"])

    imp = get_feature_importance(model, names, X_val, y_val, permutation=mcfg.get("permutation_importance", False))
    meta = {
        "stage": "coarse_10km",
        "model_type": model_type,
        "hyperparams": params,
        "best_iteration": int(getattr(model, "best_iteration", -1) or -1),
        "n_train_rows": int(len(y_train)),
        "n_val_rows": int(len(y_val)),
        "feature_names": names,
        "feature_importance": imp.to_dict(orient="records"),
        "top10_features": imp["feature"].head(10).tolist(),
        "metrics": metrics,
        "train_period": [cfg["time"]["train_start"], cfg["time"]["train_end"]],
        "val_period": [cfg["time"]["val_start"], cfg["time"]["val_end"]],
        "domain": cfg["domain"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    mdir = Path(cfg["paths"]["models"])
    save_model(model, mdir / "downscaler_coarse.pkl", meta)
    save_json(meta, mdir / "model_metadata.json")
    imp.to_csv(mdir / "feature_importance_coarse.csv", index=False)
    return meta


# --------------------------------------------------------------------------- #
# Stage 2: 1 km residual model
# --------------------------------------------------------------------------- #
def compute_coarse_predictions(cfg: dict, grids: GridPair, model) -> xr.Dataset:
    """Stage-1 predictions plus the coarse fields needed by the fine builder, all days."""
    from .predict import predict_coarse

    proc = Path(cfg["paths"]["processed"])
    imerg = open_dataset(proc / "imerg_aligned_10km.nc")["precip"].load()
    nlcd = open_dataset(proc / "nlcd_aligned_1km.nc").load()
    clim = open_dataset(proc / "imerg_climatology_10km.nc")["imerg_sorted"].load()
    feats = coarse_features_10km(imerg, nlcd, clim, grids, cfg)
    pred = predict_coarse(model, feats, coarse_feature_names(cfg))
    win = int(cfg["features"].get("rolling_days", 7))
    out = xr.Dataset({"pred": pred, "imerg": feats["imerg"], f"imerg_roll{win}": feats[f"imerg_roll{win}"],
                      "imerg_pct": feats["imerg_pct"]})
    return out, nlcd


def train_fine_stage(cfg: dict, grids: GridPair, coarse_model=None) -> dict:
    proc = Path(cfg["paths"]["processed"])
    mdir = Path(cfg["paths"]["models"])
    fcfg = cfg["model"]["fine"]
    tcfg = cfg["time"]
    if coarse_model is None:
        coarse_model, _ = load_model(mdir / "downscaler_coarse.pkl")
    coarse, nlcd = compute_coarse_predictions(cfg, grids, coarse_model)
    fine_static = fine_features_1km(nlcd, grids)
    aorc = open_dataset(proc / "aorc_aligned_1km.nc", chunks={"time": 64})["precip"]

    seed = int(fcfg.get("seed", 42))
    X_tr, y_tr, base_tr = sample_fine_training_data(
        cfg, grids, coarse, fine_static, aorc, slice(tcfg["train_start"], tcfg["train_end"]),
        int(fcfg.get("max_samples", 3_000_000)), seed)
    X_va, y_va, base_va = sample_fine_training_data(
        cfg, grids, coarse, fine_static, aorc, slice(tcfg["val_start"], tcfg["val_end"]),
        int(fcfg.get("val_samples", 500_000)), seed + 1)
    names = fine_feature_names(cfg)
    model_type = fcfg.get("type", "xgb")
    params = fcfg.get("params" if model_type != "rf" else "rf_params", {})

    # Temporal early-stopping hold-out: last 15% of sampled training days.
    n = len(y_tr)
    cut = int(n * 0.85)  # samples are generated in time order
    model = train_model(X_tr[:cut], y_tr[:cut], model_type, params, X_tr[cut:], y_tr[cut:],
                        fcfg.get("early_stopping_rounds", 25))
    # Method selection on the training-tail hold-out (never on 2019-2020): does
    # the residual model beat plain bilinear upsampling of the stage-1 field?
    res_ho = model.predict(X_tr[cut:])
    obs_ho = base_tr[cut:] + y_tr[cut:]
    rmse_res_ho = rmse(np.clip(base_tr[cut:] + res_ho, 0, None), obs_ho)
    rmse_bl_ho = rmse(np.clip(base_tr[cut:], 0, None), obs_ho)
    min_gain = float(fcfg.get("min_gain_over_bilinear", 0.005))
    recommended = "residual" if rmse_res_ho < rmse_bl_ho * (1 - min_gain) else "bilinear"
    LOG.info("Stage 2 hold-out RMSE %.3f vs bilinear %.3f -> recommended upsampling: %s",
             rmse_res_ho, rmse_bl_ho, recommended)
    if model_type == "xgb":
        best = int(getattr(model, "best_iteration", params.get("n_estimators", 200)) or params.get("n_estimators", 200)) + 1
        model = train_model(X_tr, y_tr, model_type, {**params, "n_estimators": best}, None, None, None)

    res_va = model.predict(X_va)
    pred_va = np.clip(base_va + res_va, 0, None)
    obs_va = base_va + y_va
    metrics = {
        "holdout_rmse_residual_model": rmse_res_ho,
        "holdout_rmse_bilinear_stage1": rmse_bl_ho,
        "val_rmse_residual_model": rmse(pred_va, obs_va),
        "val_rmse_bilinear_stage1": rmse(np.clip(base_va, 0, None), obs_va),
        "val_corr_residual_model": float(np.corrcoef(pred_va, obs_va)[0, 1]),
        "val_corr_bilinear_stage1": float(np.corrcoef(base_va, obs_va)[0, 1]),
    }
    LOG.info("Stage 2 val RMSE %.3f (bilinear stage-1 %.3f)", metrics["val_rmse_residual_model"],
             metrics["val_rmse_bilinear_stage1"])
    imp = get_feature_importance(model, names)
    meta = {
        "stage": "fine_1km_residual",
        "recommended_method": recommended,
        "model_type": model_type,
        "hyperparams": params,
        "best_iteration": int(getattr(model, "best_iteration", -1) or -1),
        "n_train_rows": int(len(y_tr)),
        "n_val_rows": int(len(y_va)),
        "feature_names": names,
        "feature_importance": imp.to_dict(orient="records"),
        "top10_features": imp["feature"].head(10).tolist(),
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    save_model(model, mdir / "downscaler_fine.pkl", meta)
    save_json(meta, mdir / "model_metadata_fine.json")
    imp.to_csv(mdir / "feature_importance_fine.csv", index=False)
    return meta
