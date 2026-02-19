"""
Forecasting module for XAI4HEAT

This module contains functions and models for heat load forecasting
in district heating systems.
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sympy import denom
from xgboost import XGBRegressor, XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll.base import scope




def train_xgb_qizm_walkforward(
    df: pd.DataFrame,
    *,
    target_col: str = "qizm",
    season_col: str = "season",
    test_seasons: list = None,
    n_lags: int = 24,
    add_temp_lags: list | None = None,
    temp_lag_cols: list | None = None,
    base_features: list = None,
    drop_cols: list | None = None,
    xgb_params: dict | None = None,
    require_complete_cases: bool = True,
    verbose: int = 1,
    # -----------------------------
    # OPTIONAL two-stage (on/off + regression)
    # -----------------------------
    two_stage: bool = False,
    on_col: str = "heating_on",
    clf_threshold: float = 0.5,
    xgb_clf_params: dict | None = None,
    # -----------------------------
    # OPTIONAL Bayesian optimization (Hyperopt) ONLY on the LAST fold
    # -----------------------------
    optimize_last_fold: bool = False,
    optimize_metric: str = "MAE",            # "MAE" | "RMSE" | "MASE" | "sMAPE"
    val_frac_last_fold: float = 0.2,
    max_evals: int = 40,
    random_state_opt: int = 42,
):
    """
    As before, plus timing:
      - fit_time_*: wall-clock training time for final fitted model(s) in the fold
      - pred_time_*: wall-clock prediction time on the fold's test set
      - pred_time_per_row_ms: (pred_time_total / n_test) * 1000

    Note:
      - Hyperopt tuning time (if enabled on the last fold) is reported separately as opt_time_*.
    """

    # ----------------------------
    # Helpers
    # ----------------------------
    def _smape_safe(y_true, y_pred) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        out = np.zeros_like(denom, dtype=float)
        m = denom > 0
        out[m] = np.abs(y_true[m] - y_pred[m]) / denom[m]
        return float(np.mean(out) * 100.0)

    def _metric_value(y_true, y_pred, y_train_for_mase=None) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        smape = _smape_safe(y_true, y_pred)

        mase = np.nan
        if y_train_for_mase is not None:
            ytr = pd.Series(y_train_for_mase).astype(float)
            naive = ytr.shift(1)
            m = naive.notna()
            mae_naive = float(np.mean(np.abs(ytr[m] - naive[m]))) if m.any() else np.nan
            mase = mae / mae_naive if (np.isfinite(mae_naive) and mae_naive > 0) else np.nan

        mname = optimize_metric.upper()
        if mname == "MAE":
            return mae
        if mname == "RMSE":
            return rmse
        if mname == "SMAPE":
            return smape
        if mname == "MASE":
            return mase if np.isfinite(mase) else 1e18
        raise ValueError("optimize_metric must be one of: MAE, RMSE, MASE, sMAPE")

    def _time_order_split(X, y, frac):
        n = len(X)
        cut = int(np.floor(n * (1.0 - frac)))
        cut = max(1, min(cut, n - 1))
        return X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:]

    def _cast_params(params: dict) -> dict:
        outp = dict(params)
        for k in ["n_estimators", "max_depth", "min_child_weight"]:
            if k in outp:
                outp[k] = int(outp[k])
        for k, v in list(outp.items()):
            if isinstance(v, (np.integer,)):
                outp[k] = int(v)
            if isinstance(v, (np.floating,)):
                outp[k] = float(v)
        return outp

    def _optimize_regressor_hyperopt(Xtr_full, ytr_full, base_params: dict):
        X_tr, y_tr, X_va, y_va = _time_order_split(Xtr_full, ytr_full, val_frac_last_fold)

        space = {
            "n_estimators": scope.int(hp.quniform("reg_n_estimators", 400, 2200, 50)),
            "learning_rate": hp.loguniform("reg_learning_rate", np.log(0.01), np.log(0.15)),
            "max_depth": scope.int(hp.quniform("reg_max_depth", 3, 10, 1)),
            "subsample": hp.uniform("reg_subsample", 0.6, 1.0),
            "colsample_bytree": hp.uniform("reg_colsample_bytree", 0.6, 1.0),
            "min_child_weight": scope.int(hp.quniform("reg_min_child_weight", 1, 10, 1)),
            "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-3), np.log(20.0)),
            "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-4), np.log(5.0)),
        }

        def objective(hp_params):
            hp_params = _cast_params(hp_params)
            params = dict(base_params)
            params.update(hp_params)
            params.update({
                "objective": "reg:squarederror",
                "n_jobs": -1,
                "random_state": random_state_opt,
            })
            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr)

            pred = model.predict(X_va)
            loss = _metric_value(
                y_va.to_numpy(),
                pred,
                y_train_for_mase=y_tr if optimize_metric.upper() == "MASE" else None,
            )
            return {"loss": float(loss), "status": STATUS_OK}

        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=int(max_evals),
            trials=trials,
            rstate=np.random.default_rng(random_state_opt),
            show_progressbar=True,   # progress
            verbose=True,
        )

        best_typed = _cast_params({
            "n_estimators": int(best["reg_n_estimators"]),
            "learning_rate": float(best["reg_learning_rate"]),
            "max_depth": int(best["reg_max_depth"]),
            "subsample": float(best["reg_subsample"]),
            "colsample_bytree": float(best["reg_colsample_bytree"]),
            "min_child_weight": int(best["reg_min_child_weight"]),
            "reg_lambda": float(best["reg_lambda"]),
            "reg_alpha": float(best["reg_alpha"]),
        })

        final_params = dict(base_params)
        final_params.update(best_typed)
        final_params.update({
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "random_state": random_state_opt,
        })

        # refit on full fold training
        model = XGBRegressor(**final_params)
        model.fit(Xtr_full, ytr_full)

        best_loss = float(min(t["result"]["loss"] for t in trials.trials if "result" in t and "loss" in t["result"]))
        return model, final_params, best_loss

    def _optimize_classifier_hyperopt(Xtr_full, y_on_full, base_params: dict):
        if pd.Series(y_on_full).nunique() < 2:
            return None, base_params, np.nan

        X_tr, y_tr, X_va, y_va = _time_order_split(Xtr_full, y_on_full, val_frac_last_fold)

        space = {
            "n_estimators": scope.int(hp.quniform("clf_n_estimators", 200, 1600, 50)),
            "learning_rate": hp.loguniform("clf_learning_rate", np.log(0.01), np.log(0.25)),
            "max_depth": scope.int(hp.quniform("clf_max_depth", 2, 8, 1)),
            "subsample": hp.uniform("clf_subsample", 0.6, 1.0),
            "colsample_bytree": hp.uniform("clf_colsample_bytree", 0.6, 1.0),
            "min_child_weight": scope.int(hp.quniform("clf_min_child_weight", 1, 10, 1)),
            "reg_lambda": hp.loguniform("clf_lambda", np.log(1e-3), np.log(20.0)),
            "reg_alpha": hp.loguniform("clf_alpha", np.log(1e-4), np.log(5.0)),
        }

        def objective(hp_params):
            hp_params = _cast_params(hp_params)
            params = dict(base_params)
            params.update(hp_params)
            params.update({
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "n_jobs": -1,
                "random_state": random_state_opt,
            })

            clf = XGBClassifier(**params)
            clf.fit(X_tr, y_tr.astype(int))
            p = clf.predict_proba(X_va)[:, 1]
            pred = (p >= float(clf_threshold)).astype(int)

            # optimize for misclassification rate (stable for gating)
            loss = float(np.mean(pred != y_va.to_numpy().astype(int)))
            return {"loss": loss, "status": STATUS_OK}

        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=int(max_evals),
            trials=trials,
            rstate=np.random.default_rng(random_state_opt),
            show_progressbar=True,   # progress
            verbose=True,
        )

        best_typed = _cast_params({
            "n_estimators": int(best["clf_n_estimators"]),
            "learning_rate": float(best["clf_learning_rate"]),
            "max_depth": int(best["clf_max_depth"]),
            "subsample": float(best["clf_subsample"]),
            "colsample_bytree": float(best["clf_colsample_bytree"]),
            "min_child_weight": int(best["clf_min_child_weight"]),
            "reg_lambda": float(best["clf_lambda"]),
            "reg_alpha": float(best["clf_alpha"]),
        })

        final_params = dict(base_params)
        final_params.update(best_typed)
        final_params.update({
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_jobs": -1,
            "random_state": random_state_opt,
        })

        clf = XGBClassifier(**final_params)
        clf.fit(Xtr_full, y_on_full.astype(int))

        best_loss = float(min(t["result"]["loss"] for t in trials.trials if "result" in t and "loss" in t["result"]))
        return clf, final_params, best_loss

    # ----------------------------
    # Validate inputs
    # ----------------------------
    if season_col not in df.columns:
        raise ValueError(f"df must contain '{season_col}' column")
    if target_col not in df.columns:
        raise ValueError(f"df must contain '{target_col}'")

    work = df.copy().sort_index()
    seasons = sorted(work[season_col].dropna().unique().tolist())
    if len(seasons) < 2:
        raise ValueError("Need at least 2 seasons for walk-forward evaluation")

    if test_seasons is None:
        test_seasons = seasons[1:]
    if len(test_seasons) == 0:
        raise ValueError("test_seasons resolved to empty list")

    #print(test_seasons)

    last_test_season = test_seasons[-1]

    # ----------------------------
    # Targets
    # ----------------------------
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work["y"] = work[target_col].shift(-1)

    if two_stage:
        if on_col not in work.columns:
            raise ValueError(f"two_stage=True requires '{on_col}' column")
        work[on_col] = pd.to_numeric(work[on_col], errors="coerce")
        work["y_on"] = (work[on_col].shift(-1) > 0.5).astype("int64")

    # ----------------------------
    # Lags for target
    # ----------------------------
    for L in range(1, n_lags + 1):
        work[f"{target_col}_lag{L}"] = work[target_col].shift(L)

    # ----------------------------
    # Optional lags
    # ----------------------------
    lag_steps = None
    if add_temp_lags is not None:
        if not isinstance(add_temp_lags, (list, tuple)) or len(add_temp_lags) == 0:
            raise ValueError("add_temp_lags must be None or a non-empty list")
        lag_steps = [int(x) for x in add_temp_lags]
        if any(L <= 0 for L in lag_steps):
            raise ValueError("All add_temp_lags must be positive integers")

        if temp_lag_cols is None:
            temp_lag_cols = [
                "t_amb", "t_sup_prim", "t_ret_prim", "t_sup_sec", "t_ret_sec",
                "temp", "humidity", "windspeed", "sealevelpressure"
            ]

        for c in temp_lag_cols:
            if c not in work.columns:
                continue
            if not pd.api.types.is_numeric_dtype(work[c]):
                work[c] = pd.to_numeric(work[c], errors="coerce")
            for L in lag_steps:
                work[f"{c}_lag{L}"] = work[c].shift(L)

    # ----------------------------
    # Features
    # ----------------------------
    if base_features is None:
        #base_features = df.columns.difference([season_col, target_col, "y"]).tolist()
        base_features = [c for c in df.columns if c != "y_on" and c!= "y"]
        #if two_stage:
        #    base_features = [c for c in base_features if c != "y_on" and c!= "y"]

    lag_features = [f"{target_col}_lag{L}" for L in range(1, n_lags + 1)]

    extra_lag_features = []
    if lag_steps is not None:
        for c in (temp_lag_cols or []):
            if c in work.columns:
                for L in lag_steps:
                    name = f"{c}_lag{L}"
                    if name in work.columns:
                        extra_lag_features.append(name)

    features = lag_features + extra_lag_features + list(base_features)
    seen = set()
    features = [c for c in features if c in work.columns and not (c in seen or seen.add(c))]

    if drop_cols:
        drop_set = set(drop_cols)
        features = [c for c in features if c not in drop_set]

    # ----------------------------
    # Default params
    # ----------------------------
    if xgb_params is None:
        xgb_params = dict(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            min_child_weight=1.0,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        )

    if two_stage and xgb_clf_params is None:
        xgb_clf_params = dict(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            min_child_weight=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )

    # ----------------------------
    # Walk-forward
    # ----------------------------
    models = {}
    oof_pred = pd.Series(index=work.index, dtype="float64")
    rows = []
    best_params_last_fold = None

    for test_season in test_seasons:
        train_seasons = [s for s in seasons if s < test_season]
        train_mask = work[season_col].isin(train_seasons)
        test_mask = work[season_col].isin([test_season])

        X_train = work.loc[train_mask, features]
        y_train = work.loc[train_mask, "y"]
        X_test = work.loc[test_mask, features]
        y_test = work.loc[test_mask, "y"]

        train_ok = y_train.notna()
        test_ok = y_test.notna()

        if two_stage:
            y_on_train = work.loc[train_mask, "y_on"]
            y_on_test = work.loc[test_mask, "y_on"]
            train_ok &= y_on_train.notna()
            test_ok &= y_on_test.notna()

        if require_complete_cases:
            train_ok &= X_train.notna().all(axis=1)
            test_ok &= X_test.notna().all(axis=1)

        Xtr = X_train.loc[train_ok]
        ytr = y_train.loc[train_ok]
        Xte = X_test.loc[test_ok]
        yte = y_test.loc[test_ok]

        if len(Xtr) < 500:
            raise ValueError(f"Too few training rows for test season {test_season}: {len(Xtr)}")
        if len(Xte) < 100:
            if verbose:
                print(f"[WARN] Too few test rows for season {test_season}: {len(Xte)} (skipping)")
            continue

        is_last_fold = bool(optimize_last_fold and (test_season == last_test_season))

        # Timing holders
        opt_time_reg_s = np.nan
        opt_time_clf_s = np.nan
        fit_time_reg_s = np.nan
        fit_time_clf_s = np.nan
        pred_time_reg_s = np.nan
        pred_time_clf_s = np.nan

        # ----- classifier (optional) -----
        clf = None
        on_pred = None
        clf_params_used = dict(xgb_clf_params) if two_stage else None

        if two_stage:
            y_on_tr = work.loc[train_mask, "y_on"].loc[train_ok].astype(int)

            if is_last_fold:
                t0 = time.perf_counter()
                clf_opt, clf_params_opt, clf_best_loss = _optimize_classifier_hyperopt(Xtr, y_on_tr, clf_params_used)
                opt_time_clf_s = time.perf_counter() - t0

                if clf_opt is not None:
                    clf = clf_opt
                    clf_params_used = clf_params_opt
                else:
                    clf = XGBClassifier(**clf_params_used)
            else:
                clf = XGBClassifier(**clf_params_used)

            if y_on_tr.nunique() < 2:
                only_class = int(y_on_tr.iloc[0])
                if verbose:
                    print(f"[WARN] Season {test_season}: classifier degenerate (only class={only_class}); using constant.")
                on_pred = np.full(shape=len(Xte), fill_value=only_class, dtype=int)
                fit_time_clf_s = 0.0
                pred_time_clf_s = 0.0
            else:
                t0 = time.perf_counter()
                clf.fit(Xtr, y_on_tr)
                fit_time_clf_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                p_on = clf.predict_proba(Xte)[:, 1]
                on_pred = (p_on >= float(clf_threshold)).astype(int)
                pred_time_clf_s = time.perf_counter() - t0

        # ----- regressor -----
        reg_params_used = dict(xgb_params)

        if is_last_fold:
            t0 = time.perf_counter()
            reg_opt, reg_params_opt, reg_best_loss = _optimize_regressor_hyperopt(Xtr, ytr, reg_params_used)
            opt_time_reg_s = time.perf_counter() - t0

            reg = reg_opt
            reg_params_used = reg_params_opt

            best_params_last_fold = {"reg": reg_params_used, "reg_best_loss": float(reg_best_loss)}
            if two_stage:
                best_params_last_fold["clf"] = clf_params_used
                try:
                    best_params_last_fold["clf_best_loss"] = float(clf_best_loss)
                except Exception:
                    best_params_last_fold["clf_best_loss"] = np.nan

            # reg already fitted inside optimizer final refit
            fit_time_reg_s = np.nan  # unknown here (included in opt_time_reg_s)
        else:
            reg = XGBRegressor(**reg_params_used)
            t0 = time.perf_counter()
            reg.fit(Xtr, ytr)
            fit_time_reg_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred_reg = reg.predict(Xte)
        pred_time_reg_s = time.perf_counter() - t0

        # ----- combine + total prediction time -----
        pred_time_total_s = (0.0 if np.isnan(pred_time_clf_s) else pred_time_clf_s) + pred_time_reg_s

        if two_stage:
            pred_final = np.where(on_pred == 1, pred_reg, 0.0)
            models[test_season] = {"clf": clf, "reg": reg}
        else:
            pred_final = pred_reg
            models[test_season] = {"clf": None, "reg": reg}

        

        oof_pred.loc[Xte.index] = pred_final

        # ----- metrics on final -----
        y_true = yte.to_numpy()
        mae = float(np.mean(np.abs(pred_final - y_true)))
        rmse = float(np.sqrt(np.mean((pred_final - y_true) ** 2)))
        r2 = r2_score(yte, pred_final)

        naive_train = ytr.shift(1)
        naive_mask = naive_train.notna()
        mae_naive = float(np.mean(np.abs(ytr[naive_mask] - naive_train[naive_mask]))) if naive_mask.any() else np.nan
        mase = mae / mae_naive if (np.isfinite(mae_naive) and mae_naive > 0) else np.nan

        smape = _smape_safe(y_true, pred_final)

        pred_time_per_row_ms = float(pred_time_total_s / len(Xte) * 1000.0)

        rows.append(dict(
            test_season=test_season,
            train_seasons=train_seasons,
            n_train=len(Xtr),
            n_test=len(Xte),
            MAE=mae,
            RMSE=rmse,
            R2=r2,
            MASE=mase,
            sMAPE=smape,
            two_stage=bool(two_stage),
            clf_threshold=float(clf_threshold) if two_stage else np.nan,
            optimized_last_fold=bool(is_last_fold),

            # ---- timings ----
            opt_time_reg_s=opt_time_reg_s,
            opt_time_clf_s=opt_time_clf_s,
            fit_time_reg_s=fit_time_reg_s,
            fit_time_clf_s=fit_time_clf_s,
            pred_time_reg_s=pred_time_reg_s,
            pred_time_clf_s=pred_time_clf_s,
            pred_time_total_s=pred_time_total_s,
            pred_time_per_row_ms=pred_time_per_row_ms,
        ))

        if verbose:
            msg = (
                f"Season {test_season} | "
                f"MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}, "
                f"MASE={mase:.3f}, sMAPE={smape:.2f}%"
                f" | pred={pred_time_per_row_ms:.3f} ms/row"
            )
            if two_stage:
                msg += f" | ON-rate={float(np.mean(on_pred)):.2f}"
            if is_last_fold:
                msg += " | OPT(last fold)"
            print(msg)

    metrics = pd.DataFrame(rows)
    return models, oof_pred, metrics, features, work, best_params_last_fold, y_true, pred_final



def summarize_walkforward_results(
    df: pd.DataFrame,
    oof_pred: pd.Series,
    season_col: str = "season",
    target_col: str = "qizm",
):
    """
    Computes macro (season-balanced) and micro (global OOS) metrics.

    Returns
    -------
    summary : pd.DataFrame
        Table with macro_mean, macro_std and micro metrics.
    """

    work = df.copy().sort_index()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work["y"] = work[target_col].shift(-1)

    # Keep only rows where we have OOS prediction
    valid_mask = oof_pred.notna() & work["y"].notna()

    y_true_all = work.loc[valid_mask, "y"]
    y_pred_all = oof_pred.loc[valid_mask]

    # -----------------------------
    # MICRO (global)
    # -----------------------------
    mae_micro = np.mean(np.abs(y_true_all - y_pred_all))
    rmse_micro = np.sqrt(np.mean((y_true_all - y_pred_all) ** 2))
    r2_micro = r2_score(y_true_all, y_pred_all)

    # sMAPE
    denom = (np.abs(y_true_all) + np.abs(y_pred_all)) / 2.0
    smape_micro = np.mean(
        np.where(denom == 0, 0, np.abs(y_true_all - y_pred_all) / denom)
    ) * 100

    # Global naive benchmark (lag-1)
    naive_all = y_true_all.shift(1)
    naive_mask = naive_all.notna()
    mae_naive_global = np.mean(np.abs(y_true_all[naive_mask] - naive_all[naive_mask]))
    mase_micro = mae_micro / mae_naive_global if mae_naive_global > 0 else np.nan

    # -----------------------------
    # MACRO (season-balanced)
    # -----------------------------
    macro_rows = []

    for season in sorted(work.loc[valid_mask, season_col].unique()):
        season_mask = valid_mask & (work[season_col] == season)

        y_true = work.loc[season_mask, "y"]
        y_pred = oof_pred.loc[season_mask]

        if len(y_true) == 0:
            continue

        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        r2 = r2_score(y_true, y_pred)

        denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        smape = np.mean(
            np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom)
        ) * 100

        # naive seasonal benchmark
        naive = y_true.shift(1)
        mask = naive.notna()
        mae_naive = np.mean(np.abs(y_true[mask] - naive[mask]))
        mase = mae / mae_naive if mae_naive > 0 else np.nan

        macro_rows.append([mae, rmse, r2, mase, smape])

    macro_array = np.array(macro_rows)

    macro_mean = macro_array.mean(axis=0)
    macro_std = macro_array.std(axis=0)

    # -----------------------------
    # Summary Table
    # -----------------------------
    summary = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R2", "MASE", "sMAPE"],
        "Macro Mean": macro_mean,
        "Macro Std": macro_std,
        "Micro (Global OOS)": [
            mae_micro,
            rmse_micro,
            r2_micro,
            mase_micro,
            smape_micro,
        ],
    })

    return summary


def plot_actual_vs_predicted(y_true, pred_final, figsize=(10, 10), title="Actual vs Predicted", bins=30, alpha_hist=0.3):
    """
    Create a scatter plot of actual vs predicted values with a histogram of actual values on twin y-axis.
    
    Parameters
    ----------
    y_true : array-like
        True values
    pred_final : array-like
        Predicted values
    figsize : tuple, optional
        Figure size (width, height) in inches (default is (10, 10) for square plot)
    title : str, optional
        Plot title (default is "Actual vs Predicted")
    bins : int, optional
        Number of histogram bins (default is 30)
    alpha_hist : float, optional
        Transparency of histogram bars (default is 0.3)
        
    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Matplotlib figure and axes objects
    """
    
    # Convert to numpy arrays if needed
    y_true = np.array(y_true)
    pred_final = np.array(pred_final)
    
    # Create figure and axis with square aspect
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot scatter: actual on x-axis, predicted on y-axis
    ax.scatter(y_true, pred_final, alpha=0.6, s=20, edgecolors='k', linewidth=0.3, label='Predictions')
    
    # Plot diagonal line (perfect prediction) starting from origin
    max_val = max(y_true.max(), pred_final.max())
    # ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', label='Perfect prediction')

    # Set same limits for both axes starting from 0
    #ax.set_xlim(0, max_val)
    #ax.set_ylim(0, max_val)
    
    # Set equal aspect ratio
    #ax.set_aspect('equal', adjustable='box')
    
    # Create a twin y-axis for the histogram
    ax2 = ax.twinx()
    ax2.hist(y_true, bins=30, alpha=0.3, color='green', orientation='vertical')

    
    
    # Set labels
    ax.set_xlabel('Actual Values (KWh)', fontsize=11)
    ax.set_ylabel('Predicted Values (KWh)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Add grid
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax, ax2


def get_feature_importances(model, feature_names, importance_type='gain', top_n=None):
    """
    Extract feature importances from XGBoost model(s).
    
    Parameters
    ----------
    model : XGBRegressor, XGBClassifier, or dict
        - Single XGBoost model (regressor or classifier)
        - Dict with keys 'reg' and/or 'clf' (for two-stage models)
    feature_names : list
        List of feature names corresponding to the model's training features
    importance_type : str, optional
        Type of importance to extract: 'gain', 'weight', 'cover', 'total_gain', 'total_cover'
        (default is 'gain')
    top_n : int, optional
        If specified, return only the top N most important features (default is None - returns all)
    
    Returns
    -------
    pd.DataFrame or dict of pd.DataFrame
        - If single model: DataFrame with columns ['feature', 'importance'] sorted by importance
        - If dict model (two-stage): dict with keys 'reg' and/or 'clf', each containing a DataFrame
    
    Examples
    --------
    # Single-stage model
    importances = get_feature_importances(models[7], feat_cols)
    
    # Two-stage model
    importances = get_feature_importances(models[7], feat_cols)
    # Access: importances['reg'] and importances['clf']
    """
    
    def _extract_importances(xgb_model, feat_names, imp_type, n):
        """Helper to extract importances from a single XGBoost model."""
        try:
            # Get importance dictionary from XGBoost model
            imp_dict = xgb_model.get_booster().get_score(importance_type=imp_type)
            
            # Create DataFrame
            df = pd.DataFrame([
                {'feature': feat, 'importance': imp} 
                for feat, imp in imp_dict.items()
            ])
            
            # Sort by importance descending
            df = df.sort_values('importance', ascending=False).reset_index(drop=True)
            
            # Add features with zero importance if they're in feature_names but not in imp_dict
            missing_features = set(feat_names) - set(df['feature'])
            if missing_features:
                missing_df = pd.DataFrame([
                    {'feature': feat, 'importance': 0.0} 
                    for feat in missing_features
                ])
                df = pd.concat([df, missing_df], ignore_index=True)
            
            # Return top N if specified
            if n is not None:
                df = df.head(n)
            
            return df
            
        except Exception as e:
            print(f"Warning: Could not extract importances - {str(e)}")
            return pd.DataFrame({'feature': feat_names, 'importance': 0.0})
    
    # Handle dict of models (two-stage)
    if isinstance(model, dict):
        result = {}
        
        if 'reg' in model and model['reg'] is not None:
            result['reg'] = _extract_importances(model['reg'], feature_names, importance_type, top_n)
        
        if 'clf' in model and model['clf'] is not None:
            result['clf'] = _extract_importances(model['clf'], feature_names, importance_type, top_n)
        
        return result
    
    # Handle single model
    else:
        return _extract_importances(model, feature_names, importance_type, top_n)


def plot_feature_rank_heatmap(feature_importances_dict, importance_type='gain', figsize=None, cmap='RdYlGn_r'):
    """
    Plot a heatmap showing feature importance ranks across walk-forward validation seasons.
    
    Parameters
    ----------
    feature_importances_dict : dict
        Dictionary where keys are season/walk numbers and values are DataFrames 
        with 'feature' and 'importance' columns (output from get_feature_importances)
    importance_type : str, optional
        Type of importance used ('gain', 'weight', 'cover', etc.) - shown in title (default is 'gain')
    figsize : tuple, optional
        Figure size (width, height). If None, automatically sized based on data (default is None)
    cmap : str, optional
        Colormap name. Default 'RdYlGn_r' shows red for low ranks (important) 
        to green for high ranks (less important)
    
    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Matplotlib figure and axes objects
        
    Examples
    --------
    # After getting feature importances for each walk
    feature_importances = {}
    for season in models.keys():
        feature_importances[season] = get_feature_importances(
            models[season]['reg'], feat_cols, importance_type='gain', top_n=20
        )
    
    # Plot heatmap
    fig, ax = plot_feature_rank_heatmap(feature_importances, importance_type='gain')
    plt.show()
    """
    
    # Create feature rank table across all walks
    rank_data = {}
    
    for season, df_importance in feature_importances_dict.items():
        # Sort by importance and add rank column (1 = most important)
        df_sorted = df_importance.sort_values('importance', ascending=False).reset_index(drop=True)
        df_sorted['rank'] = range(1, len(df_sorted) + 1)
        
        # Store ranks in dictionary with season as key
        rank_dict = dict(zip(df_sorted['feature'], df_sorted['rank']))
        rank_data[f'Walk {season}'] = rank_dict
    
    # Create DataFrame with features as index and walks as columns
    df_ranks = pd.DataFrame(rank_data)
    
    # Sort by average rank across all walks
    df_ranks['Avg Rank'] = df_ranks.mean(axis=1)
    df_ranks = df_ranks.sort_values('Avg Rank')
    
    # Prepare data for heatmap (exclude 'Avg Rank' column)
    walk_cols = [col for col in df_ranks.columns if col.startswith('Walk')]
    heatmap_data = df_ranks[walk_cols]
    
    # Determine figure size
    if figsize is None:
        width = max(10, len(walk_cols) * 1.5)
        height = max(8, len(heatmap_data) * 0.4)
        figsize = (width, height)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap - use reversed colormap so lower ranks (more important) are darker
    sns.heatmap(heatmap_data, 
                annot=True,           # Show rank numbers
                fmt='.0f',            # Integer format
                cmap=cmap,
                cbar_kws={'label': 'Rank'},
                linewidths=0.5,
                linecolor='gray',
                ax=ax)
    
    # Customize title with importance type
    title = f'Feature importance ranks across walk-forward validation\n'
    title += f'Importance type: {importance_type.upper()}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    ax.set_xlabel('Walk-Forward season', fontsize=11)
    ax.set_ylabel('Feature', fontsize=11)
    
    # Rotate labels for better readability
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig, ax