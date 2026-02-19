"""
Data module for XAI4HEAT

This module contains functions for data loading, preprocessing, and feature engineering
for district heating systems.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import requests
import zipfile
import io
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt


from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class RemovalDecision:
    column: str
    reason: str

def get_scada_data(substation: str,
             strip_out_season_data: bool = True,
             verbose: bool = True,
             interpolate: str = None,
             handle_outliers: bool = False,
             handle_outliers_contamination: float = 0.01
             ) -> pd.DataFrame:
    """
    Load and return data for a given substation.
    
    Parameters
    ----------
    substation : str
        Name or identifier of the substation (e.g., 'L4', 'L8', 'L12', 'L17', 'L22')
    strip_out_season_data : bool, optional
        Whether to remove seasonal data from the dataset (default is True)
    verbose : bool, optional
        Whether to print additional information during data loading (default is True)
    handle_outliers : bool, optional
        Whether to handle outliers in the dataset (default is False)
    handle_outliers_contamination : float, optional
        The proportion of outliers in the dataset when handling outliers (default is 0.01)
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the substation data
    """

    github_folder = 'https://github.com/xai4heat/xai4heat/raw/main/datasets/'
    scada_1_path = f'xai4heat_scada_{substation}.csv'
    weather_path_1 = 'weather/ni_20_rs.csv'
    weather_path_2 = "weather/weather_2024_25.txt"



    #Opening weather file
    dfw_1 = pd.read_csv(github_folder+weather_path_1)
    dfw_1['datetime'] = pd.to_datetime(dfw_1['datetime'])
    dfw_1.set_index('datetime', inplace=True)

    dfw_2 = pd.read_csv(github_folder+weather_path_2)
    dfw_2['datetime'] = pd.to_datetime(dfw_2['datetime'])
    dfw_2.set_index('datetime', inplace=True)

    # Removing irrelevant data
    dfw_1 = dfw_1.drop(['name',
                    'precipprob',
                    'preciptype',
                    'icon',
                    'stations', 'severerisk', 'conditions',
                    'precip',
                    'snow',
                    'snowdepth',
                    'windgust',
                    'cloudcover',
                    'visibility',
                    'solarradiation',
                    'solarenergy',
                    'uvindex', 'winddir'], axis=1, errors='ignore')
    
    # Removing irrelevant data
    dfw_2 = dfw_2.drop(['name',
                    'precipprob',
                    'preciptype',
                    'icon',
                    'stations', 'severerisk', 'conditions',
                    'precip',
                    'snow',
                    'snowdepth',
                    'windgust',
                    'cloudcover',
                    'visibility',
                    'solarradiation',
                    'solarenergy',
                    'uvindex', 'winddir'], axis=1, errors='ignore')
             

    df = pd.read_csv(github_folder+scada_1_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime',inplace=True)
    # Keep only full hour data (15-min resolution always has data at full hour)
    df = df[df.index.minute == 0]
    df = df.sort_index()



    #Transmitted energy.
    qizm=(df['e']-df['e'].shift(1))*1000
    df['qizm']=qizm
    dropcolumns=['e',
                 'pe']
    df = df.drop(columns=dropcolumns)
    #Merging with weather data
    df = pd.merge(df, dfw_1, left_index=True, right_index=True, how='outer')


    # Opening 2024-25 SCADA data, resampling to full hour, and merging with existing dataframe
    url_scada = "https://github.com/xai4heat/xai4heat/raw/main/datasets/scada_data_2024-25.zip"
    response_scada = requests.get(url_scada)
    with zipfile.ZipFile(io.BytesIO(response_scada.content)) as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        with z.open(csv_files[0]) as f:
            dfs = pd.read_csv(f)

    dfs= dfs[dfs['location'] == f'TPS Lamela {substation}']
    dfs.drop(columns='location', inplace=True)
    
    # Process dfs: set datetime index and resample to full hour
    dfs['datetime'] = pd.to_datetime(dfs['datetime'])
    dfs.set_index('datetime', inplace=True)
    
    # For each hour, find the timestamp closest to the full hour
    dfs["_hour"] = dfs.index.floor("H")
    dfs["_dist_sec"] = (dfs.index - dfs["_hour"]).abs().dt.total_seconds()
    dfs_hourly = (
        dfs.sort_values(["_hour", "_dist_sec"])
            .groupby("_hour", as_index=False)
            .first()
    )
    # Set the full-hour timestamp as the index
    dfs_hourly.set_index("_hour", inplace=True)
    dfs_hourly.index.name = "datetime"
    dfs_hourly = dfs_hourly.drop(columns=["_dist_sec"])
    #Transmitted energy.
    qizm=(dfs_hourly['e']-dfs_hourly['e'].shift(1))*1000
    dfs_hourly['qizm']=qizm
    dropcolumns=['e',
                 'pe']
    dfs_hourly = dfs_hourly.drop(columns=dropcolumns)

    dfs_hourly = pd.merge(dfs_hourly, dfw_2, left_index=True, right_index=True, how='outer')


    # Concatenate dataframes
    df = pd.concat([df, dfs_hourly])
    
    # Drop duplicate indices (keep last to prioritize newer 2024-25 data)
    df = df[~df.index.duplicated(keep='last')]
    
    # Sort by index to maintain chronological order
    df = df.sort_index()


    
    # Filter data for substation L12 before October 8th, 2021
    if substation in ['L12', 'L22', 'L8']:
        df = df[df.index >= pd.to_datetime('2021-10-08')]
        print("Filtered L12, L22, and L8 data to only include timestamps from October 8th, 2021 onwards due to data quality issues before that date.")



    if(verbose):
        print('Timeline (from/to): ', df.index.min(), df.index.max())


    

    #Insert missing timepoints, populate with NaNs
    complete_time_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
    df = df.reindex(complete_time_index)

    # Remove data outside of the heating season
    if strip_out_season_data:
        date_range_season1 = (df.index >= pd.to_datetime('2018-11-18 06:00:00')) & (df.index < pd.to_datetime('2019-03-31 23:00:00'))
        date_range_season2 = (df.index >= pd.to_datetime('2019-11-18 06:00:00')) & (df.index < pd.to_datetime('2020-03-31 23:00:00'))
        date_range_season3 = (df.index >= pd.to_datetime('2020-11-18 06:00:00')) & (df.index < pd.to_datetime('2021-03-31 23:00:00'))
        date_range_season4 = (df.index >= pd.to_datetime('2021-11-18 06:00:00')) & (df.index < pd.to_datetime('2022-03-31 23:00:00'))
        date_range_season5 = (df.index >= pd.to_datetime('2022-11-18 06:00:00')) & (df.index < pd.to_datetime('2023-03-31 23:00:00'))
        date_range_season6 = (df.index >= pd.to_datetime('2023-11-18 06:00:00')) & (df.index < pd.to_datetime('2024-03-31 23:00:00'))
        date_range_season7 = (df.index >= pd.to_datetime('2024-11-18 06:00:00')) & (df.index < pd.to_datetime('2025-03-31 23:00:00'))
        date_range_season8 = (df.index >= pd.to_datetime('2025-11-18 06:00:00')) & (df.index < pd.to_datetime('2026-03-31 23:00:00'))
        df = df[date_range_season1 | date_range_season2 | date_range_season3 | date_range_season4 | date_range_season5 | date_range_season6 | date_range_season7 | date_range_season8]


    df['qizm'] = df['qizm'].round(1)
    df['qizm'] = df['qizm'].apply(lambda x: 0 if x < 31 else x)
    df['qizm'] = df['qizm'].apply(lambda x: 0 if x >= 1000 else x)


    df = df.drop(columns=["t_ref"])


    # Linear interpolation for weather columns
    weather_cols = ['temp', 'feelslike', 'dew', 'humidity', 'windspeed', 'sealevelpressure']
    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear')


    temp_cols = ['t_sup_prim', 't_ret_prim', 't_sup_sec', 't_ret_sec']
    existing = [c for c in temp_cols if c in df.columns]

    if existing:
        mask_bad = (df[existing] == 0).any(axis=1)

        df[existing] = df[existing].replace(0, np.nan)

        if "t_amb" in df.columns:
            df.loc[mask_bad, "t_amb"] = np.nan


    # Handle outliers by replacing them with NaN before imputation
    if handle_outliers:
        if verbose:
            print(f"Detecting and removing outliers with contamination={handle_outliers_contamination}")
        
        # Detect outliers
        outliers_df = get_outliers_isolation_forest(df.copy(), contamination=handle_outliers_contamination, verbose=verbose)
        
        # Get outlier indices
        outlier_indices = outliers_df.index
        
        # Replace outliers with NaN for specified columns
        outlier_cols = ['qizm', 't_sup_prim', 't_ret_prim', 't_sup_sec', 't_ret_sec']
        for col in outlier_cols:
            if col in df.columns:
                df.loc[outlier_indices, col] = np.nan
        
        if verbose:
            print(f"Replaced {len(outlier_indices)} outlier timestamps with NaN in {[c for c in outlier_cols if c in df.columns]}")


    if(interpolate=='linear'):
        # Interpolate remaining NaN values linearly
        df.interpolate(method='linear', inplace=True)
    if(interpolate=='model'):
        df=impute_by_model(df)
        df['qizm'] = df['qizm'].round(1)
        df['qizm'] = df['qizm'].apply(lambda x: 0 if x < 31 else x)

    # New features
    df['hour_of_day'] = df.index.hour
    df['heating_on'] = df['qizm'].apply(lambda x: 1 if x >= 31 else 0)
    df['delta_primary'] = df['t_sup_prim'] - df['t_ret_prim']
    df['delta_secondary'] = df['t_sup_sec'] - df['t_ret_sec']
    df['sin_hour'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
   
    
    df = df.drop(columns=["feelslike", "dew"], errors='ignore')

    # Adding season labels for walk-forward evaluation (1, 2, 3, ... for each season; NaN for non-season data)
    idx = pd.to_datetime(df.index)
    starts = pd.to_datetime([
    "2018-11-18 06:00:00",
    "2019-11-18 06:00:00",
    "2020-11-18 06:00:00",
    "2021-11-18 06:00:00",
    "2022-11-18 06:00:00",
    "2023-11-18 06:00:00",
    "2024-11-18 06:00:00",
    "2025-11-18 06:00:00",
    ])
    ends = pd.to_datetime([
    "2019-03-31 23:00:00",
    "2020-03-31 23:00:00",
    "2021-03-31 23:00:00",
    "2022-03-31 23:00:00",
    "2023-03-31 23:00:00",
    "2024-03-31 23:00:00",
    "2025-03-31 23:00:00",
    "2026-03-31 23:00:00",
    ])
    conditions = [(idx >= s) & (idx < e) for s, e in zip(starts, ends)]
    choices = np.arange(1, len(conditions) + 1)
    df["season"] = pd.Series(np.select(conditions, choices, default=np.nan), index=df.index).astype("Int64")
    min_season = df["season"].min()
    df['season'] = df['season'] - min_season + 1



    # Save the combined DataFrame with full-hour timestamps
    df.to_csv(f'{substation}_combined_data.csv')

    return df





def impute_by_model(
    df: pd.DataFrame,
    max_iter_temps: int = 4,
    lags=(1, 2),
    use_calendar_features: bool = True,
    random_state: int = 42,
    return_clean: bool = True,
) -> pd.DataFrame:
    """
    Model-based imputations.

    CURRENTLY IMPLEMENTED
    ---------------------
    1) t_amb from temp (Ridge)
    2) temperature iterative imputation (LightGBM) WITHOUT using qizm as predictor
    3) qizm imputation (LightGBM, clipped >= 0) AFTER temperature imputation step

    Parameters
    ----------
    return_clean : bool
        If True, remove all helper columns generated inside the method.

    Returns
    -------
    pd.DataFrame
    """

    out = df.copy()
    created_cols = []  # track helper columns created by this function

    # ------------------------------------------------------------------
    # index
    # ------------------------------------------------------------------
    if not isinstance(out.index, pd.DatetimeIndex):
        if "datetime" in out.columns:
            out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
            out = out.set_index("datetime")
        else:
            raise ValueError("df must have DatetimeIndex or 'datetime' column")

    out = out.sort_index()

    def _to_num(cols):
        for c in cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")

    # ------------------------------------------------------------------
    # 1) t_amb from temp
    # ------------------------------------------------------------------
    if "t_amb" in out.columns and "temp" in out.columns:
        _to_num(["t_amb", "temp"])

        train_mask = out["t_amb"].notna() & out["temp"].notna()
        pred_mask = out["t_amb"].isna() & out["temp"].notna()

        if "t_amb_imputed_model" not in out.columns:
            out["t_amb_imputed_model"] = False

        if train_mask.sum() >= 20 and pred_mask.sum() > 0:
            model = Ridge(alpha=1.0)
            model.fit(out.loc[train_mask, ["temp"]], out.loc[train_mask, "t_amb"])

            out.loc[pred_mask, "t_amb"] = model.predict(out.loc[pred_mask, ["temp"]])
            out.loc[pred_mask, "t_amb_imputed_model"] = True

    # ------------------------------------------------------------------
    # calendar
    # ------------------------------------------------------------------
    if use_calendar_features:
        for c in ["_hour", "_dow", "_month"]:
            if c not in out.columns:
                created_cols.append(c)

        out["_hour"] = out.index.hour
        out["_dow"] = out.index.dayofweek
        out["_month"] = out.index.month

    # ------------------------------------------------------------------
    # lag builder
    # ------------------------------------------------------------------
    def _ensure_lags(base_cols):
        for c in base_cols:
            if c not in out.columns:
                continue
            for L in lags:
                name = f"{c}_lag{L}"
                if name not in out.columns:
                    created_cols.append(name)
                out[name] = out[c].shift(L)

    # ------------------------------------------------------------------
    # 2) temperatures (NO qizm as predictor)
    # ------------------------------------------------------------------
    temp_targets = ["t_sup_prim", "t_ret_prim", "t_sup_sec", "t_ret_sec"]
    if all(c in out.columns for c in temp_targets) and "t_amb" in out.columns:
        # NOTE: do NOT coerce qizm here; we are not using it for temperature models
        _to_num(["t_amb"] + temp_targets)

        for t in temp_targets:
            flag = f"{t}_imputed_model"
            if flag not in out.columns:
                out[flag] = False

        # lag base excludes qizm
        lag_base = ["t_amb"] + temp_targets
        _ensure_lags(lag_base)

        def _make_lgbm():
            return LGBMRegressor(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=63,
                min_child_samples=30,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1
            )

        def _features(target):
            feats = ["t_amb"]

            # other temps (current)
            feats += [x for x in temp_targets if x != target]

            # lags: target + its pair + t_amb
            feats += [f"{target}_lag{L}" for L in lags]

            if target in ("t_sup_prim", "t_ret_prim"):
                pair = "t_ret_prim" if target == "t_sup_prim" else "t_sup_prim"
            else:
                pair = "t_ret_sec" if target == "t_sup_sec" else "t_sup_sec"

            feats += [f"{pair}_lag{L}" for L in lags]
            feats += [f"t_amb_lag{L}" for L in lags]

            if use_calendar_features:
                feats += ["_hour", "_dow", "_month"]

            return [f for f in feats if f in out.columns]

        for _ in range(max_iter_temps):
            any_filled = False

            for target in temp_targets:
                feats = _features(target)

                train_mask = out[target].notna()
                pred_mask = out[target].isna()

                if pred_mask.sum() == 0 or train_mask.sum() < 50:
                    continue

                model = _make_lgbm()
                model.fit(out.loc[train_mask, feats], out.loc[train_mask, target])

                out.loc[pred_mask, target] = model.predict(out.loc[pred_mask, feats])
                out.loc[pred_mask, f"{target}_imputed_model"] = True
                any_filled = True

            if not any_filled:
                break

            # update lags because we changed temps
            _ensure_lags(lag_base)

    # ------------------------------------------------------------------
    # 3) qizm (AFTER temps are imputed)
    # ------------------------------------------------------------------
    if "qizm" in out.columns:
        _to_num(["qizm"] + (["t_amb"] if "t_amb" in out.columns else []) + temp_targets)

        if "qizm_imputed_model" not in out.columns:
            out["qizm_imputed_model"] = False

        q_feats = []

        if "t_amb" in out.columns:
            q_feats.append("t_amb")

        q_feats += [c for c in temp_targets if c in out.columns]

        # include ΔT features if computable (temps should now be more complete)
        if "t_sup_prim" in out.columns and "t_ret_prim" in out.columns:
            if "dT_prim" not in out.columns:
                created_cols.append("dT_prim")
            out["dT_prim"] = out["t_sup_prim"] - out["t_ret_prim"]
            q_feats.append("dT_prim")

        if "t_sup_sec" in out.columns and "t_ret_sec" in out.columns:
            if "dT_sec" not in out.columns:
                created_cols.append("dT_sec")
            out["dT_sec"] = out["t_sup_sec"] - out["t_ret_sec"]
            q_feats.append("dT_sec")

        # qizm lags (key)
        _ensure_lags(["qizm"])
        q_feats += [f"qizm_lag{L}" for L in lags]

        # optional calendar
        if use_calendar_features:
            q_feats += ["_hour", "_dow", "_month"]

        q_feats = [f for f in q_feats if f in out.columns]

        train_mask = out["qizm"].notna()
        pred_mask = out["qizm"].isna()

        if pred_mask.sum() > 0 and train_mask.sum() >= 50:
            model = LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                min_child_samples=30,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1
            )
            model.fit(out.loc[train_mask, q_feats], out.loc[train_mask, "qizm"])

            out.loc[pred_mask, "qizm"] = model.predict(out.loc[pred_mask, q_feats])

            # hourly transmitted energy should not be negative
            out["qizm"] = out["qizm"].clip(lower=0)
            out.loc[pred_mask, "qizm_imputed_model"] = True

        # cleanup dT helpers (they're tracked anyway, but keep it explicit)
        out = out.drop(columns=["dT_prim", "dT_sec"], errors="ignore")

    # ------------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------------
    if return_clean:
        out = out.drop(columns=[c for c in created_cols if c in out.columns], errors="ignore")
        out = out.drop(columns=[c for c in out.columns if c.endswith("_imputed_model")], errors="ignore")

    return out


def get_outliers_isolation_forest(df, contamination, verbose=False):
    features = ['t_amb', 'qizm']
    df=df[features]
    hourly_outliers=[]
    for hour in range(24):
        dataset_for_hour = df[df.index.hour == hour][features].copy()

        model = IsolationForest(contamination=contamination)
        model.fit(dataset_for_hour)
        outliers = model.predict(dataset_for_hour)
        dataset_for_hour['is_outlier'] = outliers
        
        dataset_for_hour = dataset_for_hour[dataset_for_hour['qizm'] != 0]
        dataset_for_hour = dataset_for_hour[dataset_for_hour['is_outlier'] == -1]
        hourly_outliers.append(dataset_for_hour)


    df = pd.concat(hourly_outliers)
    df.sort_index(inplace=True)


    if verbose:
      print(f"Number of outliers detected: {(df['is_outlier'] == -1).sum()}")
      
    return df


def plot_signals_with_outliers(df, outliers_df=None, title="Transmitted Energy and Ambient Temperature with Outliers"):
    """
    Plot transmitted energy (qizm) and ambient temperature (t_amb) signals with outliers indicated.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the full time series with 'qizm' and 't_amb' columns
    outliers_df : pd.DataFrame, optional
        DataFrame with outliers as returned by get_outliers_isolation_forest.
        If None, no outliers will be highlighted.
    title : str, optional
        Title for the plot
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plotly figure with dual y-axes
    """
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Plot qizm (transmitted energy) on primary y-axis
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['qizm'],
            mode='lines',
            name='qizm',
            line=dict(color='blue', width=1.5),
            hovertemplate='<b>Date</b>: %{x}<br><b>qizm</b>: %{y:.2f} Wh<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Plot t_amb (ambient temperature) on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['t_amb'],
            mode='lines',
            name='t_amb',
            line=dict(color='green', width=1.5),
            hovertemplate='<b>Date</b>: %{x}<br><b>t_amb</b>: %{y:.2f} °C<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Add outliers if provided
    if outliers_df is not None and not outliers_df.empty:
        # Plot outliers for qizm on primary y-axis
        fig.add_trace(
            go.Scatter(
                x=outliers_df.index,
                y=outliers_df['qizm'],
                mode='markers',
                name='Outliers (qizm)',
                marker=dict(color='red', size=8, symbol='x', line=dict(width=2)),
                hovertemplate='<b>Date</b>: %{x}<br><b>qizm</b>: %{y:.2f} Wh<br><b>Outlier</b><extra></extra>'
            ),
            secondary_y=False
        )
        
        # Plot outliers for t_amb on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=outliers_df.index,
                y=outliers_df['t_amb'],
                mode='markers',
                name='Outliers (t_amb)',
                marker=dict(color='darkred', size=8, symbol='x', line=dict(width=2)),
                hovertemplate='<b>Date</b>: %{x}<br><b>t_amb</b>: %{y:.2f} °C<br><b>Outlier</b><extra></extra>'
            ),
            secondary_y=True
        )
    
    # Update axes labels
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Transmitted Energy [Wh]", secondary_y=False)
    fig.update_yaxes(title_text="Temperature [°C]", secondary_y=True)
    
    # Update layout
    fig.update_layout(
        title=title,
        height=600,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def plot_monthly_median_energy_comparison(
    substations: list,
    areas: list = None,
    strip_out_season_data: bool = True,
    interpolate: str = None,
    handle_outliers: bool = False,
    handle_outliers_contamination: float = 0.01,
    verbose: bool = False,
    title: str = None,
    figsize: tuple = (14, 6)
):
    """
    Create a grouped bar plot comparing monthly median transmitted energy across substations.
    
    Parameters
    ----------
    substations : list
        List of substation identifiers (e.g., ['L4', 'L8', 'L12'])
    areas : list, optional
        List of heated areas in m² corresponding to each substation.
        If provided, energy will be normalized (Wh/m²)
    strip_out_season_data : bool, optional
        Whether to remove data outside heating season (default is True)
    interpolate : str, optional
        Interpolation method to use ('linear', 'model', or None)
    handle_outliers : bool, optional
        Whether to handle outliers (default is False)
    handle_outliers_contamination : float, optional
        Contamination parameter for outlier detection (default is 0.01)
    verbose : bool, optional
        Whether to print progress information (default is False)
    title : str, optional
        Custom title for the plot
    figsize : tuple, optional
        Figure size (width, height) in inches (default is (14, 6))
        
    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Matplotlib figure and axes objects
    """
    
    # Validate inputs
    if areas is not None and len(areas) != len(substations):
        raise ValueError(f"Length of areas ({len(areas)}) must match length of substations ({len(substations)})")
    
    # Store monthly median data for each substation
    monthly_data = {}
    
    # Load data for each substation
    for i, substation in enumerate(substations):
        if verbose:
            print(f"Loading data for substation {substation}...")
        
        # Get data
        df = get_scada_data(
            substation=substation,
            strip_out_season_data=strip_out_season_data,
            verbose=False,
            interpolate=interpolate,
            handle_outliers=handle_outliers,
            handle_outliers_contamination=handle_outliers_contamination
        )
        
        # Filter to only when heating is on (qizm > 0)
        df_heating_on = df[df['qizm'] > 0]
        
        # Calculate monthly median first (only on heating-on periods)
        df_monthly = df_heating_on['qizm'].resample('M').median()
        
        # Normalize by area if provided
        if areas is not None:
            df_monthly = df_monthly / areas[i]
        
        # Extract month number and group by month across all years
        month_numbers = df_monthly.index.month
        df_by_month = pd.DataFrame({'value': df_monthly.values, 'month': month_numbers})
        
        # Calculate median for each month across all years
        monthly_medians = df_by_month.groupby('month')['value'].median()
        
        # Store
        monthly_data[substation] = monthly_medians
    
    # Combine all data into a single DataFrame
    df_combined = pd.DataFrame(monthly_data)
    
    # Filter to heating season months only: Nov, Dec, Jan, Feb, Mar
    heating_months = [11, 12, 1, 2, 3]  # Month numbers
    df_combined = df_combined[df_combined.index.isin(heating_months)]
    
    # Reorder according to heating season: Nov, Dec, Jan, Feb, Mar
    df_combined = df_combined.reindex(heating_months)
    
    # Create month labels in heating season order
    month_labels = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar']
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up bar positions
    x = np.arange(len(month_labels))
    width = 0.8 / len(substations)  # Width of bars
    
    # Plot bars for each substation
    for i, substation in enumerate(substations):
        offset = width * i - (width * len(substations) / 2) + width / 2
        ax.bar(x + offset, df_combined[substation], width, label=substation)
    
    # Determine y-axis label and title
    if areas is not None:
        y_label = "Monthly Median Transmitted Energy [KWh/m²]"
        default_title = "Monthly Median Transmitted Energy per m² Across Substations"
    else:
        y_label = "Monthly Median Transmitted Energy [KWh]"
        default_title = "Monthly Median Transmitted Energy Across Substations"
    
    # Set labels and title
    ax.set_xlabel('Month')
    ax.set_ylabel(y_label)
    ax.set_title(title if title else default_title)
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax


def describe_substations(
    substations: List[str],
    areas: List[float] = None,
    strip_out_season_data: bool = True,
    interpolate: str = None,
    handle_outliers: bool = False,
    handle_outliers_contamination: float = 0.01,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Generate a comprehensive description of multiple substation datasets.
    
    Parameters
    ----------
    substations : List[str]
        List of substation names (e.g., ['L4', 'L8', 'L12', 'L17', 'L22'])
    areas : List[float], optional
        List of areas for each substation (default is None)
    strip_out_season_data : bool, optional
        Whether to remove out-of-season data (default is True)
    interpolate : str, optional
        Interpolation method: None, 'linear', or 'model' (default is None)
    handle_outliers : bool, optional
        Whether to handle outliers (default is False)
    handle_outliers_contamination : float, optional
        Contamination parameter for outlier detection (default is 0.01)
    verbose : bool, optional
        Whether to print progress information (default is False)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with substations as columns and statistics as rows
    """
    
    results = {}
    
    for i, substation in enumerate(substations):
        if verbose:
            print(f"Processing {substation}...")
        
        # Load data for the substation
        df = get_scada_data(
            substation=substation,
            strip_out_season_data=strip_out_season_data,
            verbose=False,
            interpolate=interpolate,
            handle_outliers=handle_outliers,
            handle_outliers_contamination=handle_outliers_contamination
        )
        
        # Calculate statistics
        stats = {}
        
        # Basic temporal info
        stats['First datetime'] = df.index.min().strftime('%Y-%m-%d %H:%M')
        stats['Last datetime'] = df.index.max().strftime('%Y-%m-%d %H:%M')
        stats['Total datapoints'] = len(df)
        
        seasons = df['season'].dropna().unique()
        stats['Number of seasons'] = len(seasons)

        stats['Heated area [m²]'] = areas[i]
    
            
        # Calculate total energy per season and get median
        season_totals = df.groupby('season')['qizm'].sum()
        season_totals_normalized = season_totals / areas[i]
        stats['Median total energy transmitted per season [kWh]'] = round(season_totals.median(), 2)
        stats['Mean total energy transmitted per season [kWh]'] = round(season_totals.mean(), 2)
        stats['Std total energy transmitted per season [kWh]'] = round(season_totals.std(), 2)
        
        stats['Normalized (per m2 of heating area) median total energy transmitted per season [kWh]'] = round(season_totals_normalized.median(), 2)


        # Energy statistics
        stats['Total energy transmitted [kWh]'] = round(df['qizm'].sum(), 2)
        stats['Mean energy transmitted [kWh]'] = round(df['qizm'].mean(), 2)
        stats['Median energy transmitted [kWh]'] = round(df['qizm'].median(), 2)
        stats['Max energy transmitted [kWh]'] = round(df['qizm'].max(), 2)

        stats['Normalized (per m2 of heating area) total energy transmitted [kWh]'] = round(df['qizm'].sum(), 2) / areas[i]
        
        # Heating operation statistics
        if 'heating_on' in df.columns:
            heating_on_pct = (df['heating_on'].sum() / len(df)) * 100
            stats['Heating on percentage [%]'] = round(heating_on_pct, 2)
            stats['Hours heating on'] = int(df['heating_on'].sum())
            stats['Hours heating off'] = int((df['heating_on'] == 0).sum())
        else:
            stats['Heating on percentage [%]'] = np.nan
            stats['Hours heating on'] = np.nan
            stats['Hours heating off'] = np.nan
        
        # Temperature statistics
        if 't_amb' in df.columns:
            stats['Mean ambient temperature [°C]'] = round(df['t_amb'].mean(), 2)
            stats['Min ambient temperature [°C]'] = round(df['t_amb'].min(), 2)
            stats['Max ambient temperature [°C]'] = round(df['t_amb'].max(), 2)
        
        
        results[substation] = stats
    
    # Convert to DataFrame with statistics as rows
    df_describe = pd.DataFrame(results)
    
    return df_describe
