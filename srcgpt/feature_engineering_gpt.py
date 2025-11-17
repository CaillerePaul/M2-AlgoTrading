# srcgpt/feature_engineering_gpt.py

from typing import List, Optional

import numpy as np
import pandas as pd


def select_leaders_from_dtw(
    dtw_results: pd.DataFrame,
    target_col: str,
    top_k: int = 2,
) -> List[str]:
    """
    À partir des résultats LeadLagDTW, sélectionne des 'leaders' pour target_col.

    dtw_results doit contenir :
    - 'Leader'
    - 'Lagging'
    - 'Average Lag'
    """
    if dtw_results.empty:
        return []

    # On commence par les lignes où la target est "Lagging"
    mask = dtw_results["Lagging"] == target_col
    subset = dtw_results[mask].copy()

    # Si rien trouvé, on prend toutes les lignes où la target apparaît
    if subset.empty:
        subset = dtw_results[
            (dtw_results["Leader"] == target_col)
            | (dtw_results["Lagging"] == target_col)
        ].copy()

    if subset.empty:
        return []

    # On ordonne par Average Lag décroissant (heuristique)
    subset = subset.sort_values(by="Average Lag", ascending=False)

    leaders_raw = subset["Leader"].unique().tolist()

    # On enlève la target elle-même si elle apparaît dans "Leader"
    leaders = [l for l in leaders_raw if l != target_col]

    # On s'assure que tout est bien "hashable/simple" (par ex. pas de tuples chelous)
    cleaned_leaders: List[str] = []
    for l in leaders:
        # Si MultiIndex / tuple etc. on convertit en str (et ça ne matchera pas les colonnes, donc sera ignoré ensuite)
        if isinstance(l, (tuple, list)):
            cleaned_leaders.append(str(l))
        else:
            cleaned_leaders.append(l)

    return cleaned_leaders[:top_k]


def _ensure_series(obj: pd.Series | pd.DataFrame) -> Optional[pd.Series]:
    """
    Garantit qu'on travaille avec une Series.
    - Si obj est une Series -> ok.
    - Si obj est un DataFrame 1-colonne -> on retourne la première colonne.
    - Sinon -> None (on skip).
    """
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            return obj.iloc[:, 0]
        else:
            return None
    return None


def build_leadlag_supervised_dataset(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    target_col: str,
    leaders: List[str],
    n_lags: int = 5,
    use_filtered: bool = False,
    filtered_prices: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Construit un dataset supervisé avec :
    - lags de retours de la target et des leaders,
    - spreads de prix target - leader (+ lags),
    - éventuellement des prix filtrés (+ lags).

    Retour :
    - DataFrame avec colonnes [features..., target_col].
    """
    if target_col not in returns.columns:
        raise ValueError(f"Target column '{target_col}' not in returns.")

    # Fallback : si aucun leader fourni, on prend tous les autres actifs
    if not leaders:
        leaders = [c for c in returns.columns if c != target_col]

    # Copie de base
    df = returns.copy()
    feature_cols: List[str] = []

    # ---------- 1) Lags de la target ----------
    for lag in range(1, n_lags + 1):
        col_name = f"{target_col}_ret_lag{lag}"
        df[col_name] = returns[target_col].shift(lag)
        feature_cols.append(col_name)

    # ---------- 2) Lags de retours des leaders ----------
    for leader in leaders:
        if leader not in returns.columns:
            continue
        for lag in range(1, n_lags + 1):
            col_name = f"{leader}_ret_lag{lag}"
            df[col_name] = returns[leader].shift(lag)
            feature_cols.append(col_name)

    # ---------- 3) Spreads de prix target - leader ----------
    # On aligne les index de prices et returns
    common_idx = prices.index.intersection(returns.index)
    px = prices.loc[common_idx]
    df = df.loc[common_idx]

    target_px = _ensure_series(px[target_col]) if target_col in px.columns else None

    if target_px is not None:
        for leader in leaders:
            if leader == target_col:
                continue
            if leader not in px.columns:
                continue

            leader_obj = px[leader]
            leader_px = _ensure_series(leader_obj)
            if leader_px is None:
                # On ne sait pas réduire proprement -> on skip
                continue

            spread_name = f"spread_{target_col}_{leader}"
            # Series - Series -> Series
            spread = target_px - leader_px

            df[spread_name] = spread
            feature_cols.append(spread_name)

            # Lags du spread
            for lag in range(1, n_lags + 1):
                lag_name = f"{spread_name}_lag{lag}"
                df[lag_name] = spread.shift(lag)
                feature_cols.append(lag_name)

    # ---------- 4) Prix filtrés comme features ----------
    if use_filtered and (filtered_prices is not None):
        fp = filtered_prices.loc[common_idx]

        for col in fp.columns:
            base_series = _ensure_series(fp[col])
            if base_series is None:
                continue

            base_name = f"filt_{col}"
            df[base_name] = base_series
            feature_cols.append(base_name)

            for lag in range(1, n_lags + 1):
                lag_name = f"{base_name}_lag{lag}"
                df[lag_name] = base_series.shift(lag)
                feature_cols.append(lag_name)

    # ---------- 5) Nettoyage final & target ----------
    df = df.dropna()
    supervised = df[feature_cols + [target_col]]

    return supervised
