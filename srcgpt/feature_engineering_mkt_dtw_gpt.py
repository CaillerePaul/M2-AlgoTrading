# srcgpt/feature_engineering_mkt_dtw_gpt.py
import logging

logger = logging.getLogger(__name__)


from typing import List, Tuple

import numpy as np
import pandas as pd


def extract_leaders_and_lags(
    dtw_results: pd.DataFrame,
    target_col: str,
    top_k: int = 3,
    max_lag: int = 10,
) -> List[Tuple[str, int]]:
    """
    À partir des résultats LeadLagDTW (Leader, Lagging, Average Lag),
    extrait les leaders pertinents pour target_col, avec un lag moyen discret.

    On prend les lignes où Lagging == target_col :
        Leader = actif qui 'lead'
        Average Lag ~ nombre moyen de pas entre les mouvements.

    Retourne une liste de (leader, lag), lag borné à [0, max_lag].
    """
    if dtw_results.empty:
        return []

    mask = dtw_results["Lagging"] == target_col
    subset = dtw_results[mask].copy()

    if subset.empty:
        # fallback : si aucune ligne avec target en "Lagging",
        # on prend les lignes où target apparaît quelque part
        subset = dtw_results[
            (dtw_results["Leader"] == target_col)
            | (dtw_results["Lagging"] == target_col)
        ].copy()

    if subset.empty:
        return []

    # On ordonne par Average Lag décroissant (heuristique)
    subset = subset.sort_values(by="Average Lag", ascending=False)

    leaders_and_lags: List[Tuple[str, int]] = []
    for _, row in subset.iterrows():
        leader = row["Leader"]
        # on s'assure que c'est bien un str
        if isinstance(leader, (tuple, list)):
            leader = leader[0]
        lag = int(round(float(row["Average Lag"])))
        lag = max(0, min(max_lag, lag))
        if leader != target_col:
            leaders_and_lags.append((leader, lag))

    # Déduplication en conservant l'ordre
    seen = set()
    out: List[Tuple[str, int]] = []
    for leader, lag in leaders_and_lags:
        if leader not in seen:
            seen.add(leader)
            out.append((leader, lag))
        if len(out) >= top_k:
            break

    return out


def build_mkt_dtw_features_supervised(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    prices_smooth: pd.DataFrame,
    dtw_results: pd.DataFrame,
    target_col: str,
    horizons: List[int] | None = None,
    top_k_leaders: int = 3,
) -> pd.DataFrame:
    """
    Construit un dataset supervisé pour prédire le log-return de target_col en combinant :

    - Features 'market' classiques :
        * Momenta de returns (log) pour tous les actifs et différents horizons.
        * Volatilités glissantes sur la target.
        * Ratio GC/SI (si dispo).

    - Features filtrées :
        * Prix Savitzky-Golay (prices_savgol).
        * Noise = prix brut - prix filtré.
        * Momenta filtrés (log(p_filt / p_filt_{t-h})).

    - Features DTW :
        * Pour chaque leader L de target_col (via dtw_results) :
            - returns alignés avec le lag DTW,
            - momemta alignés,
            - ratio target/leader aligné.

    Target = log-return courant de la target.
    """
    if horizons is None:
        horizons = [1, 3, 5, 10, 21]

    if target_col not in returns.columns:
        raise ValueError(f"Target column '{target_col}' not in returns.")

    # Alignement des index
    idx = prices.index.intersection(returns.index).intersection(prices_smooth.index)
    px = prices.loc[idx].copy()
    ret = returns.loc[idx].copy()
    px_filt = prices_smooth.loc[idx].copy()

    # DataFrame des features
    df = pd.DataFrame(index=idx)
    feature_cols: list[str] = []

    # ---------- 1) Features market 'brutes' (momenta & vol) ----------
    # Momenta log-returns pour tous les actifs
    for col in ret.columns:
        for h in horizons:
            col_name = f"{col}_mom_{h}"
            # somme des log-returns sur h jours, décalée de 1 pour éviter look-ahead
            df[col_name] = ret[col].rolling(h).sum().shift(1)
            feature_cols.append(col_name)

    # Volatilités glissantes sur la target
    for h in horizons:
        vol_name = f"{target_col}_vol_{h}"
        df[vol_name] = ret[target_col].rolling(h).std().shift(1)
        feature_cols.append(vol_name)

    # Ratio GC/SI si c'est pertinent (exemple spécifique)
    if target_col in px.columns and "SI=F" in px.columns:
        ratio = np.log(px[target_col] / px["SI=F"])
        for h in horizons:
            rname = f"ratio_{target_col}_SI=F_mom_{h}"
            df[rname] = ratio.rolling(h).sum().shift(1)
            feature_cols.append(rname)

    # ---------- 2) Features filtrées (Savitzky-Golay & noise) ----------
    for col in px.columns:
        if col not in px_filt.columns:
            continue

        # Prix filtré et noise
        filt_col = f"{col}_filt"
        noise_col = f"{col}_noise"

        df[filt_col] = px_filt[col].shift(1)           # niveau filtré (t-1)
        df[noise_col] = (px[col] - px_filt[col]).shift(1)
        feature_cols.extend([filt_col, noise_col])

        # Momenta filtrés (trend lissée)
        for h in horizons:
            fm_name = f"{col}_filt_mom_{h}"
            df[fm_name] = np.log(px_filt[col] / px_filt[col].shift(h)).shift(1)
            feature_cols.append(fm_name)

    # ---------- 3) Features DTW-alignées ----------
    leaders_and_lags = extract_leaders_and_lags(
        dtw_results=dtw_results,
        target_col=target_col,
        top_k=top_k_leaders,
        max_lag=max(horizons),
    )

    for leader, lag in leaders_and_lags:
        if leader not in ret.columns or leader not in px.columns:
            continue

        # returns alignés : le leader est observé en avance, donc on shift d'au moins lag+1
        # pour que la feature à t ne voie que l'info <= t-1.
        dtw_ret = ret[leader].shift(lag + 1)

        # momemta DTW
        for h in horizons:
            dm_name = f"{leader}_dtw_mom_{h}_lag{lag}"
            df[dm_name] = dtw_ret.rolling(h).sum()
            feature_cols.append(dm_name)

        # ratio target/leader aligné
        ratio_tl = np.log(px[target_col] / px[leader]).shift(lag + 1)
        for h in horizons:
            dr_name = f"ratio_{target_col}_{leader}_dtw_mom_{h}_lag{lag}"
            df[dr_name] = ratio_tl.rolling(h).sum()
            feature_cols.append(dr_name)

        # ---------- 4) Target = log-return courant de la target ----------
    y = ret[target_col]
    df[target_col] = y

    # ---------- 5) Nettoyage final ----------

    # On garde d'abord toutes les colonnes (features + target)
    df = df[feature_cols + [target_col]]

    # a) Drop des colonnes de features qui sont entièrement NaN
    nan_all_cols = [c for c in feature_cols if df[c].isna().all()]
    if nan_all_cols:
        logger.warning("Dropping all-NaN feature columns: %s", nan_all_cols)
        df = df.drop(columns=nan_all_cols)
        # on met à jour feature_cols pour refléter les colonnes restantes
        feature_cols = [c for c in feature_cols if c not in nan_all_cols]

    # b) Drop des lignes contenant au moins un NaN
    supervised = df[feature_cols + [target_col]].dropna()

    if supervised.empty:
        raise ValueError(
            "Supervised dataset is empty after feature construction and dropna(). "
            "Cela signifie que les features construites sont trop agressives "
            "(NaN partout) sur cette période. Réduis les horizons, vérifie les filtres, "
            "ou inspecte manuellement df.head()."
        )

    return supervised

