# srcgpt/feature_engineering_mkt_gpt.py

from typing import List

import numpy as np
import pandas as pd


def build_mkt_features_supervised(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    target_col: str,
    horizons: List[int] | None = None,
) -> pd.DataFrame:
    """
    Construit un dataset supervisé pour prédire le log-return de target_col.

    Features :
    - Momenta de returns sur plusieurs horizons (rolling sum des log-returns),
    - Volatilités glissantes de la target,
    - Ratio GC/SI (log) et ses momemta si dispo.

    Attention :
    - On shift(1) les features pour ne pas utiliser l'info du jour t pour prédire r_t.
    - Target = log-return de la target au jour t.

    Paramètres
    ----------
    prices : DataFrame
        Prix (déjà nettoyés) avec colonnes = tickers.
    returns : DataFrame
        Log-returns, mêmes colonnes que prices.
    target_col : str
        Ticker cible, ex : "GC=F".
    horizons : liste d'int
        Horizons pour les indicateurs (ex: [1, 3, 5, 10, 21]).

    Retour
    ------
    supervised : DataFrame
        index = dates, colonnes = features + [target_col].
    """
    if horizons is None:
        horizons = [1, 3, 5, 10, 21]

    if target_col not in returns.columns:
        raise ValueError(f"Target column '{target_col}' not in returns.")

    # Alignement indices
    common_idx = prices.index.intersection(returns.index)
    px = prices.loc[common_idx]
    ret = returns.loc[common_idx]

    # DataFrame de base des features
    df = pd.DataFrame(index=common_idx)
    feature_cols: list[str] = []

    # 1) Momenta de returns (pour tous les actifs)
    for col in ret.columns:
        for h in horizons:
            col_name = f"{col}_mom_{h}"
            # somme des log-returns sur la fenêtre (approx log-momentum)
            df[col_name] = ret[col].rolling(h).sum().shift(1)
            feature_cols.append(col_name)

    # 2) Volatilités glissantes (target seulement, pour limiter la dimension)
    for h in horizons:
        vol_name = f"{target_col}_vol_{h}"
        df[vol_name] = ret[target_col].rolling(h).std().shift(1)
        feature_cols.append(vol_name)

    # 3) Ratio GC/SI si dispo
    if target_col in px.columns and "SI=F" in px.columns:
        ratio = np.log(px[target_col] / px["SI=F"])
        for h in horizons:
            rname = f"ratio_{target_col}_SI=F_mom_{h}"
            df[rname] = ratio.rolling(h).sum().shift(1)
            feature_cols.append(rname)

    # 4) Target = log-return courant de la target
    y = ret[target_col]

    # Assemblage final & drop des NaN
    df[target_col] = y
    supervised = df[feature_cols + [target_col]].dropna()

    return supervised
