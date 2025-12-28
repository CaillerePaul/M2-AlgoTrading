# tradingalgo.py

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from srcgpt.data_loader_gpt import DataLoader, DataLoaderConfig
from srcgpt.preprocessing_gpt import Preprocessing
from srcgpt.eda_gpt import EDA
from srcgpt.LeadLagDTW_gpt import LeadLagDTW
from srcgpt.feature_engineering_mkt_dtw_gpt import build_mkt_dtw_features_supervised
from srcgpt.model_tuning_gpt import ModelTuningValidation
from srcgpt.backtest_gpt import backtest_signals
from srcgpt.visual_eda_gpt import VisualEDA

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Paramètres globaux
# ---------------------------------------------------------

TICKERS: List[str] = ["GC=F", "SI=F", "EURUSD=X", "JPYUSD=X", "^TNX"]
START_DATE = "2022-01-01"
END_DATE = "2023-12-31"

# On peut vouloir tester plusieurs cibles (ex: GC=F, SI=F)
TARGET_TICKERS: List[str] = ["GC=F", "SI=F"]

# Où lire les données déjà téléchargées
USE_LOCAL_CSV = True
CSV_PATH = "data/prices_raw.csv"

# EDA lourde optionnelle
RUN_HEAVY_EDA = False

# Config modèles
MODEL_CONFIGS = {
    "xgboost": {
        "model_cls": XGBRegressor,
        "base_params": {
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        },
        "param_space": {
            "max_depth": ("int", 2, 6),
            "n_estimators": ("int", 100, 400),
            "learning_rate": ("float", 0.01, 0.2),
            "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.6, 1.0),
            "reg_lambda": ("float", 0.0, 10.0),
        },
        "n_trials": 30,
        "n_splits_walk": 4,
    },
    "ridge": {
        "model_cls": Ridge,
        "base_params": {
            "random_state": 42,
            "fit_intercept": True,
        },
        "param_space": {
            # alpha > 0, en log, pour chercher large
            "alpha": ("logfloat", 1e-3, 100.0),
        },
        "n_trials": 30,
        "n_splits_walk": 4,
    },
    "elasticnet": {
        "model_cls": ElasticNet,
        "base_params": {
            "fit_intercept": True,
            "random_state": 42,
            "max_iter": 10_000,
        },
        "param_space": {
            # force du shrinkage
            "alpha": ("logfloat", 1e-4, 10.0),
            # mélange L1 / L2 : 0 = Ridge, 1 = Lasso
            "l1_ratio": ("float", 0.0, 1.0),
        },
        "n_trials": 30,
        "n_splits_walk": 4,
    },
}


# ---------------------------------------------------------
# 1) Construction du dataset (prix, returns, filtres, anomalies)
# ---------------------------------------------------------

def build_dataset(
    tickers: List[str] = TICKERS,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    use_local_csv: bool = True,
    csv_path: str = CSV_PATH,
) -> Dict[str, pd.DataFrame]:
    """
    Construit les datasets de base :
      - prices : prix nettoyés
      - returns : log-returns
      - scaled_returns : log-returns standardisés
      - moving_avg : moyennes mobiles
      - prices_savgol, prices_kalman, prices_butter, prices_ma10, prices_talib
      - anomalies : régimes Markov / anomalies Prophet (ici Prophet désactivé)
    """
    # ---------- 1) Charger les prix bruts ----------
    if use_local_csv:
        logger.info("Loading local CSV: %s", csv_path)
        df = pd.read_csv(csv_path)

        # Ton CSV original :
        # Price,GC=F,SI=F,EURUSD=X,JPYUSD=X,^TNX
        # Ticker,GC=F,...
        # Date,,,,,
        # 2022-01-03,...
        # -> on renomme "Price" en "Date", on parse, on enlève les lignes non-datées.
        if "Price" in df.columns:
            df = df.rename(columns={"Price": "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.set_index("Date")
        # On sélectionne explicitement les colonnes des tickers
        prices_raw = df[tickers].astype(float)
    else:
        logger.info("Fetching data from Yahoo Finance...")
        loader_config = DataLoaderConfig(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            price_column="Adj Close",
            resample_rule=None,
        )
        loader = DataLoader(loader_config)

        raw_data_dict = loader.fetch_data()
        merged_data = loader.merge_data(raw_data_dict)

        # EDA sur manquants (plots désactivés par défaut pour ne pas polluer)
        loader.eda_missing(merged_data, show=False)
        loader.missing_test(merged_data)

        prices_raw = loader.impute_data(merged_data)

    # ---------- 2) Pré-traitement : imputations, returns, filtres ----------
    preproc = Preprocessing(prices_raw)
    cleaned = preproc.clean_data()  # ffill + bfill final
    features = preproc.transform_data(window_ma=10, scale_returns=True)

    # Filtres sur prix
    prices_savgol = preproc.apply_filter(filter_type="savgol", window=11, polyorder=3)
    prices_kalman = preproc.apply_filter(filter_type="kalman")
    prices_butter = preproc.apply_filter(filter_type="butterworth", cutoff=0.05, order=2)
    prices_ma10 = preproc.apply_filter(filter_type="moving_average", window=10)
    prices_talib = preproc.apply_filter(filter_type="ta_lib", timeperiod=14)

    def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [
            c[0] if isinstance(c, tuple) and len(c) > 0 else c
            for c in df.columns
        ]
        return df

    cleaned = _flatten_cols(cleaned)
    returns = _flatten_cols(features["returns"])
    scaled_returns = _flatten_cols(features["scaled_returns"])
    prices_savgol = _flatten_cols(prices_savgol)
    prices_kalman = _flatten_cols(prices_kalman)
    prices_butter = _flatten_cols(prices_butter)
    prices_ma10 = _flatten_cols(prices_ma10)
    prices_talib = _flatten_cols(prices_talib)

    anomalies = preproc.anomaly_detection(
        use_markov=True,
        use_prophet=False,
    )

    logger.info("Dataset built successfully. prices shape=%s", cleaned.shape)

    return {
        "prices": cleaned,
        "returns": returns,
        "scaled_returns": scaled_returns,
        "moving_avg": features["moving_avg"],
        "prices_savgol": prices_savgol,
        "prices_kalman": prices_kalman,
        "prices_butter": prices_butter,
        "prices_ma10": prices_ma10,
        "prices_talib": prices_talib,
        "anomalies": anomalies,
    }


# ---------------------------------------------------------
# 2) EDA basique et Lead-Lag DTW
# ---------------------------------------------------------

def run_eda_on_returns(returns: pd.DataFrame) -> None:
    """
    EDA simple : corrélations et DTW clustermap sur les returns.
    """
    logger.info("Running EDA on returns...")
    eda = EDA(returns)

    corr_results = eda.correlation_matrix(max_lag=5, show_plot=True)
    logger.info("Correlation matrix head:\n%s", corr_results["corr"].head())

    dist_df = eda.dtw_clustermap(show_plot=True)
    logger.info("DTW distance matrix head:\n%s", dist_df.head())

    eda.seasonality_tracker(period=252, show_plots=True)


def run_leadlag_analysis(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse Lead-Lag DTW (utilisée ensuite pour les features alignées).
    """
    logger.info("Running Lead-Lag DTW analysis...")
    dtw_model = LeadLagDTW(returns)
    results = dtw_model.identify_lead_lag()
    logger.info("Lead-Lag DTW results:\n%s", results.head())
    return results


def get_smoothed_prices(data_dict: Dict[str, pd.DataFrame], smoothing_name: str) -> pd.DataFrame:
    """
    Récupère les prix lissés selon le smoothing choisi.
    """
    if smoothing_name == "raw":
        return data_dict["prices"]
    elif smoothing_name == "savgol":
        return data_dict["prices_savgol"]
    elif smoothing_name == "kalman":
        return data_dict["prices_kalman"]
    elif smoothing_name == "butterworth":
        return data_dict["prices_butter"]
    elif smoothing_name == "ma10":
        return data_dict["prices_ma10"]
    elif smoothing_name == "talib":
        return data_dict["prices_talib"]
    else:
        raise ValueError(f"Unknown smoothing_name: {smoothing_name}")


# ---------------------------------------------------------
# 3) Génération de signaux à partir des prédictions
# ---------------------------------------------------------

def generate_signals_from_preds(
    preds: pd.Series,
    threshold: float = 0.0,
    min_holding_days: int = 2,
) -> pd.Series:
    """
    À partir des prédictions (retours futurs attendus), génère un signal brut
    puis applique une contrainte de "min_holding_days" pour éviter de changer
    de position tous les jours.

    Convention simple :
      - pred > threshold  -> +1 (long)
      - pred < -threshold -> -1 (short)
      - sinon             -> 0 (flat)

    La contrainte min_holding_days s'applique sur les changements de régime.
    """
    preds = preds.copy()
    preds = preds.dropna()

    raw = np.where(preds > threshold, 1, np.where(preds < -threshold, -1, 0))
    raw = pd.Series(raw, index=preds.index)

    signals = pd.Series(index=preds.index, dtype=float)
    current = 0
    hold_len = 0

    for dt in preds.index:
        desired = raw.loc[dt]

        if current == 0:
            # on peut immédiatement ouvrir une position
            current = desired
            hold_len = 1 if current != 0 else 0
        else:
            hold_len += 1
            # on ne permet le changement que si min_holding_days est respecté
            if desired != current and hold_len >= min_holding_days:
                current = desired
                hold_len = 1 if current != 0 else 0
            # sinon on garde current

        signals.loc[dt] = current

    return signals


# ---------------------------------------------------------
# 4) Walk-forward backtest (vrai découpage temporel)
# ---------------------------------------------------------

def walk_forward_backtest(
    supervised: pd.DataFrame,
    target_col: str,
    model_cls,
    model_params: Dict,
    n_splits: int = 4,
    threshold: float = 0.0,
    min_holding_days: int = 2,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.Series, pd.Series]:
    """
    Backtest en walk-forward :

    - On découpe la série en n_splits fenêtres temporelles via TimeSeriesSplit.
    - Pour chaque split :
        * train = toutes les dates jusqu'à t_train_end
        * test  = la fenêtre qui suit
        * on entraîne le modèle sur train puis on prédit sur test
    - On agrège toutes les prédictions de test, on génère des signaux,
      puis on lance backtest_signals sur l'ensemble.

    Retourne :
      - equity_curve : DataFrame equity dans le temps
      - stats        : dict des métriques backtest
      - preds        : Series de prédictions (log-returns attendus)
      - signals      : Series de signaux (+1/-1/0)
    """
    supervised = supervised.sort_index()
    X = supervised.drop(columns=[target_col])
    y = supervised[target_col]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    preds = pd.Series(index=supervised.index, dtype=float)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        model = model_cls(**model_params)
        model.fit(X_train.values, y_train.values)

        fold_preds = model.predict(X_test.values)
        preds.iloc[test_idx] = fold_preds

        logger.info(
            "Walk-forward fold %d/%d : train=%d pts, test=%d pts",
            fold, n_splits, len(train_idx), len(test_idx),
        )

    # Signaux walk-forward (une prédiction uniquement là où on a du test)
    signals = generate_signals_from_preds(
        preds=preds,
        threshold=threshold,
        min_holding_days=min_holding_days,
    )

    # Log-returns -> simple returns
    simple_ret = np.exp(y) - 1.0

    # Remplir les trous éventuels (dates jamais en test -> on reste flat)
    signals_full = signals.reindex(simple_ret.index).fillna(0.0)

    equity_curve, stats = backtest_signals(
        target_returns=simple_ret,
        signals=signals_full,
        initial_capital=100_000.0,
        trading_cost_bps=5.0,
    )

    return equity_curve, stats, preds, signals_full


# ---------------------------------------------------------
# 5) Une expérience : (target, smoothing, modèle)
# ---------------------------------------------------------

def run_single_experiment(
    data_dict: Dict[str, pd.DataFrame],
    dtw_results: pd.DataFrame,
    target_ticker: str,
    smoothing_name: str,
    model_name: str,
) -> Dict[str, float]:
    """
    Lance une expérience complète :

      1) Choisir la cible (target_ticker) et les prix lissés (smoothing_name)
      2) Construire le dataset supervisé (features marché + DTW + filtres)
      3) Tuner les hyperparamètres (TimeSeriesSplit) sur l'ensemble supervisé
      4) Backtest en walk-forward sur toute la période 2022-2023

    Retourne un dict avec (model, target, smoothing, + stats backtest).
    """
    prices = data_dict["prices"]
    returns = data_dict["returns"]

    prices_smooth = get_smoothed_prices(data_dict, smoothing_name)

    # ---------- 1) Dataset supervisé ----------
    supervised = build_mkt_dtw_features_supervised(
        prices=prices,
        returns=returns,
        prices_smooth=prices_smooth,  # (assume signature mise à jour)
        dtw_results=dtw_results,
        target_col=target_ticker,
        horizons=[1, 3, 5, 10, 21],
        top_k_leaders=3,
    )

    logger.info(
        "[%s | %s | %s] supervised shape=%s",
        model_name, target_ticker, smoothing_name, supervised.shape
    )

    # ---------- 2) Tuning des hyperparamètres ----------
    cfg = MODEL_CONFIGS[model_name]

    base_model = cfg["model_cls"](**cfg["base_params"])

    validator = ModelTuningValidation(
        model=base_model,
        validation_data=supervised,
        target_col=target_ticker,
        n_splits=5,
    )

    best_params = validator.tune_model(
        n_trials=cfg["n_trials"],
        param_space=cfg["param_space"],
    )

    # Merge base_params + best_params
    model_params = {**cfg["base_params"], **best_params}
    logger.info(
        "Best params for [%s | %s | %s] : %s",
        model_name, target_ticker, smoothing_name, model_params
    )

    # ---------- 3) Walk-forward backtest ----------
    equity_curve, stats, preds, signals = walk_forward_backtest(
        supervised=supervised,
        target_col=target_ticker,
        model_cls=cfg["model_cls"],
        model_params=model_params,
        n_splits=cfg["n_splits_walk"],
        threshold=0.0,
        min_holding_days=2,
    )

    # Petit résumé console
    print(f"\n=== Backtest stats [{model_name} | {target_ticker} | {smoothing_name}] ===")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")

    # On renvoie un dict plat : idéal pour mettre en DataFrame
    out = {
        "model": model_name,
        "target": target_ticker,
        "smoothing": smoothing_name,
    }
    out.update(stats)
    return out


# ---------------------------------------------------------
# 6) MAIN : boucle sur (target, smoothing, modèle)
# ---------------------------------------------------------

def main():
    logger.info("Starting trading research pipeline (walk-forward)...")

    # 1) Dataset de base
    data_dict = build_dataset(use_local_csv=USE_LOCAL_CSV)
    prices = data_dict["prices"]
    returns = data_dict["returns"]
    scaled_returns = data_dict["scaled_returns"]

    # 2) EDA légère (optionnelle)
    #    -> plots corr, dtw clustermap, saisonnalité
    #    -> tu peux mettre False si ça gêne
    if RUN_HEAVY_EDA:
        run_eda_on_returns(scaled_returns)

        veda = VisualEDA(prices=prices, returns=returns)
        veda.plot_macro_overlay_prices(["GC=F", "SI=F", "^TNX"], normalize=True)
        veda.plot_returns_zscore_heatmap(window=60)
        veda.plot_cumulated_returns()

    # 3) Lead-Lag DTW sur returns standardisés (partagé entre toutes les expériences)
    dtw_results = run_leadlag_analysis(scaled_returns)
    print("\n=== Lead-Lag relationships (top 10) ===")
    print(dtw_results.head(10))

    # 4) Boucle d'expériences
    all_results: List[Dict[str, float]] = []

    smoothing_list = ["raw", "savgol", "kalman", "butterworth", "ma10"]
    model_list = ["xgboost", "ridge", "elasticnet"]

    for target_ticker in TARGET_TICKERS:
        for smoothing_name in smoothing_list:
            for model_name in model_list:
                res = run_single_experiment(
                    data_dict=data_dict,
                    dtw_results=dtw_results,
                    target_ticker=target_ticker,
                    smoothing_name=smoothing_name,
                    model_name=model_name,
                )
                all_results.append(res)

    # 5) Résumé global en DataFrame
    results_df = pd.DataFrame(all_results)
    print("\n=== Summary of all experiments ===")
    print(results_df)
    
    results_df.to_pickle("all_results.pkl")

    logger.info("All experiments finished.")


if __name__ == "__main__":
    main()


import pandas as pd

pd.set_option("display.max_rows", None)   # ← affiche toutes les lignes
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

res = pd.read_pickle("all_results.pkl")
print(res)