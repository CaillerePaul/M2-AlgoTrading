# tradingalgo.py
"""
Pipeline de recherche pour stratégie de trading macro :
- Télécharge ou charge des données de marché (Yahoo Finance ou CSV local)
- Prépare les données (returns, filtres, anomalies, DTW, features supervisées)
- Entraîne un modèle XGBoost (tuning via Optuna)
- Génère des signaux et backteste la stratégie
- Fournit une batterie de visualisations (EDA + EDA visuelle avancée)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from srcgpt.data_loader_gpt import DataLoader, DataLoaderConfig
from srcgpt.preprocessing_gpt import Preprocessing
from srcgpt.eda_gpt import EDA
from srcgpt.LeadLagDTW_gpt import LeadLagDTW
from srcgpt.feature_engineering_mkt_dtw_gpt import build_mkt_dtw_features_supervised
from srcgpt.model_tuning_gpt import ModelTuningValidation
from srcgpt.strategy_gpt import make_predictions_and_signals
from srcgpt.backtest_gpt import backtest_signals
from srcgpt.visual_eda_gpt import VisualEDA

# ---------------------------------------------------------------------------
# LOGGING GLOBAL
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIG GLOBALE
# ---------------------------------------------------------------------------

# Univers d’actifs
TICKERS = ["GC=F", "SI=F", "EURUSD=X", "JPYUSD=X", "^TNX"]

# Fenêtre temporelle
START_DATE = "2022-01-01"
END_DATE = "2023-12-31"

# Actif cible pour le modèle + backtest
TARGET_TICKER = "GC=F"

# Mode reproductible : utiliser un CSV local plutôt que yfinance
USE_LOCAL_CSV = True
CSV_PATH = "data/prices_raw.csv"

# EDA lourde (heatmaps + DTW + plein de graphes)
RUN_HEAVY_EDA = True


# ---------------------------------------------------------------------------
# UTILITAIRE : Charger les prix depuis un CSV local
# ---------------------------------------------------------------------------

def load_prices_from_csv(csv_path: Path, tickers: list[str]) -> pd.DataFrame:
    """
    Charge un fichier CSV de prix déjà téléchargés, du type :

        Price,GC=F,SI=F,EURUSD=X,JPYUSD=X,^TNX
        Ticker,GC=F,SI=F,EURUSD=X,JPYUSD=X,^TNX
        Date,,,,,
        2022-01-03,1799.4,...

    On :
    - renomme "Price" en "Date"
    - supprime les 3 premières lignes parasites
    - met "Date" en index datetime
    - garde uniquement les colonnes souhaitées (TICKERS)
    - convertit tout en float
    """
    logger.info("Loading local CSV data from %s", csv_path)

    df = pd.read_csv(csv_path)

    # 1) La colonne "Price" contient en fait la date
    df = df.rename(columns={"Price": "Date"})

    # 2) Supprimer les 3 premières lignes ("Ticker", "Date", etc.)
    df = df.iloc[3:].copy()

    # 3) Date -> datetime + index
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    # 4) Garder uniquement les tickers qui nous intéressent
    df = df[tickers]

    # 5) Convertir en float, forcer les éventuels strings en NaN
    df = df.apply(pd.to_numeric, errors="coerce")

    logger.info("Local prices loaded. Shape=%s", df.shape)
    return df


# ---------------------------------------------------------------------------
# DATASET BUILDING : prix, returns, filtres, anomalies
# ---------------------------------------------------------------------------

def build_dataset(
    tickers: list[str] = TICKERS,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    use_local_csv: bool = USE_LOCAL_CSV,
) -> dict[str, pd.DataFrame]:
    """
    Construit les datasets de base :
    - prices (clean),
    - returns (log-returns),
    - scaled_returns,
    - moving_avg,
    - prices_savgol / kalman / butter / ma / talib,
    - anomalies (Markov, Prophet optionnel).

    Deux modes :
    - use_local_csv = True  -> lit CSV local
    - use_local_csv = False -> utilise yfinance via DataLoader
    """

    # ---------- 1) Charger les prix bruts ----------

    if use_local_csv:
        # Mode reproductible : on lit directement le CSV déjà téléchargé
        prices_raw = load_prices_from_csv(CSV_PATH, tickers)
    else:
        # Mode “réel” : on télécharge avec yfinance via DataLoader
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

        # EDA des missing values
        loader.eda_missing(merged_data, show=False)
        loader.missing_test(merged_data)

        # Imputation simple (ffill + bfill)
        prices_raw = loader.impute_data(merged_data)

    # ---------- 2) Préprocessing de base (returns, scaling, etc.) ----------

    preproc = Preprocessing(prices_raw)
    cleaned = preproc.clean_data()  # re-ffill/bfill + sanity check

    features = preproc.transform_data(
        window_ma=10,       # window pour les moyennes mobiles de prix
        scale_returns=True  # StandardScaler sur les log-returns
    )

    # ---------- 3) Filtres de prix (smoothing) ----------

    # Savitzky-Golay
    prices_savgol = preproc.apply_filter(filter_type="savgol", window=11, polyorder=3)

    # Kalman
    prices_kalman = preproc.apply_filter(filter_type="kalman")

    # Butterworth (low-pass)
    prices_butter = preproc.apply_filter(filter_type="butterworth", cutoff=0.05, order=2)

    # Moyenne mobile simple
    prices_ma = preproc.apply_filter(filter_type="moving_average", window=10)

    # TA-Lib SMA
    prices_talib = preproc.apply_filter(filter_type="ta_lib", timeperiod=14)

    # ---------- 4) Aplatir les colonnes si MultiIndex ----------

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
    prices_ma = _flatten_cols(prices_ma)
    prices_talib = _flatten_cols(prices_talib)

    # ---------- 5) Détection de régimes / anomalies ----------

    anomalies = preproc.anomaly_detection(
        use_markov=True,
        use_prophet=False,   # Prophet un peu lourd, tu peux le remettre à True si tu veux
    )

    logger.info("Dataset built successfully.")

    return {
        "prices": cleaned,
        "returns": returns,
        "scaled_returns": scaled_returns,
        "moving_avg": features["moving_avg"],
        "prices_savgol": prices_savgol,
        "prices_kalman": prices_kalman,
        "prices_butter": prices_butter,
        "prices_ma10": prices_ma,
        "prices_talib": prices_talib,
        "anomalies": anomalies,
    }


# ---------------------------------------------------------------------------
# EDA de base sur les returns
# ---------------------------------------------------------------------------

def run_eda_on_returns(returns: pd.DataFrame) -> None:
    """
    EDA classique :
    - corrélations simples et décalées (lags)
    - distance DTW + clustermap
    - décomposition saisonnière éventuelle
    """
    logger.info("Running basic EDA on returns...")
    eda = EDA(returns)

    corr_results = eda.correlation_matrix(max_lag=5, show_plot=True)
    logger.info("Correlation matrix head:\n%s", corr_results["corr"].head())

    dist_df = eda.dtw_clustermap(show_plot=True)
    logger.info("DTW distance matrix head:\n%s", dist_df.head())

    eda.seasonality_tracker(period=252, show_plots=True)


# ---------------------------------------------------------------------------
# Analyse Lead-Lag DTW (pour les features)
# ---------------------------------------------------------------------------

def run_leadlag_analysis(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse Lead-Lag DTW (utilisée ensuite pour les features DTW-alignées).
    """
    logger.info("Running Lead-Lag DTW analysis...")
    dtw_model = LeadLagDTW(returns)
    results = dtw_model.identify_lead_lag()
    logger.info("Lead-Lag DTW results:\n%s", results.head())
    return results


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    logger.info("Starting trading research pipeline (XGBoost + filters + DTW)...")

    # 1) Dataset de base (prix, returns, filtres, anomalies)
    data_dict = build_dataset(use_local_csv=USE_LOCAL_CSV)

    prices = data_dict["prices"]
    returns = data_dict["returns"]          # log-returns
    scaled_returns = data_dict["scaled_returns"]
    prices_savgol = data_dict["prices_savgol"]
    prices_kalman = data_dict["prices_kalman"]
    prices_butter = data_dict["prices_butter"]
    prices_talib = data_dict["prices_talib"]
    anomalies = data_dict["anomalies"]

    # Dico de prix filtrés pour comparaisons interactives
    filtered_dict = {
        "savgol": prices_savgol,
        "kalman": prices_kalman,
        "butterworth": prices_butter,
        "talib_sma": prices_talib,
    }

    # 2) Visualisations ciblées (volatilité multi-window + comparaison filtres)
    #    → Plotly en mode "browser" pour éviter les pb de nbformat.
    veda_quick = VisualEDA(prices=prices, returns=returns)

    veda_quick.plot_rolling_volatility_interactive(
        asset=TARGET_TICKER,
        windows=(10, 20, 60, 120, 252),
        annualize=True,
        renderer="browser",
    )

    veda_quick.plot_filters_comparison_interactive(
        asset=TARGET_TICKER,
        filtered_prices=filtered_dict,
        normalize=True,
        renderer="browser",
    )

    # 3) EDA de base sur les returns standardisés
    run_eda_on_returns(scaled_returns)

    # 4) EDA visuelle avancée (optionnel)
    if RUN_HEAVY_EDA:
        veda = VisualEDA(prices=prices, returns=returns)

        veda.plot_rolling_correlation_heatmap(window=60, step=5)
        veda.plot_rolling_volatility_heatmap(window=60)
        veda.plot_returns_zscore_heatmap(window=60)
        veda.plot_correlation_clustermap()
        veda.plot_distance_clustermap()
        veda.plot_dtw_alignment("GC=F", "SI=F")
        veda.plot_rolling_dtw_distance_heatmap(window=90, step=10)
        veda.plot_macro_overlay_prices(["GC=F", "SI=F", "^TNX"], normalize=True)
        veda.plot_macro_overlay_returns(["GC=F", "SI=F"])
        veda.plot_overlay_spread("GC=F", "SI=F")
        veda.plot_returns_distribution("GC=F")
        veda.plot_distribution_evolution("GC=F", window=90)
        veda.plot_smoothed_returns(method="ewma", span=10)
        veda.plot_cumulated_returns()

    # 5) Lead-Lag DTW (pour construire les features)
    lead_lag_results = run_leadlag_analysis(scaled_returns)
    print("\n=== Lead-Lag relationships (top 10) ===")
    print(lead_lag_results.head(10))

    # 6) Construction du dataset supervisé (features marché + filtres + DTW)
    supervised = build_mkt_dtw_features_supervised(
        prices=prices,
        returns=returns,
        prices_savgol=prices_savgol,
        dtw_results=lead_lag_results,
        target_col=TARGET_TICKER,
        horizons=[1, 3, 5, 10, 21],
        top_k_leaders=3,
    )

    logger.info("Supervised dataset shape: %s", supervised.shape)

    # 7) Modèle XGBoost + tuning Optuna via ModelTuningValidation
    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    param_space = {
        "max_depth": ("int", 2, 6),
        "n_estimators": ("int", 100, 400),
        "learning_rate": ("float", 0.01, 0.2),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
        "reg_lambda": ("float", 0.0, 10.0),
    }

    validator = ModelTuningValidation(
        model=base_model,
        validation_data=supervised,
        target_col=TARGET_TICKER,
        n_splits=5,
    )

    best_model_params = validator.tune_model(
        n_trials=30,
        param_space=param_space,
    )
    logger.info("Best XGBoost hyperparameters: %s", best_model_params)

    # 8) Modèle final avec meilleurs hyperparamètres
    final_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **best_model_params,
    )

    # 9) Prédictions + signaux
    signals_df = make_predictions_and_signals(
        model=final_model,
        supervised_df=supervised,
        target_col=TARGET_TICKER,
        threshold=0.0,       # baseline : à tuner éventuellement
        min_holding_days=2,  # lissage minimal des positions
    )

    # 10) Backtest sur les returns de la target
    log_ret = supervised[TARGET_TICKER]      # log-returns de la target
    simple_ret = np.exp(log_ret) - 1.0       # simple returns

    equity_curve, stats = backtest_signals(
        target_returns=simple_ret,
        signals=signals_df["signal"],
        initial_capital=100_000.0,
        trading_cost_bps=5.0,
    )

    print("\n=== Backtest stats (XGBoost + filters + DTW) ===")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")

    logger.info("Pipeline finished.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
