# tradingalgo.py

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

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


TICKERS = ["GC=F", "SI=F", "EURUSD=X", "JPYUSD=X", "^TNX"]
START_DATE = "2022-01-01"
END_DATE = "2023-12-31"
TARGET_TICKER = "GC=F"

RUN_HEAVY_EDA = True  # <<< mets True si tu veux lancer les 20 graphes

USE_LOCAL_CSV = True
CSV_PATH = "data/prices_raw.csv"

# ---------- DATASET BUILDING ----------

# ---------- DATASET BUILDING ----------

def build_dataset(
    tickers=TICKERS,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    use_local_csv: bool = True,
    csv_path: str = CSV_PATH,
) -> dict[str, pd.DataFrame]:
    """
    Construit les datasets de base :
    - prices (clean),
    - returns (log-returns),
    - scaled_returns,
    - moving_avg,
    - prices_savgol (filtrés Savitzky-Golay),
    - anomalies (Markov, Prophet optionnel).

    Si use_cached_csv=True, lit les données déjà téléchargées (CSV).
    Sinon, télécharge via yfinance (DataLoader).
    """

    # ---------- 1) Charger les prix bruts ----------
    if use_local_csv:
        # Lecture du CSV tel que tu l’as montré
        df = pd.read_csv(csv_path)

        # La première colonne s'appelle "Price" et contient en fait la date
        # + deux lignes parasites "Ticker" et "Date"
        df = df.rename(columns={"Price": "Date"})
        # On supprime les lignes parasites
        df = df[df["Date"].notna()]
        df = df[~df["Date"].isin(["Ticker", "Date"])]

        # Conversion en datetime + index
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

        # On garde seulement les tickers que tu utilises
        merged_data = df[tickers].astype(float)

        logger.info("Loaded prices from CSV %s, shape=%s", csv_path, merged_data.shape)

        # imputation simple (comme le DataLoader)
        imputed_data = merged_data.ffill().bfill()

    else:
        # --- Version "réaliste" avec yfinance / DataLoader ---
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

        loader.eda_missing(merged_data, show=False)
        loader.missing_test(merged_data)

        imputed_data = loader.impute_data(merged_data)

    # ---------- 2) Préprocessing / features de base ----------
    preproc = Preprocessing(imputed_data)
    cleaned = preproc.clean_data()
    features = preproc.transform_data(window_ma=10, scale_returns=True)

    # Filtres Savitzky-Golay (prix filtrés)
    savgol_prices = preproc.apply_filter(filter_type="savgol", window=11, polyorder=3)
    kalman_prices = preproc.apply_filter(filter_type="kalman")
    butter_prices = preproc.apply_filter(filter_type="butterworth", cutoff=0.05, order=2)
    ma_prices = preproc.apply_filter(filter_type="moving_average", window=10)
    ta_prices = preproc.apply_filter(filter_type="ta_lib", timeperiod=14)

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
    savgol_prices = _flatten_cols(savgol_prices)
    kalman_prices = _flatten_cols(kalman_prices)
    butter_prices = _flatten_cols(butter_prices)
    ma_prices = _flatten_cols(ma_prices)
    ta_prices = _flatten_cols(ta_prices)

    anomalies = preproc.anomaly_detection(
        use_markov=True,
        use_prophet=False,
    )

    logger.info("Dataset built successfully.")

    return {
        "prices": cleaned,
        "returns": returns,
        "scaled_returns": scaled_returns,
        "moving_avg": features["moving_avg"],
        "prices_savgol": savgol_prices,
        "prices_kalman": kalman_prices,
        "prices_butter": butter_prices,
        "prices_ma10": ma_prices,
        "prices_talib": ta_prices,
        "anomalies": anomalies,
    }


# ---------- EDA (optionnel) ----------

def run_eda_on_returns(returns: pd.DataFrame) -> None:
    logger.info("Running EDA on returns...")
    eda = EDA(returns)

    corr_results = eda.correlation_matrix(max_lag=5, show_plot=True)
    logger.info("Correlation matrix head:\n%s", corr_results["corr"].head())

    dist_df = eda.dtw_clustermap(show_plot=True)
    logger.info("DTW distance matrix head:\n%s", dist_df.head())

    eda.seasonality_tracker(period=252, show_plots=True)


def run_leadlag_analysis(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse Lead-Lag DTW (utilisée ensuite pour les features DTW-alignées).
    """
    logger.info("Running Lead-Lag DTW analysis...")
    dtw_model = LeadLagDTW(returns)
    results = dtw_model.identify_lead_lag()
    logger.info("Lead-Lag DTW results:\n%s", results.head())
    return results




from srcgpt.visual_eda_gpt import VisualEDA, VisualEDAConfig



# ---------- MAIN PIPELINE (XGBoost + filtres + DTW) ----------

def main():
    logger.info("Starting trading research pipeline (XGBoost + filters + DTW)...")

    # 1) Dataset de base
    data_dict = build_dataset(use_local_csv=USE_LOCAL_CSV)

    prices = data_dict["prices"]
    returns = data_dict["returns"]          # log-returns
    scaled_returns = data_dict["scaled_returns"]
    prices_savgol = data_dict["prices_savgol"]
    prices_kalman = data_dict["prices_kalman"]
    prices_butter = data_dict["prices_butter"]
    prices_talib = data_dict["prices_talib"]
    anomalies = data_dict["anomalies"]

    filtered_dict = {
    "savgol": prices_savgol,
    "kalman": prices_kalman,
    "butterworth": prices_butter,
    "talib_sma": prices_talib,
    }

    # Volatilité glissante interactive multi-window
    VisualEDA(prices=prices, returns=returns).plot_rolling_volatility_interactive(
    asset="GC=F",
    windows=(10, 20, 60, 120, 252),
    annualize=True,
    renderer="browser",  # évite ton erreur nbformat
    )


    VisualEDA(prices=prices, returns=returns).plot_filters_comparison_interactive(
    asset="GC=F",
    filtered_prices=filtered_dict,
    normalize=True,
    renderer="browser",  # safe pour ton Jupyter
    )  
    # 2) EDA (light)

    run_eda = EDA(scaled_returns)
    run_eda.correlation_matrix(max_lag=5, show_plot=True)
    run_eda.dtw_clustermap(show_plot=True)

    # 3) (optionnel) EDA visuelle avancée
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

    VisualEDA(prices=prices_savgol, returns=returns).plot_cumulated_returns()
    VisualEDA(prices=prices_kalman, returns=returns).plot_cumulated_returns()


    # 4) Lead-Lag DTW
    lead_lag_results = run_leadlag_analysis(scaled_returns)
    print("\n=== Lead-Lag relationships (top 10) ===")
    print(lead_lag_results.head(10))

    # 5) Features supervisées marché + filtres + DTW
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

    # 5) Modèle XGBoost + tuning via ModelTuningValidation
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

    # Modèle final avec les meilleurs hyperparamètres
    final_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **best_model_params,
    )

    # 6) Prédictions + signaux
    signals_df = make_predictions_and_signals(
        model=final_model,
        supervised_df=supervised,
        target_col=TARGET_TICKER,
        threshold=0.0,       # baseline : on pourra tuner ce seuil ensuite
        min_holding_days=2,  # lissage minimal des positions
    )

    # 7) Backtest : conversion log-returns -> simple returns
    log_ret = supervised[TARGET_TICKER]          # log-returns de la target
    simple_ret = np.exp(log_ret) - 1.0          # simple returns

    equity_curve, stats = backtest_signals(
        target_returns=simple_ret,
        signals=signals_df["signal"],
        initial_capital=100_000.0,
        trading_cost_bps=5.0,
    )

    print("\n=== Backtest stats (XGBoost + filters + DTW) ===")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")

    # Optionnel : visualiser la courbe d'equity dans un notebook
    # equity_curve["equity"].plot(figsize=(10, 4), title="Strategy Equity Curve")

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
