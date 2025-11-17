# tradingalgo.py

import logging

import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from srcgpt.data_loader_gpt import DataLoader, DataLoaderConfig
from srcgpt.preprocessing_gpt import Preprocessing
from srcgpt.eda_gpt import EDA
from srcgpt.LeadLagDTW_gpt import LeadLagDTW
from srcgpt.feature_engineering_gpt import (
    select_leaders_from_dtw,
    build_leadlag_supervised_dataset,
)
from srcgpt.ModelTuningValidation_gpt import ModelTuningValidation
from srcgpt.strategy_gpt import make_predictions_and_signals
from srcgpt.backtest_gpt import backtest_signals

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


TICKERS = ["GC=F", "SI=F", "EURUSD=X", "JPYUSD=X", "^TNX"]
START_DATE = "2022-01-01"
END_DATE = "2023-12-31"
TARGET_TICKER = "GC=F"


# ---------- DATASET BUILDING ----------

def build_dataset(
    tickers=TICKERS,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> dict[str, pd.DataFrame]:
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

    preproc = Preprocessing(imputed_data)
    cleaned = preproc.clean_data()
    features = preproc.transform_data(window_ma=10, scale_returns=True)

    # 🔹 Aplatir les colonnes (au cas où ce sont des tuples / MultiIndex)
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

    savgol_prices = preproc.apply_filter(filter_type="savgol", window=11, polyorder=3)
    savgol_prices = _flatten_cols(savgol_prices)

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
        "anomalies": anomalies,
    }


# ---------- EDA ----------

def run_eda_on_returns(returns: pd.DataFrame) -> None:
    logger.info("Running EDA on returns...")
    eda = EDA(returns)

    corr_results = eda.correlation_matrix(max_lag=5, show_plot=True)
    logger.info("Correlation matrix head:\n%s", corr_results["corr"].head())

    dist_df = eda.dtw_clustermap(show_plot=True)
    logger.info("DTW distance matrix head:\n%s", dist_df.head())

    eda.seasonality_tracker(period=252, show_plots=True)


def run_leadlag_analysis(returns: pd.DataFrame) -> pd.DataFrame:
    logger.info("Running Lead-Lag DTW analysis...")
    dtw_model = LeadLagDTW(returns)
    results = dtw_model.identify_lead_lag()
    logger.info("Lead-Lag DTW results:\n%s", results.head())
    return results


# ---------- SYSTEM HYPERPARAM OPTIM (n_lags / threshold / use_filtered) ----------

def optimize_system_hyperparams(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    filtered_prices: pd.DataFrame,
    target_col: str,
    leaders: list[str],
    n_trials: int = 30,
):
    """
    Optimise conjointement :
    - n_lags (features lead/lag),
    - threshold (pour les signaux),
    - use_filtered (True/False : ajouter ou non les prix filtrés comme features).

    Objectif : minimiser la MSE de prédiction (TimeSeriesSplit) avec un Ridge(alpha fixe).
    """

    def objective(trial):
        n_lags = trial.suggest_int("n_lags", 1, 10)
        threshold = trial.suggest_float("threshold", 0.0, 0.5)
        use_filtered = trial.suggest_categorical("use_filtered", [False, True])

        supervised = build_leadlag_supervised_dataset(
            prices=prices,
            returns=returns,
            target_col=target_col,
            leaders=leaders,
            n_lags=n_lags,
            use_filtered=use_filtered,
            filtered_prices=filtered_prices if use_filtered else None,
        )

        X = supervised.drop(columns=[target_col])
        y = supervised[target_col]

        model = Ridge(alpha=1.0)  # alpha fixe pour cette étape

        tscv = TimeSeriesSplit(n_splits=5)
        mses = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mses.append(mean_squared_error(y_test, preds))

        avg_mse = float(np.mean(mses))

        # On stocke threshold et use_filtered pour pouvoir les récupérer
        trial.set_user_attr("threshold", threshold)
        trial.set_user_attr("use_filtered", use_filtered)

        return avg_mse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_threshold = study.best_trial.user_attrs.get("threshold", 0.0)
    best_use_filtered = study.best_trial.user_attrs.get("use_filtered", False)

    best_params["threshold"] = best_threshold
    best_params["use_filtered"] = best_use_filtered

    return best_params


# ---------- MAIN PIPELINE ----------

def main():
    logger.info("Starting trading research pipeline...")

    # 1) Build dataset
    data_dict = build_dataset()
    prices = data_dict["prices"]
    returns = data_dict["scaled_returns"]  # on travaille sur les returns standardisés
    filtered_prices = data_dict["prices_savgol"]

    # 2) EDA
    run_eda_on_returns(returns)

    # 3) Lead-Lag DTW
    lead_lag_results = run_leadlag_analysis(returns)
    print("\n=== Lead-Lag relationships (top 10) ===")
    print(lead_lag_results.head(10))

    # 4) Sélection des leaders pour la target
    leaders = select_leaders_from_dtw(
        dtw_results=lead_lag_results,
        target_col=TARGET_TICKER,
        top_k=2,
    )
    logger.info("Selected leaders for %s: %s", TARGET_TICKER, leaders)

    # 5) Optimisation système : n_lags / threshold / use_filtered
    best_sys_params = optimize_system_hyperparams(
        prices=prices,
        returns=returns,
        filtered_prices=filtered_prices,
        target_col=TARGET_TICKER,
        leaders=leaders,
        n_trials=30,
    )
    logger.info("Best system hyperparameters: %s", best_sys_params)

    best_n_lags = best_sys_params["n_lags"]
    best_threshold = best_sys_params["threshold"]
    best_use_filtered = best_sys_params["use_filtered"]

    # 6) Construction dataset supervisé final (lead/lag + filtres)
    supervised = build_leadlag_supervised_dataset(
        prices=prices,
        returns=returns,
        target_col=TARGET_TICKER,
        leaders=leaders,
        n_lags=best_n_lags,
        use_filtered=best_use_filtered,
        filtered_prices=filtered_prices if best_use_filtered else None,
    )

    # 7) Tuning alpha avec ModelTuningValidation
    base_model = Ridge()
    param_space = {
        "alpha": ("float", 1e-4, 10.0),
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
    logger.info("Best model hyperparameters (alpha): %s", best_model_params)

    model = Ridge(alpha=best_model_params["alpha"])

     # 8) Prédictions + signaux (avec min_holding_days=2 pour limiter le churn)
    signals_df = make_predictions_and_signals(
        model=model,
        supervised_df=supervised,
        target_col=TARGET_TICKER,
        threshold=best_threshold,
        min_holding_days=2,
    )

    # 9) Backtest
    # target_ret = supervised[TARGET_TICKER]  # <- log-returns, à NE PAS utiliser direct
    log_ret = supervised[TARGET_TICKER]
    simple_ret = np.exp(log_ret) - 1.0      # conversion log -> simple returns

    equity_curve, stats = backtest_signals(
        target_returns=simple_ret,
        signals=signals_df["signal"],
        initial_capital=100_000.0,
        trading_cost_bps=5.0,
    )


if __name__ == "__main__":
    main()
