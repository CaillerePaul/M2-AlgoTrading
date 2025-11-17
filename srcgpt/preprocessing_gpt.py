import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from scipy.signal import butter, filtfilt, savgol_filter
from pykalman import KalmanFilter
from prophet import Prophet
import talib

logger = logging.getLogger(__name__)


class Preprocessing:
    """
    Nettoyage, transformations (returns, moving average, scaling),
    filtres (Kalman, Butterworth, Savitzky-Golay, MA, TA-Lib), anomalies.
    """

    def __init__(self, data: pd.DataFrame):
        self.raw_data = data
        self.cleaned_data: pd.DataFrame | None = None
        self.returns: pd.DataFrame | None = None
        self.scaled_returns: pd.DataFrame | None = None
        self.moving_avg: pd.DataFrame | None = None
        self.scaler: StandardScaler | None = None

    # ---------- CLEANING ----------

    def clean_data(self) -> pd.DataFrame:
        logger.info("Cleaning data: imputing missing values (ffill + bfill)...")
        self.cleaned_data = self.raw_data.ffill().bfill()
        missing_after = self.cleaned_data.isnull().sum().sum()
        if missing_after > 0:
            logger.warning(
                "Warning: %d values remain missing after imputation.", missing_after
            )
        else:
            logger.info("Missing values successfully handled.")
        return self.cleaned_data

    # ---------- TRANSFORMATION ----------

    def transform_data(
        self,
        window_ma: int = 10,
        scale_returns: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        if self.cleaned_data is None:
            raise ValueError("You must call clean_data() before transform_data().")

        logger.info("Transforming data: log-returns, moving averages, scaling...")

        returns = np.log(self.cleaned_data / self.cleaned_data.shift(1)).dropna()
        moving_avg = self.cleaned_data.rolling(window=window_ma).mean().dropna()

        self.returns = returns
        self.moving_avg = moving_avg

        if scale_returns:
            self.scaler = StandardScaler()
            scaled_returns_array = self.scaler.fit_transform(returns)
            scaled_returns = pd.DataFrame(
                scaled_returns_array, index=returns.index, columns=returns.columns
            )
            self.scaled_returns = scaled_returns
        else:
            self.scaled_returns = returns

        return {
            "returns": self.returns,
            "scaled_returns": self.scaled_returns,
            "moving_avg": self.moving_avg,
        }

    # ---------- FILTERS ----------

    def apply_filter(self, filter_type: str = "moving_average", **kwargs) -> pd.DataFrame:
        if self.cleaned_data is None:
            raise ValueError("You must call clean_data() before apply_filter().")

        filters = {
            "kalman": self.apply_kalman_filter,
            "butterworth": self.apply_butterworth_filter,
            "savgol": self.apply_savgol_filter,
            "moving_average": self.apply_moving_average,
            "ta_lib": self.apply_ta_lib_filter,
        }

        if filter_type not in filters:
            raise ValueError(f"Unknown filter type: {filter_type}")

        logger.info("Applying filter: %s", filter_type)
        return filters[filter_type](**kwargs)

    def apply_kalman_filter(self) -> pd.DataFrame:
        logger.info("Applying Kalman filter...")
        filtered = pd.DataFrame(index=self.cleaned_data.index)

        for col in self.cleaned_data.columns:
            kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
            series = self.cleaned_data[col].values
            mask = ~np.isnan(series)
            data = series[mask]

            if len(data) == 0:
                continue

            state_means, _ = kf.em(data, n_iter=5).filter(data)
            full_series = pd.Series(
                state_means.flatten(), index=self.cleaned_data.index[mask]
            )
            filtered[col] = full_series

        return filtered

    def apply_butterworth_filter(self, cutoff: float = 0.05, order: int = 2) -> pd.DataFrame:
        logger.info("Applying Butterworth filter...")
        b, a = butter(order, cutoff, btype="low", analog=False)

        def _filter(col: pd.Series) -> pd.Series:
            if col.isnull().any():
                col = col.ffill().bfill()
            return pd.Series(filtfilt(b, a, col.values), index=col.index)

        return self.cleaned_data.apply(_filter)

    def apply_savgol_filter(self, window: int = 7, polyorder: int = 2) -> pd.DataFrame:
        logger.info("Applying Savitzky-Golay filter...")
        if window % 2 == 0:
            raise ValueError("window length must be odd for Savitzky-Golay filter.")

        def _filter(col: pd.Series) -> pd.Series:
            if col.isnull().any():
                col = col.ffill().bfill()
            return pd.Series(
                savgol_filter(col.values, window_length=window, polyorder=polyorder),
                index=col.index,
            )

        return self.cleaned_data.apply(_filter)

    def apply_moving_average(self, window: int = 10) -> pd.DataFrame:
        logger.info("Applying Moving Average filter...")
        return self.cleaned_data.rolling(window=window).mean()

    def apply_ta_lib_filter(self, timeperiod: int = 10) -> pd.DataFrame:
        logger.info("Applying TA-Lib SMA filter...")
        ta_filtered = pd.DataFrame(index=self.cleaned_data.index)
        for col in self.cleaned_data.columns:
            ta_filtered[col] = talib.SMA(self.cleaned_data[col].values, timeperiod=timeperiod)
        return ta_filtered

    # ---------- ANOMALY / REGIME DETECTION ----------

    def anomaly_detection(
        self,
        use_markov: bool = True,
        use_prophet: bool = True,
    ) -> pd.DataFrame:
        if self.cleaned_data is None:
            raise ValueError("You must call clean_data() before anomaly_detection().")

        logger.info("Detecting anomalies using Markov Switching and Prophet models...")
        anomalies: dict[str, pd.Series] = {}

        if use_markov:
            for col in self.cleaned_data.columns:
                col_name = str(col)
                logger.info("Markov model for %s", col_name)

                try:
                    model = MarkovRegression(
                        self.cleaned_data[col].dropna(),
                        k_regimes=2,
                        trend="c",
                        switching_variance=True,
                    )
                    res = model.fit(disp=False)
                    regimes = res.smoothed_marginal_probabilities[1]
                    regimes = regimes.reindex(self.cleaned_data.index, method="nearest")
                    anomalies[f"{col_name}_regime"] = regimes
                except Exception as e:
                    logger.warning("Could not fit Markov model for %s: %s", col_name, e)

        if use_prophet:
            for col in self.cleaned_data.columns:
                col_name = str(col)
                logger.info("Prophet anomaly detection for %s", col_name)

                df = pd.DataFrame(
                    {
                        "ds": self.cleaned_data.index,
                        "y": self.cleaned_data[col].ffill().bfill(),
                    }
                )

                try:
                    m = Prophet(daily_seasonality=True)
                    m.fit(df)
                    future = m.make_future_dataframe(periods=0)
                    forecast = m.predict(future)
                    yhat = forecast["yhat"]
                    residuals = df["y"] - yhat
                    threshold = 2 * np.std(residuals)
                    outliers = (residuals.abs() > threshold).reindex(
                        self.cleaned_data.index
                    )
                    anomalies[f"{col_name}_prophet_outlier"] = outliers
                except Exception as e:
                    logger.warning("Prophet failed for %s: %s", col_name, e)

        return pd.DataFrame(anomalies, index=self.cleaned_data.index)
