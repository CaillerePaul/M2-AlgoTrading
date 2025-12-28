import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from scipy.signal import butter, lfilter  # <- filtfilt remplacé par lfilter (causal)
from pykalman import KalmanFilter
from prophet import Prophet
import talib

logger = logging.getLogger(__name__)


class Preprocessing:
    """
    Pipeline preprocessing :

    1) Cleaning
       - Imputation simple des trous (ffill + bfill)
       - Optionnel : impose une fréquence calendrier (ex: Business Days)

    2) Transformations
       - Log-returns (stationnaires)
       - Moving average sur prix (tendance lente)
       - Standard scaling des returns (z-score global)

    3) Filtres / smoothing sur PRIX (tous pensés pour être causaux)
       - Kalman filter (état latent estimé au fil du temps)
       - Butterworth passe-bas (version lfilter -> ne regarde que le passé)
       - Savitzky-Golay *causal* (fenêtre glissante vers le passé)
       - Moving Average (baseline)
       - TA-Lib SMA (équivalent MA)

    4) Détection de régimes / anomalies
       - Markov Switching : détecte 2 régimes (calme/stress) sur RETURNS
       - Prophet : détecte outliers via résidus d'un modèle trend+saison
    """

    def __init__(self, data: pd.DataFrame):
        self.raw_data = data
        self.cleaned_data: pd.DataFrame | None = None

        # Outputs des transformations
        self.returns: pd.DataFrame | None = None
        self.scaled_returns: pd.DataFrame | None = None
        self.moving_avg: pd.DataFrame | None = None

        self.scaler: StandardScaler | None = None

    # ---------- CLEANING ----------

    def clean_data(self, set_freq: Optional[str] = "B") -> pd.DataFrame:
        """
        1) ffill/bfill pour combler les trous de calendrier.
        2) optionnel : impose une fréquence régulière (par ex. jours ouvrés).

        set_freq:
          - "B" : jours ouvrés
          - "D" : calendrier quotidien
          - None : ne force rien
        """

        logger.info("Cleaning data: imputing missing values (ffill + bfill)...")

        # Imputation simple
        self.cleaned_data = self.raw_data.ffill().bfill()

        # Optionnel : forcer un calendrier régulier
        if set_freq is not None:
            try:
                self.cleaned_data = (
                    self.cleaned_data
                    .asfreq(set_freq)  # impose la fréquence
                    .ffill()
                    .bfill()
                )
            except Exception as e:
                logger.warning("Could not set frequency %s: %s", set_freq, e)

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
        """
        Construit :
        - log-returns
        - moving average des prix
        - scaled returns (si scale_returns=True)
        """
        if self.cleaned_data is None:
            raise ValueError("You must call clean_data() before transform_data().")

        logger.info("Transforming data: log-returns, moving averages, scaling...")

        prices = self.cleaned_data.astype(float)

        # log-returns (stationnaires)
        returns = np.log(prices / prices.shift(1)).dropna()

        # moving average sur PRIX (tendance lente)
        moving_avg = prices.rolling(window=window_ma).mean().dropna()

        self.returns = returns
        self.moving_avg = moving_avg

        # scaling global des returns (z-score)
        if scale_returns:
            self.scaler = StandardScaler()
            scaled_returns_array = self.scaler.fit_transform(returns.values)
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
        """
        Applique un smoothing sur les prix de cleaned_data.

        Tous les filtres sont pensés ici pour être utilisables en backtest
        sans fuite d'information (causaux ou au moins utilisables via shift).
        """
        if self.cleaned_data is None:
            raise ValueError("You must call clean_data() before apply_filter().")

        filters = {
            "kalman": self.apply_kalman_filter,
            "butterworth": self.apply_butterworth_filter,
            "savgol": self.apply_savgol_filter_causal,
            "moving_average": self.apply_moving_average,
            "ta_lib": self.apply_ta_lib_filter,
        }

        if filter_type not in filters:
            raise ValueError(f"Unknown filter type: {filter_type}")

        logger.info("Applying filter: %s", filter_type)
        return filters[filter_type](**kwargs)

    def apply_kalman_filter(self) -> pd.DataFrame:
        """
        Kalman smoothing : estime une tendance "cachée" sous le bruit.
        Filtre naturellement causal (l'état latent à t ne dépend que du passé).
        """
        logger.info("Applying Kalman filter...")
        filtered = pd.DataFrame(index=self.cleaned_data.index)

        for col in self.cleaned_data.columns:
            series = self.cleaned_data[col].astype(float).values

            mask = ~np.isnan(series)
            data = series[mask]
            if len(data) == 0:
                logger.warning("Kalman skipped empty series: %s", col)
                continue

            # On initialise autour de la première observation
            kf = KalmanFilter(initial_state_mean=data[0], n_dim_obs=1)
            state_means, _ = kf.em(data, n_iter=5).filter(data)

            full_series = pd.Series(np.nan, index=self.cleaned_data.index)
            full_series.loc[self.cleaned_data.index[mask]] = state_means.flatten()

            filtered[col] = full_series.ffill().bfill()

        return filtered

    def apply_butterworth_filter(
        self, cutoff: float = 0.05, order: int = 2
    ) -> pd.DataFrame:
        """
        Butterworth passe-bas (version CAUSALE) :
        - on utilise lfilter (et non plus filtfilt)
        - chaque valeur filtrée à t ne dépend que de l'historique <= t
        cutoff petit => lissage fort.
        """
        logger.info("Applying Butterworth filter (causal lfilter)...")
        b, a = butter(order, cutoff, btype="low", analog=False)

        def _filter(col: pd.Series) -> pd.Series:
            col = col.astype(float)
            if col.isnull().any():
                col = col.ffill().bfill()
            # lfilter : filtre causal
            y = lfilter(b, a, col.values)
            return pd.Series(y, index=col.index)

        return self.cleaned_data.apply(_filter)

    def apply_savgol_filter_causal(
        self,
        window: int = 7,
        polyorder: int = 2,
    ) -> pd.DataFrame:
        """
        Savitzky-Golay *causal* :

        Pour chaque date t, on regarde UNIQUEMENT une fenêtre [t-window+1, t]
        (donc uniquement le passé), on fit un polynôme de degré polyorder
        et on évalue ce polynôme en t.

        Implémenté via rolling().apply() + np.polyfit (pas le savgol_filter centré).
        """
        logger.info("Applying Savitzky-Golay *causal* filter...")
        if window % 2 == 0:
            raise ValueError("window length must be odd for Savitzky-Golay filter.")

        def _rolling_savgol_last(values: np.ndarray) -> float:
            # values = [y_{t-window+1}, ..., y_t]
            x = np.arange(len(values), dtype=float)
            try:
                coeffs = np.polyfit(x, values, polyorder)
                return np.polyval(coeffs, x[-1])  # valeur au dernier point (t)
            except np.linalg.LinAlgError:
                # en cas de problème numérique, on renvoie la dernière valeur brute
                return values[-1]

        def _filter(col: pd.Series) -> pd.Series:
            col = col.astype(float)
            col = col.ffill().bfill()
            # rolling "causal" : la fenêtre est [t-window+1, t]
            return col.rolling(window=window, min_periods=window).apply(
                _rolling_savgol_last, raw=True
            )

        return self.cleaned_data.apply(_filter)

    def apply_moving_average(self, window: int = 10) -> pd.DataFrame:
        """
        MA simple : baseline de lissage, causal par construction.
        """
        logger.info("Applying Moving Average filter...")
        return self.cleaned_data.rolling(window=window, min_periods=window).mean()

    def apply_ta_lib_filter(self, timeperiod: int = 10) -> pd.DataFrame:
        """
        SMA via TA-Lib (identique à MA simple mais pratique pour ajouter RSI/MACD etc).
        Causal par nature (ne regarde que le passé).
        """
        logger.info("Applying TA-Lib SMA filter...")
        ta_filtered = pd.DataFrame(index=self.cleaned_data.index)

        for col in self.cleaned_data.columns:
            arr = self.cleaned_data[col].astype(float).values
            ta_filtered[col] = talib.SMA(arr, timeperiod=timeperiod)

        ta_filtered = ta_filtered.ffill().bfill()
        return ta_filtered

    # ---------- ANOMALY / REGIME DETECTION ----------

    def anomaly_detection(
        self,
        use_markov: bool = True,
        use_prophet: bool = True,
        markov_k_regimes: int = 2,
        markov_min_std: float = 1e-6,
        prophet_sigma_threshold: float = 2.0,
    ) -> pd.DataFrame:
        """
        Retourne un DataFrame d'anomalies/régimes.

        Markov:
          - détecte des régimes sur RETURNS (pas sur prix)
          - markov_min_std protège les séries plates

        Prophet:
          - apprend trend+saisons, outlier si résidu > prophet_sigma_threshold * std
        """
        if self.cleaned_data is None:
            raise ValueError("You must call clean_data() before anomaly_detection().")

        logger.info("Detecting anomalies using Markov Switching and Prophet models...")
        anomalies: dict[str, pd.Series] = {}

        # -------- Markov regimes (SUR RETURNS) --------
        if use_markov:
            if self.returns is None:
                logger.warning(
                    "Returns not computed yet -> computing inside anomaly_detection."
                )
                self.returns = np.log(
                    self.cleaned_data / self.cleaned_data.shift(1)
                ).dropna(how="all")

            for col in self.returns.columns:
                col_name = str(col)
                logger.info("Markov model for %s", col_name)

                series = self.returns[col].dropna()

                if series.std() < markov_min_std:
                    logger.warning(
                        "Series %s too flat for Markov (std=%.2e) -> skipping",
                        col_name, series.std()
                    )
                    continue

                try:
                    model = MarkovRegression(
                        series,
                        k_regimes=markov_k_regimes,
                        trend="c",
                        switching_variance=True,
                    )
                    res = model.fit(disp=False)
                    regimes = res.smoothed_marginal_probabilities[1]

                    regimes = regimes.reindex(self.cleaned_data.index, method="nearest")
                    anomalies[f"{col_name}_regime"] = regimes

                except Exception as e:
                    logger.warning("Could not fit Markov model for %s: %s", col_name, e)

        # -------- Prophet outliers --------
        if use_prophet:
            for col in self.cleaned_data.columns:
                col_name = str(col)
                logger.info("Prophet anomaly detection for %s", col_name)

                df_prophet = pd.DataFrame(
                    {
                        "ds": pd.to_datetime(self.cleaned_data.index),
                        "y": self.cleaned_data[col].astype(float).ffill().bfill(),
                    }
                )

                try:
                    m = Prophet(
                        daily_seasonality=True,
                        weekly_seasonality=True,
                        yearly_seasonality=True,
                    )
                    m.fit(df_prophet)

                    future = m.make_future_dataframe(periods=0)
                    forecast = m.predict(future)

                    yhat = forecast["yhat"].values
                    residuals = df_prophet["y"].values - yhat

                    threshold = prophet_sigma_threshold * np.std(residuals)
                    outliers = np.abs(residuals) > threshold

                    outliers_series = pd.Series(outliers, index=self.cleaned_data.index)
                    anomalies[f"{col_name}_prophet_outlier"] = outliers_series

                except Exception as e:
                    logger.warning("Prophet failed for %s: %s", col_name, e)

        return pd.DataFrame(anomalies, index=self.cleaned_data.index)
