import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from scipy.signal import butter, filtfilt, savgol_filter
from pykalman import KalmanFilter
from prophet import Prophet
import talib

class Preprocessing:
    def __init__(self, data):
        self.raw_data = data
        self.cleaned_data = None
        self.transformed_data = None
        self.scaled_data = None

    def clean_data(self):
        """Imputes missing data using forward-fill and backward-fill strategy."""
        print("Cleaning data: imputing missing values...")
        self.cleaned_data = self.raw_data.ffill().bfill()
        missing_after = self.cleaned_data.isnull().sum().sum()
        if missing_after > 0:
            print(f"Warning: {missing_after} values remain missing after imputation.")
        else:
            print("Missing values successfully handled.")
        return self.cleaned_data

    def transform_data(self):
        """Compute log returns, moving averages, and normalize."""
        print("Transforming data: computing log returns and applying standard scaling...")
        returns = np.log(self.cleaned_data / self.cleaned_data.shift(1)).dropna()
        moving_avg = self.cleaned_data.rolling(window=10).mean().dropna()

        # Standardize returns
        scaler = StandardScaler()
        scaled_returns = pd.DataFrame(scaler.fit_transform(returns), index=returns.index, columns=returns.columns)

        self.transformed_data = returns
        self.scaled_data = scaled_returns
        self.moving_avg = moving_avg

        return {
            'returns': returns,
            'scaled_returns': scaled_returns,
            'moving_avg': moving_avg
        }

    def apply_filter(self, filter_type='moving_average', **kwargs):
        filters = {
            'kalman': self.apply_kalman_filter,
            'butterworth': self.apply_butterworth_filter,
            'savgol': self.apply_savgol_filter,
            'moving_average': self.apply_moving_average,
            'ta_lib': self.apply_ta_lib_filter,
        }
        if filter_type in filters:
            return filters[filter_type](**kwargs)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

    def apply_kalman_filter(self):
        print("Applying Kalman filter...")
        filtered = pd.DataFrame(index=self.cleaned_data.index)
        for col in self.cleaned_data.columns:
            kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
            data = self.cleaned_data[col].values
            data = data[~np.isnan(data)]  # Kalman requires no NaNs
            state_means, _ = kf.em(data, n_iter=5).filter(data)
            full_index = self.cleaned_data.index
            full_series = pd.Series(state_means.flatten(), index=full_index[-len(state_means):])
            filtered[col] = full_series
        return filtered

    def apply_butterworth_filter(self, cutoff=0.05, order=2):
        print("Applying Butterworth filter...")
        b, a = butter(order, cutoff, btype='low', analog=False)
        return self.cleaned_data.apply(lambda col: pd.Series(filtfilt(b, a, col), index=col.index))

    def apply_savgol_filter(self, window=7, polyorder=2):
        print("Applying Savitzky-Golay filter...")
        return self.cleaned_data.apply(lambda col: pd.Series(savgol_filter(col, window_length=window, polyorder=polyorder), index=col.index))

    def apply_moving_average(self, window=10):
        print("Applying Moving Average filter...")
        return self.cleaned_data.rolling(window=window).mean()

    def apply_ta_lib_filter(self):
        print("Applying TA-Lib smoothing filters (SMA)...")
        ta_filtered = pd.DataFrame(index=self.cleaned_data.index)
        for col in self.cleaned_data.columns:
            ta_filtered[col] = talib.SMA(self.cleaned_data[col], timeperiod=10)
        return ta_filtered

    def anomaly_detection(self):
        print("Detecting anomalies using Markov Switching and Prophet models...")
        anomalies = {}

        for col in self.cleaned_data.columns:
            col_str = "_".join(col) if isinstance(col, tuple) else col
            print(f"Markov model for {col}")

            try:
                model = MarkovRegression(self.cleaned_data[col].dropna(), k_regimes=2, trend='c', switching_variance=True)
                res = model.fit(disp=False)
                regimes = res.smoothed_marginal_probabilities[1]
                anomalies[col_str + '_regime'] = regimes
            except Exception as e:
                print(f"Could not fit Markov model for {col}: {e}")

        for col in self.cleaned_data.columns:
            col_str = "_".join(col) if isinstance(col, tuple) else col
            print(f"Prophet anomaly detection for {col}")

            df = pd.DataFrame({
                'ds': self.cleaned_data.index,
                'y': self.cleaned_data[col].fillna(method='ffill')
            })

            try:
                m = Prophet(daily_seasonality=True)
                m.fit(df)
                future = m.make_future_dataframe(periods=0)
                forecast = m.predict(future)
                yhat = forecast['yhat']
                residuals = df['y'] - yhat
                threshold = 2 * np.std(residuals)
                outliers = abs(residuals) > threshold
                anomalies[col_str + '_prophet_outlier'] = outliers
            except Exception as e:
                print(f"Prophet failed for {col}: {e}")

        return pd.DataFrame(anomalies, index=self.cleaned_data.index)

