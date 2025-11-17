import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw  # Fast approximation of DTW

class LeadLagDTW:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with a DataFrame of financial time series (columns = assets, index = time).
        """
        self.data = data.dropna()  # Drop missing data for simplicity

    def compute_dtw(self, series1, series2, plot=False):
        """
        Compute DTW distance and warping path between two time series.
        """
        # Reshape the series to ensure they are 1D
        # Ensure both series are 1D arrays
        series1 = series1.flatten()  # Flatten the series to 1D
        series2 = series2.flatten()  # Flatten the series to 1D

        def custom_euclidean(u, v):
            return euclidean([u], [v])
        
        distance, path = fastdtw(series1, series2, dist=custom_euclidean)

        if plot:
            plt.figure(figsize=(10, 4))
            plt.plot(series1, label="Series 1", alpha=0.7)
            plt.plot(series2, label="Series 2", alpha=0.7)
            for (i, j) in path[::len(path)//100 or 1]:  # Sample some path points
                plt.plot([i, j], [series1[i], series2[j]], color='gray', alpha=0.3)
            plt.legend()
            plt.title("DTW Alignment Path")
            plt.show()

        return distance, path

    def identify_lead_lag(self):
        """
        For all asset pairs, compute DTW path and infer who leads whom.
        """
        assets = self.data.columns
        lead_lag_results = []

        for i, asset_a in enumerate(assets):
            for j, asset_b in enumerate(assets):
                if i >= j:
                    continue

                series_a = self.data[asset_a].values
                series_b = self.data[asset_b].values

                distance, path = self.compute_dtw(series_a, series_b)
                lead_counts = [i - j for i, j in path]

                avg_lag = np.mean(lead_counts)
                relationship = {
                    'Leader': asset_a if avg_lag < 0 else asset_b,
                    'Lagging': asset_b if avg_lag < 0 else asset_a,
                    'Average Lag': abs(avg_lag),
                    'DTW Distance': distance,
                    'Pair': (asset_a, asset_b)
                }

                lead_lag_results.append(relationship)

        return pd.DataFrame(lead_lag_results).sort_values(by='Average Lag', ascending=False)
