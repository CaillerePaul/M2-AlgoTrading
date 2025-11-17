from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


class LeadLagDTW:
    """
    Analyse lead/lag entre séries temporelles à partir de DTW.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data.dropna(how="all")

    @staticmethod
    def _to_1d(series: np.ndarray) -> np.ndarray:
        return np.asarray(series).flatten()

    def compute_dtw(
        self,
        series1: np.ndarray,
        series2: np.ndarray,
        plot: bool = False,
        max_segments: int = 100,
    ) -> Tuple[float, List[Tuple[int, int]]]:
        s1 = self._to_1d(series1)
        s2 = self._to_1d(series2)

        def custom_euclidean(u, v):
            return euclidean([u], [v])

        distance, path = fastdtw(s1, s2, dist=custom_euclidean)

        if plot:
            plt.figure(figsize=(10, 4))
            plt.plot(s1, label="Series 1", alpha=0.7)
            plt.plot(s2, label="Series 2", alpha=0.7)

            step = max(len(path) // max_segments, 1)
            for (i, j) in path[::step]:
                plt.plot([i, j], [s1[i], s2[j]], color="gray", alpha=0.3)

            plt.legend()
            plt.title("DTW Alignment Path")
            plt.tight_layout()
            plt.show()

        return distance, path

    def identify_lead_lag(self) -> pd.DataFrame:
        assets = self.data.columns
        lead_lag_results = []

        for i, asset_a in enumerate(assets):
            for j, asset_b in enumerate(assets):
                if i >= j:
                    continue

                series_a = self.data[asset_a].values
                series_b = self.data[asset_b].values

                distance, path = self.compute_dtw(series_a, series_b, plot=False)
                lead_counts = [ia - jb for ia, jb in path]
                avg_lag = float(np.mean(lead_counts))

                if avg_lag < 0:
                    leader, lagging = asset_a, asset_b
                else:
                    leader, lagging = asset_b, asset_a

                relationship = {
                    "Leader": leader,
                    "Lagging": lagging,
                    "Average Lag": abs(avg_lag),
                    "DTW Distance": distance,
                    "Pair": (asset_a, asset_b),
                }

                lead_lag_results.append(relationship)

        return pd.DataFrame(lead_lag_results).sort_values(
            by="Average Lag", ascending=False
        )
