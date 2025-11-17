from typing import Dict, List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource

from sklearn.preprocessing import StandardScaler
from tslearn.metrics import cdist_dtw
from statsmodels.tsa.seasonal import seasonal_decompose


class EDA:
    """
    Exploratory Data Analysis sur un DataFrame de séries temporelles.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.standardized_data = pd.DataFrame(
            StandardScaler().fit_transform(data),
            index=data.index,
            columns=data.columns,
        )

    def plot_timeseries(self, show_plots: bool = True):
        plots: List[figure] = []

        for col in self.data.columns:
            p = figure(
                title=f"Time Series: {col}",
                x_axis_type="datetime",
                width=800,
                height=300,
            )
            source = ColumnDataSource(data={"x": self.data.index, "y": self.data[col]})
            p.line(x="x", y="y", source=source, line_width=2)
            p.xaxis.axis_label = "Date"
            p.yaxis.axis_label = "Value"
            plots.append(p)

        if show_plots and len(plots) > 0:
            show(column(*plots))

        return plots

    def correlation_matrix(
        self, max_lag: int = 5, show_plot: bool = True
    ) -> Dict[str, pd.DataFrame]:
        results: Dict[str, pd.DataFrame] = {}

        corr = self.data.corr()
        results["corr"] = corr

        if show_plot:
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Matrix")
            plt.tight_layout()
            plt.show()

        for lag in range(1, max_lag + 1):
            shifted = self.data.shift(lag)
            cross_corr = self.data.corrwith(shifted)
            results[f"lag_{lag}"] = pd.DataFrame(
                cross_corr, columns=[f"Lag {lag} Corr"]
            )

        return results

    def dtw_clustermap(self, show_plot: bool = True):
        ts = self.standardized_data.T.values
        dist_matrix = cdist_dtw(ts)

        dist_df = pd.DataFrame(
            dist_matrix,
            index=self.data.columns,
            columns=self.data.columns,
        )

        if show_plot:
            sns.clustermap(dist_df, cmap="viridis")
            plt.title("DTW Clustermap")
            plt.show()

        return dist_df

    def seasonality_tracker(self, period: int = 252, show_plots: bool = True):
        results: Dict[str, object] = {}

        for col in self.data.columns:
            try:
                decomposition = seasonal_decompose(
                    self.data[col].dropna(), model="additive", period=period
                )
                results[col] = decomposition

                if show_plots:
                    decomposition.plot()
                    plt.suptitle(f"Seasonal Decomposition of {col}")
                    plt.tight_layout()
                    plt.show()
            except Exception as e:
                print(f"Could not decompose {col}: {e}")

        return results
