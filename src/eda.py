# Set the backend for matplotlib at the very top of the file
import matplotlib
# matplotlib.use('Agg')  # Use 'Agg' if you don't need interactive plotting (suitable for scripts)
# or
matplotlib.use('TkAgg')  # Use 'TkAgg' for GUI-based interactive plotting, if supported

# Now import other libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_notebook
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from scipy.stats import zscore
from tslearn.metrics import cdist_dtw
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

# output_notebook()

class EDA:
    def __init__(self, data):
        self.data = data
        self.standardized_data = pd.DataFrame(StandardScaler().fit_transform(data), 
                                              index=data.index, columns=data.columns)

    def plot_timeseries(self):
        """Visualize each time series using Bokeh."""
        print("Plotting time series with Bokeh...")
        plots = []
        for col in self.data.columns:
            p = figure(title=f"Time Series: {col}", x_axis_type='datetime', width=700, height=300)
            source = ColumnDataSource(data={'x': self.data.index, 'y': self.data[col]})
            p.line(x='x', y='y', source=source, line_width=2)
            p.xaxis.axis_label = "Date"
            p.yaxis.axis_label = "Value"
            plots.append(p)
        show(column(*plots))

    def correlation_matrix(self, max_lag=5):
        """Display correlation matrix and cross-correlation for lags 1–5."""
        print("Computing correlation and cross-correlations...")

        # Standard correlation
        corr = self.data.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.show()

        # Cross-correlation
        for lag in range(1, max_lag + 1):
            print(f"\nLag {lag} Cross-Correlation:")
            shifted = self.data.shift(lag)
            cross_corr = self.data.corrwith(shifted)
            display = pd.DataFrame(cross_corr, columns=[f"Lag {lag} Corr"])
            print(display)

    def dtw_clustermap(self):
        """Calculate DTW distances and plot as a seaborn clustermap."""
        print("Creating DTW clustermap...")

        # Ensure time series are the same length and standardized
        ts = self.standardized_data.T.values  # shape (n_series, n_timepoints)
        dist_matrix = cdist_dtw(ts)

        # Convert to DataFrame for seaborn
        dist_df = pd.DataFrame(dist_matrix, index=self.data.columns, columns=self.data.columns)
        sns.clustermap(dist_df, cmap="viridis")
        plt.title("DTW Clustermap")
        plt.show()

    def seasonality_tracker(self, period=252):
        """Track and visualize seasonal trends using decomposition."""
        print("Tracking seasonality...")

        for col in self.data.columns:
            try:
                print(f"Seasonality for {col}")
                decomposition = seasonal_decompose(self.data[col].dropna(), model='additive', period=period)
                decomposition.plot()
                plt.suptitle(f"Seasonal Decomposition of {col}")
                plt.show()
            except Exception as e:
                print(f"Could not decompose {col}: {e}")
