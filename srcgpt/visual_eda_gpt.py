# srcgpt/visual_eda_gpt.py
"""
VisualEDA (fusion EDA v1 + VisualEDA v2) — sans Bokeh, interactif via Plotly.

Contenu :
- Time series interactives (Plotly)  ✅ remplace Bokeh
- Matrices de corrélation + corr à lag (EDA v1)
- Clustermap DTW (EDA v1) + clustermap corr/dist (v2)
- Seasonality tracker (EDA v1)
- Rolling heatmaps + overlays + spreads + distributions (v2)
- Comparaison interactive de filtres (Plotly)

Notes :
- Les clustermaps hiérarchiques restent seaborn/matplotlib (pas de “clustermap”
  native Plotly). Tout ce qui nécessite zoom/hover => Plotly.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from tslearn.metrics import cdist_dtw
from statsmodels.tsa.seasonal import seasonal_decompose

logger = logging.getLogger(__name__)


# -----------------------------
# Helpers
# -----------------------------

def _ensure_df(x: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    if x.empty:
        raise ValueError(f"{name} is empty.")
    return x


def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns if needed (ex: tuples)."""
    df = df.copy()
    df.columns = [
        c[0] if isinstance(c, tuple) and len(c) > 0 else c
        for c in df.columns
    ]
    return df


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std(ddof=0)


def _try_import_plotly():
    """Import Plotly only when needed (interactive plots)."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        return go, make_subplots
    except Exception as e:
        raise ImportError(
            "Plotly is required for interactive plots. Install with `pip install plotly nbformat`."
        ) from e


def _try_import_fastdtw():
    """Import fastdtw only for DTW alignment plots."""
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        return fastdtw, euclidean
    except Exception as e:
        raise ImportError(
            "fastdtw is required for DTW alignment plots. Install with `pip install fastdtw`."
        ) from e


# -----------------------------
# Config dataclass (optionnel)
# -----------------------------

@dataclass
class VisualEDAConfig:
    style: str = "whitegrid"
    figsize: Tuple[int, int] = (12, 6)
    cmap: str = "coolwarm"
    dpi: int = 110


# -----------------------------
# Main class
# -----------------------------

class VisualEDA:
    """
    Visual EDA pour pipeline de trading macro (fusion EDA + VisualEDA).

    ✅ Interactif Plotly :
      - plot_timeseries()
      - plot_interactive_prices / returns / corr
      - plot_filters_comparison_interactive()

    📌 Statique seaborn/mpl :
      - clustermaps hiérarchiques
      - heatmaps rolling
      - saisonnalité decomposition
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        returns: Optional[pd.DataFrame] = None,
        config: Optional[VisualEDAConfig] = None,
    ):
        sns.set_theme(style=(config.style if config else "whitegrid"))
        self.config = config or VisualEDAConfig()

        prices = _ensure_df(prices, "prices")
        self.prices = _flatten_cols(prices).dropna(how="all")

        # Returns par défaut = log-returns sur prix
        if returns is None:
            returns = np.log(self.prices / self.prices.shift(1))

        returns = _ensure_df(returns, "returns")
        self.returns = _flatten_cols(returns).dropna(how="all")

        # Align indices prices/returns
        idx = self.prices.index.intersection(self.returns.index)
        self.prices = self.prices.loc[idx]
        self.returns = self.returns.loc[idx]

        # Standardisation utile pour DTW/compare multi-assets
        self.standardized_returns = pd.DataFrame(
            StandardScaler().fit_transform(self.returns.dropna()),
            index=self.returns.dropna().index,
            columns=self.returns.columns,
        )

        logger.info("VisualEDA initialized. prices=%s returns=%s",
                    self.prices.shape, self.returns.shape)

    # =========================================================
    # 0) Time series interactives (Plotly, remplace Bokeh)
    # =========================================================

    def plot_timeseries(
        self,
        assets: Optional[Sequence[str]] = None,
        use_returns: bool = False,
        normalize: bool = False,
    ):
        """
        Plot interactif Plotly des séries (prix ou returns).
        - assets=None => toutes les colonnes.
        - normalize=True => divise par la première valeur (uniquement sur prix).
        """
        go, make_subplots = _try_import_plotly()

        data = self.returns if use_returns else self.prices
        if assets is not None:
            data = data[list(assets)]
        data = data.dropna()

        if normalize and not use_returns:
            data = data / data.iloc[0]

        fig = make_subplots(
            rows=len(data.columns),
            cols=1,
            shared_xaxes=True,
            subplot_titles=list(data.columns),
            vertical_spacing=0.03
        )

        for i, col in enumerate(data.columns, start=1):
            fig.add_trace(
                go.Scatter(x=data.index, y=data[col], mode="lines", name=col),
                row=i, col=1
            )

        fig.update_layout(
            height=250 * len(data.columns),
            title="Interactive Time Series (returns)" if use_returns else "Interactive Time Series (prices)",
            hovermode="x unified",
            template="plotly_white",
        )
        fig.show()
        return fig


    # =========================================================
    # 1) Corrélation + corr à lag (EDA v1)
    # =========================================================

    def correlation_matrix(
        self,
        max_lag: int = 5,
        method: str = "pearson",
        show_plot: bool = True,
        interactive: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Corrélation simple + corr avec versions décalées (lag).
        Retourne un dict:
          {"corr": corr_matrix, "lag_1": ..., ...}
        """
        results: Dict[str, pd.DataFrame] = {}
        corr = self.returns.corr(method=method)
        results["corr"] = corr

        if show_plot:
            if interactive:
                go, _ = _try_import_plotly()
                fig = go.Figure(
                    data=go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.index,
                        colorscale="RdBu",
                        zmid=0,
                    )
                )
                fig.update_layout(title="Correlation Matrix (interactive)", height=600)
                fig.show()
            else:
                plt.figure(figsize=(8, 6))
                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
                plt.title("Correlation Matrix")
                plt.tight_layout()
                plt.show()

        # Cross-corr avec lag
        for lag in range(1, max_lag + 1):
            shifted = self.returns.shift(lag)
            cross_corr = self.returns.corrwith(shifted)
            results[f"lag_{lag}"] = pd.DataFrame(cross_corr, columns=[f"Lag {lag} Corr"])

        return results


    # =========================================================
    # 2) DTW clustermap (EDA v1)
    # =========================================================

    def dtw_clustermap(self, show_plot: bool = True) -> pd.DataFrame:
        """
        Distance DTW entre séries standardisées (returns).
        Plot hiérarchique seaborn (statique).
        """
        ts = self.standardized_returns.T.values
        dist_matrix = cdist_dtw(ts)

        dist_df = pd.DataFrame(
            dist_matrix,
            index=self.standardized_returns.columns,
            columns=self.standardized_returns.columns,
        )

        if show_plot:
            sns.clustermap(dist_df, cmap="viridis")
            plt.title("DTW Clustermap (distance)")
            plt.show()

        return dist_df


    # =========================================================
    # 3) Seasonality tracker (EDA v1)
    # =========================================================

    def seasonality_tracker(
        self,
        period: int = 252,
        show_plots: bool = True,
    ) -> Dict[str, Any]:
        """
        Décomposition saisonnière (trend / seasonal / resid).
        Statique matplotlib (statsmodels).
        """
        results: Dict[str, Any] = {}
        for col in self.prices.columns:
            try:
                decomposition = seasonal_decompose(
                    self.prices[col].dropna(),
                    model="additive",
                    period=period
                )
                results[col] = decomposition

                if show_plots:
                    decomposition.plot()
                    plt.suptitle(f"Seasonal Decomposition of {col}")
                    plt.tight_layout()
                    plt.show()
            except Exception as e:
                logger.warning("Could not decompose %s: %s", col, e)

        return results


    # =========================================================
    # 4) Rolling heatmaps (v2)
    # =========================================================

    def plot_rolling_correlation_heatmap(
        self,
        window: int = 60,
        step: int = 5,
        method: str = "pearson",
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        rets = self.returns.dropna()
        if len(rets) < window:
            raise ValueError("Not enough data for rolling correlation.")

        dates, mats = [], []
        for start in range(0, len(rets) - window + 1, step):
            end = start + window
            w = rets.iloc[start:end]
            mats.append(w.corr(method=method))
            dates.append(w.index[-1])

        pairs = []
        for i, a in enumerate(rets.columns):
            for j, b in enumerate(rets.columns):
                if j <= i:
                    continue
                pairs.append((a, b))

        corr_ts = pd.DataFrame(index=dates, columns=[f"{a}~{b}" for a, b in pairs], dtype=float)
        for t, c in zip(dates, mats):
            for a, b in pairs:
                corr_ts.loc[t, f"{a}~{b}"] = c.loc[a, b]

        plt.figure(figsize=(max(10, corr_ts.shape[1] * 0.6), 6), dpi=self.config.dpi)
        sns.heatmap(corr_ts.T, cmap=self.config.cmap, center=0,
                    cbar_kws={"label": f"Rolling Corr ({window}d)"})
        plt.title("Rolling Correlation Heatmap (pairs over time)")
        plt.xlabel("Time")
        plt.ylabel("Pairs")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
        if show:
            plt.show()
        else:
            plt.close()

        return corr_ts


    def plot_rolling_volatility_heatmap(
        self,
        window: int = 60,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        rets = self.returns.dropna()
        vol = rets.rolling(window).std()

        plt.figure(figsize=(10, 5), dpi=self.config.dpi)
        sns.heatmap(vol.T, cmap="viridis",
                    cbar_kws={"label": f"Rolling Vol ({window}d)"})
        plt.title("Rolling Volatility Heatmap")
        plt.xlabel("Time")
        plt.ylabel("Assets")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
        if show:
            plt.show()
        else:
            plt.close()

        return vol


    def plot_returns_zscore_heatmap(
        self,
        window: Optional[int] = None,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        rets = self.returns.dropna()

        if window is None:
            z = _zscore(rets)
            title = "Returns Z-Score Heatmap (global)"
        else:
            z = (rets - rets.rolling(window).mean()) / rets.rolling(window).std(ddof=0)
            title = f"Returns Z-Score Heatmap (rolling {window}d)"

        plt.figure(figsize=(10, 5), dpi=self.config.dpi)
        sns.heatmap(z.T, cmap="coolwarm", center=0,
                    cbar_kws={"label": "Z-score"})
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Assets")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
        if show:
            plt.show()
        else:
            plt.close()

        return z


    # =========================================================
    # 5) Clustermaps corr/dist (v2)
    # =========================================================

    def plot_correlation_clustermap(
        self,
        method: str = "pearson",
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        corr = self.returns.corr(method=method)

        g = sns.clustermap(
            corr, cmap=self.config.cmap, center=0,
            figsize=(8, 8), linewidths=0.5
        )
        g.fig.suptitle("Hierarchical Clustermap (Correlation)", y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
        if show:
            plt.show()
        else:
            plt.close()

        return corr


    def plot_distance_clustermap(
        self,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        corr = self.returns.corr()
        dist = 1.0 - corr

        g = sns.clustermap(dist, cmap="viridis",
                           figsize=(8, 8), linewidths=0.5)
        g.fig.suptitle("Hierarchical Clustermap (1 - Corr distance)", y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
        if show:
            plt.show()
        else:
            plt.close()

        return dist


    # =========================================================
    # 6) DTW visualisations (v2)
    # =========================================================

    def plot_dtw_alignment(
        self,
        asset_a: str,
        asset_b: str,
        use_returns: bool = True,
        max_path_points: int = 100,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        fastdtw, euclidean = _try_import_fastdtw()

        data = self.returns if use_returns else self.prices
        if asset_a not in data.columns or asset_b not in data.columns:
            raise ValueError("Assets not found in data columns.")

        s1 = data[asset_a].dropna().values
        s2 = data[asset_b].dropna().values
        dist, path = fastdtw(s1, s2, dist=lambda u, v: euclidean([u], [v]))

        plt.figure(figsize=(10, 4), dpi=self.config.dpi)
        plt.plot(s1, label=asset_a, alpha=0.8)
        plt.plot(s2, label=asset_b, alpha=0.8)

        step = max(1, len(path) // max_path_points)
        for (i, j) in path[::step]:
            plt.plot([i, j], [s1[i], s2[j]], color="gray", alpha=0.2)

        plt.title(f"DTW Alignment Path: {asset_a} vs {asset_b} (dist={dist:.2f})")
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
        if show:
            plt.show()
        else:
            plt.close()

        return {"distance": dist, "path": path}


    def plot_rolling_dtw_distance_heatmap(
        self,
        window: int = 90,
        step: int = 10,
        use_returns: bool = True,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        fastdtw, euclidean = _try_import_fastdtw()
        data = self.returns if use_returns else self.prices
        data = data.dropna()
        if len(data) < window:
            raise ValueError("Not enough data for rolling DTW.")

        cols = list(data.columns)
        pairs = [(cols[i], cols[j]) for i in range(len(cols))
                 for j in range(i + 1, len(cols))]

        dates, dist_ts = [], []
        for start in range(0, len(data) - window + 1, step):
            end = start + window
            w = data.iloc[start:end]
            dates.append(w.index[-1])

            row = {}
            for a, b in pairs:
                s1, s2 = w[a].values, w[b].values
                d, _ = fastdtw(s1, s2, dist=lambda u, v: euclidean([u], [v]))
                row[f"{a}~{b}"] = d
            dist_ts.append(row)

        dist_df = pd.DataFrame(dist_ts, index=dates)

        plt.figure(figsize=(max(10, dist_df.shape[1] * 0.6), 6), dpi=self.config.dpi)
        sns.heatmap(dist_df.T, cmap="magma",
                    cbar_kws={"label": f"Rolling DTW ({window}d)"})
        plt.title("Rolling DTW Distance Heatmap (pairs over time)")
        plt.xlabel("Time")
        plt.ylabel("Pairs")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
        if show:
            plt.show()
        else:
            plt.close()

        return dist_df


    # =========================================================
    # 7) Macro overlays (v2)
    # =========================================================

    def plot_macro_overlay_prices(
        self,
        assets: Sequence[str],
        normalize: bool = True,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        for a in assets:
            if a not in self.prices.columns:
                raise ValueError(f"{a} not in prices.")

        sub = self.prices[list(assets)].dropna()
        if normalize:
            sub = sub / sub.iloc[0]

        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        for col in sub.columns:
            plt.plot(sub.index, sub[col], label=col)

        plt.title("Macro Overlay (Prices)")
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
        if show:
            plt.show()
        else:
            plt.close()

        return sub


    def plot_macro_overlay_returns(
        self,
        assets: Sequence[str],
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        for a in assets:
            if a not in self.returns.columns:
                raise ValueError(f"{a} not in returns.")

        sub = self.returns[list(assets)].dropna()

        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        for col in sub.columns:
            plt.plot(sub.index, sub[col], label=col, alpha=0.8)

        plt.title("Macro Overlay (Returns)")
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
        if show:
            plt.show()
        else:
            plt.close()

        return sub


    def plot_overlay_spread(
        self,
        asset_a: str,
        asset_b: str,
        use_returns: bool = True,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> pd.Series:
        data = self.returns if use_returns else self.prices
        if asset_a not in data.columns or asset_b not in data.columns:
            raise ValueError("Assets not found.")

        spread = (data[asset_a] - data[asset_b]).dropna()

        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        plt.plot(spread.index, spread.values, label=f"{asset_a} - {asset_b}")
        plt.axhline(0, linestyle="--", alpha=0.5)
        plt.title("Spread Overlay")
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
        if show:
            plt.show()
        else:
            plt.close()

        return spread


    # =========================================================
    # 8) Interactif Plotly (v2)
    # =========================================================

    def plot_interactive_prices(
        self,
        assets: Optional[Sequence[str]] = None,
        normalize: bool = True,
    ):
        go, _ = _try_import_plotly()
        px = self.prices.copy()
        if assets is not None:
            px = px[list(assets)]
        px = px.dropna()

        if normalize:
            px = px / px.iloc[0]

        fig = go.Figure()
        for col in px.columns:
            fig.add_trace(go.Scatter(x=px.index, y=px[col], mode="lines", name=col))

        fig.update_layout(
            title="Interactive Prices Overlay",
            xaxis_title="Date",
            yaxis_title="Normalized Price" if normalize else "Price",
            height=500,
            hovermode="x unified",
            template="plotly_white",
        )
        fig.show()
        return fig


    def plot_interactive_returns(self, assets: Optional[Sequence[str]] = None):
        go, _ = _try_import_plotly()
        rets = self.returns.copy()
        if assets is not None:
            rets = rets[list(assets)]
        rets = rets.dropna()

        fig = go.Figure()
        for col in rets.columns:
            fig.add_trace(go.Scatter(x=rets.index, y=rets[col], mode="lines", name=col))

        fig.update_layout(
            title="Interactive Returns Overlay",
            xaxis_title="Date",
            yaxis_title="Log Return",
            height=500,
            hovermode="x unified",
            template="plotly_white",
        )
        fig.show()
        return fig


    def plot_interactive_correlation_matrix(self):
        go, _ = _try_import_plotly()
        corr = self.returns.corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale="RdBu",
                zmid=0,
            )
        )
        fig.update_layout(title="Interactive Correlation Heatmap", height=600)
        fig.show()
        return fig


    # =========================================================
    # 9) Distributions (v2)
    # =========================================================

    def plot_returns_distribution(self, asset: str, bins: int = 50,
                                  show: bool = True, save_path: Optional[str] = None):
        if asset not in self.returns.columns:
            raise ValueError(f"{asset} not in returns.")
        x = self.returns[asset].dropna()

        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        sns.histplot(x, bins=bins, kde=True)
        plt.title(f"Return Distribution: {asset}")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
        if show:
            plt.show()
        else:
            plt.close()


    def plot_distribution_evolution(self, asset: str, window: int = 90,
                                    show: bool = True, save_path: Optional[str] = None) -> pd.DataFrame:
        if asset not in self.returns.columns:
            raise ValueError(f"{asset} not in returns.")

        r = self.returns[asset].dropna()
        skew = r.rolling(window).skew()
        kurt = r.rolling(window).kurt()

        stats = pd.DataFrame({"skew": skew, "kurtosis": kurt})

        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        plt.plot(stats.index, stats["skew"], label="Skewness")
        plt.plot(stats.index, stats["kurtosis"], label="Kurtosis")
        plt.axhline(0, linestyle="--", alpha=0.4)
        plt.title(f"Distribution Evolution ({window}d rolling): {asset}")
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
        if show:
            plt.show()
        else:
            plt.close()

        return stats


    # =========================================================
    # 10) Smooth / cum / vol / zscore overlays (v2)
    # =========================================================

    def plot_smoothed_returns(self, assets: Optional[Sequence[str]] = None,
                              method: str = "ewma", window: int = 10, span: int = 10,
                              show: bool = True, save_path: Optional[str] = None) -> pd.DataFrame:
        rets = self.returns.copy()
        if assets is not None:
            rets = rets[list(assets)]
        rets = rets.dropna()

        if method == "rolling":
            smooth = rets.rolling(window).mean()
            title = f"Smoothed Returns (Rolling Mean {window}d)"
        elif method == "ewma":
            smooth = rets.ewm(span=span).mean()
            title = f"Smoothed Returns (EWMA span={span})"
        else:
            raise ValueError("method must be 'rolling' or 'ewma'")

        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        for col in smooth.columns:
            plt.plot(smooth.index, smooth[col], label=col)
        plt.axhline(0, linestyle="--", alpha=0.4)
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
        if show:
            plt.show()
        else:
            plt.close()

        return smooth


    def plot_cumulated_returns(self, assets: Optional[Sequence[str]] = None,
                               show: bool = True, save_path: Optional[str] = None) -> pd.DataFrame:
        rets = self.returns.copy()
        if assets is not None:
            rets = rets[list(assets)]
        rets = rets.dropna()

        simple_ret = np.exp(rets) - 1.0
        cum = (1.0 + simple_ret).cumprod()

        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        for col in cum.columns:
            plt.plot(cum.index, cum[col], label=col)
        plt.title("Cumulated Returns (Performance-like curves)")
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
        if show:
            plt.show()
        else:
            plt.close()

        return cum


    def plot_rolling_volatility(self, assets: Optional[Sequence[str]] = None,
                                window: int = 20, annualize: bool = True,
                                show: bool = True, save_path: Optional[str] = None) -> pd.DataFrame:
        rets = self.returns.copy()
        if assets is not None:
            rets = rets[list(assets)]
        rets = rets.dropna()

        vol = rets.rolling(window).std()
        if annualize:
            vol = vol * np.sqrt(252)
            title = f"Rolling Volatility ({window}d, annualized)"
        else:
            title = f"Rolling Volatility ({window}d)"

        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        for col in vol.columns:
            plt.plot(vol.index, vol[col], label=col)
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
        if show:
            plt.show()
        else:
            plt.close()

        return vol


    def plot_zscore_returns_overlay(self, assets: Optional[Sequence[str]] = None,
                                    window: Optional[int] = None,
                                    show: bool = True, save_path: Optional[str] = None) -> pd.DataFrame:
        rets = self.returns.copy()
        if assets is not None:
            rets = rets[list(assets)]
        rets = rets.dropna()

        if window is None:
            z = (rets - rets.mean()) / rets.std(ddof=0)
            title = "Returns Z-score Overlay (global)"
        else:
            z = (rets - rets.rolling(window).mean()) / rets.rolling(window).std(ddof=0)
            title = f"Returns Z-score Overlay (rolling {window}d)"

        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        for col in z.columns:
            plt.plot(z.index, z[col], label=col)
        plt.axhline(0, linestyle="--", alpha=0.4)
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
        if show:
            plt.show()
        else:
            plt.close()

        return z


    # =========================================================
    # 11) Comparaison interactive de filtres (Plotly)
    # =========================================================

    def plot_filters_comparison_interactive(
        self,
        asset: str,
        filtered_prices: Dict[str, pd.DataFrame],
        normalize: bool = True,
        renderer: str = "auto",
    ):
        """
        Compare interactively raw prices vs plusieurs filtres.

        filtered_prices = { "savgol": df1, "kalman": df2, "butter": df3, ... }
        """
        import plotly.graph_objects as go
        import plotly.io as pio

        # Renderer simple
        if renderer == "browser":
            pio.renderers.default = "browser"
        elif renderer == "notebook":
            pio.renderers.default = "notebook_connected"
        else:
            # auto: notebook si possible sinon browser
            try:
                pio.renderers.default = "notebook_connected"
            except Exception:
                pio.renderers.default = "browser"

        raw = self.prices[asset].dropna()
        series_dict: Dict[str, pd.Series] = {"raw": raw}

        for name, df in filtered_prices.items():
            s = df[asset].dropna()
            series_dict[name] = s

        # align index
        idx = raw.index
        for s in series_dict.values():
            idx = idx.intersection(s.index)
        if len(idx) == 0:
            raise ValueError("No overlapping dates between raw and filtered series.")

        for k, s in series_dict.items():
            s = s.loc[idx]
            if normalize:
                s = s / s.iloc[0]
            series_dict[k] = s

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=idx, y=series_dict["raw"],
            mode="lines", name="raw"
        ))

        for name, s in series_dict.items():
            if name == "raw":
                continue
            fig.add_trace(go.Scatter(x=idx, y=s, mode="lines", name=name))

        fig.update_layout(
            title=f"{asset} — Raw vs Filtered Prices",
            xaxis_title="Date",
            yaxis_title="Normalized price" if normalize else "Price",
            hovermode="x unified",
            height=550,
            template="plotly_white",
        )
        fig.show()
        return fig


    def plot_rolling_volatility_interactive(
    self,
    asset: str,
    windows=(10, 20, 60, 120, 252),
    annualize: bool = True,
    renderer: str = "auto",
    ):
        """
        Plot interactif Plotly de volatilité glissante pour un asset,
        sur plusieurs fenêtres à la fois.

        - Clique sur la légende pour afficher/cacher certaines fenêtres.
        - Permet de comprendre la sensibilité à la window.

        Parameters
        ----------
        asset : str
            ticker à afficher (ex: "GC=F")
        windows : tuple[int]
            tailles de fenêtres à comparer
        annualize : bool
            si True, multiplie par sqrt(252)
        renderer : str
            "auto", "notebook", "browser"
        """
        import plotly.graph_objects as go
        import plotly.io as pio

        # Renderer safe
        if renderer == "browser":
            pio.renderers.default = "browser"
        elif renderer == "notebook":
            pio.renderers.default = "notebook_connected"
        else:
            # auto : essai notebook, sinon browser
            try:
                pio.renderers.default = "notebook_connected"
            except Exception:
                pio.renderers.default = "browser"

        if asset not in self.returns.columns:
            raise ValueError(f"{asset} not in returns.")

        r = self.returns[asset].dropna()

        fig = go.Figure()

        for w in windows:
            vol = r.rolling(w).std()
            if annualize:
                vol = vol * np.sqrt(252)

            fig.add_trace(
                go.Scatter(
                    x=vol.index,
                    y=vol.values,
                    mode="lines",
                    name=f"window={w}",
                )
            )

        fig.update_layout(
            title=f"Rolling Volatility — {asset} (multi-window)",
            xaxis_title="Date",
            yaxis_title="Annualized Vol" if annualize else "Vol",
            hovermode="x unified",
            height=520,
            template="plotly_white",
        )

        fig.show()
        return fig
