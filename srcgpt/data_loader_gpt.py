import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
import missingno as msno
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

logger = logging.getLogger(__name__)


@dataclass
class DataLoaderConfig:
    tickers: List[str]
    start_date: str
    end_date: str
    price_column: str = "Adj Close"  # fallback sur "Close" si absent
    resample_rule: Optional[str] = None  # ex: "B", "D", "W"


class DataLoader:
    """
    Téléchargement des données, merge en matrice Date x Ticker,
    EDA des valeurs manquantes et imputation.
    """

    def __init__(self, config: DataLoaderConfig):
        self.config = config

    # ---------- FETCHING ----------

    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}

        for ticker in self.config.tickers:
            logger.info("Fetching data for %s", ticker)
            df = yf.download(
                ticker,
                start=self.config.start_date,
                end=self.config.end_date,
                auto_adjust=False,
                progress=False,
            )

            if df.empty:
                logger.warning("No data returned for %s", ticker)
                continue

            df["Ticker"] = ticker

            if self.config.resample_rule is not None:
                df = df.resample(self.config.resample_rule).last()

            data[ticker] = df

        return data

    # ---------- MERGING ----------

    def merge_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        merged = pd.DataFrame()

        for ticker, df in data_dict.items():
            price_col = self._choose_price_column(df)
            series = df[[price_col]].rename(columns={price_col: ticker})

            if merged.empty:
                merged = series
            else:
                merged = merged.join(series, how="outer")

        return merged

    def _choose_price_column(self, df: pd.DataFrame) -> str:
        if self.config.price_column in df.columns:
            return self.config.price_column
        elif "Close" in df.columns:
            logger.warning(
                "Price column '%s' not found, falling back to 'Close'",
                self.config.price_column,
            )
            return "Close"
        else:
            raise ValueError("No suitable price column found in downloaded data.")

    # ---------- MISSING DATA EDA ----------

    def eda_missing(self, merged_data: pd.DataFrame, show: bool = True) -> None:
        logger.info("Visualizing missing data with missingno")

        if merged_data.empty:
            logger.warning("Merged data is empty, skipping missingno plots.")
            return

        msno.matrix(merged_data)
        plt.title("Missing Data Matrix")
        if show:
            plt.show()
        else:
            plt.close()

        msno.heatmap(merged_data)
        plt.title("Missing Data Heatmap")
        if show:
            plt.show()
        else:
            plt.close()

    def missing_test(self, merged_data: pd.DataFrame) -> None:
        logger.info("Performing MCAR vs non-MCAR test...")

        if merged_data.empty:
            logger.warning("Merged data is empty, skipping missing test.")
            return

        missing = merged_data.isnull()
        observed = (~missing).astype(int)

        expected = pd.DataFrame(
            [observed.mean().values] * len(observed),
            columns=observed.columns,
            index=observed.index,
        )

        chi2, p, _, _ = chi2_contingency(
            [observed.to_numpy().sum(axis=0), expected.to_numpy().sum(axis=0)]
        )
        logger.info("Chi-squared test statistic: %.4f, p-value: %.4f", chi2, p)
        if p < 0.05:
            logger.info("MCAR rejected: missing data likely NOT MCAR.")
        else:
            logger.info("MCAR accepted: missing data likely MCAR.")

    # ---------- IMPUTATION ----------

    def impute_data(self, merged_data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Imputing missing values with forward fill and backward fill...")
        imputed = merged_data.ffill().bfill()
        remaining = imputed.isnull().sum().sum()
        if remaining > 0:
            logger.warning("Still %d missing values after imputation.", remaining)
        return imputed
