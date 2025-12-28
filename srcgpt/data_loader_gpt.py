import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
import missingno as msno
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

logger = logging.getLogger(__name__)

# ============================================================
# CONFIG : liste des tickers, dates, colonne de prix à utiliser
# ============================================================

@dataclass
class DataLoaderConfig:
    tickers: List[str]                  # Exemple : ["GC=F", "SI=F", "^TNX"]
    start_date: str                     # "YYYY-MM-DD"
    end_date: str                       # "YYYY-MM-DD"
    price_column: str = "Adj Close"  
    resample_rule: Optional[str] = None # "B" (business days), "D" (daily), "W" (weekly)



# ============================================================
# MAIN CLASS : FETCH → MERGE → MISSING EDA → IMPUTATION
# ============================================================

class DataLoader:
    """
    Étapes réalisées :
    1. Téléchargement via Yahoo Finance
    2. Sélection automatique de la bonne colonne prix
    3. Fusion Date x Ticker
    4. EDA sur les valeurs manquantes (missingno)
    5. Test statistique MCAR vs non-MCAR
    6. Imputation

    Objectifs :
    - Avoir un DataFrame propre pour tout le pipeline.
    - Standardiser les données multi-actifs sans surprises.
    """

    def __init__(self, config: DataLoaderConfig):
        self.config = config

    # ---------- FETCHING ----------

    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """
        Télécharge les données Yahoo Finance ticker par ticker.
        Renvoie un dict : {ticker: DataFrame complet OHLCV}
        """

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
        """
        Construit la matrice finale Date x Ticker avec UNE colonne par actif.
        Exemple :
             GC=F   SI=F   EURUSD=X ...
        date   ...    ...       ...
        """

        merged = pd.DataFrame()

        for ticker, df in data_dict.items():
            # Sélection automatique de la bonne colonne prix
            price_col = self._choose_price_column(df)

            # On convertit en série : rename = ticker
            series = df[[price_col]].rename(columns={price_col: ticker})

            if merged.empty:
                merged = series
            else:
                merged = merged.join(series, how="outer")

        return merged

    def _choose_price_column(self, df: pd.DataFrame) -> str:
        """
        Règle : 
        - si price_column existe → OK
        - sinon fallback sur "Close"
        - sinon on arrête (données suspectes)
        """
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
        """
        Affiche les matrices de valeurs manquantes (missingno).
        Aide à comprendre : 
        - Si un ticker commence plus tard
        - S'il manque des jours (jours fériés différents)
        - S'il y a un problème de source
        """

        logger.info("Visualizing missing data with missingno")

        if merged_data.empty:
            logger.warning("Merged data is empty, skipping missingno plots.")
            return

        # Missing matrix
        msno.matrix(merged_data)
        plt.title("Missing Data Matrix")
        if show:
            plt.show()
        else:
            plt.close()

        # Heatmap = corrélations dans les motifs de missing
        msno.heatmap(merged_data)
        plt.title("Missing Data Heatmap")
        if show:
            plt.show()
        else:
            plt.close()


    # MCAR vs NON-MCAR test (chi2)
    def missing_test(self, merged_data: pd.DataFrame) -> None:
        """
        Test statistique :
        - H0 = données manquantes complètement aléatoires (MCAR)
        - H1 = non-MCAR → structure → problème potentiellement sérieux
        """

        logger.info("Performing MCAR vs non-MCAR test...")

        if merged_data.empty:
            logger.warning("Merged data is empty, skipping missing test.")
            return

        # Tableau binaire "observé vs missing"
        missing = merged_data.isnull()
        observed = (~missing).astype(int)

        # expected = proportion moyenne observée
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
        """
        Imputation simple mais robuste :
        - forward fill (utilise la dernière valeur connue)
        - backward fill pour combler les trous au début

        Cette méthode est standard en finance pour des prix quotidiens
        car elle évite d'introduire du bruit ou d'interpoler inutilement.
        """

        logger.info("Imputing missing values with forward fill and backward fill...")

        imputed = merged_data.ffill().bfill()
        
        remaining = imputed.isnull().sum().sum()
        if remaining > 0:
            logger.warning("Still %d missing values after imputation.", remaining)
        return imputed
