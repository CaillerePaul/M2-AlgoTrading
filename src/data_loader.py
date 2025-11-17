import yfinance as yf
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

class DataLoader:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        data = {}
        for ticker in self.tickers:
            print(f"Fetching data for {ticker}")
            df = yf.download(ticker, start=self.start_date, end=self.end_date)
            df['Ticker'] = ticker
            data[ticker] = df
        return data

    def merge_data(self, data_dict):
        merged = pd.DataFrame()
        for ticker, df in data_dict.items():
            df = df[['Close']].rename(columns={'Close': ticker})
            if merged.empty:
                merged = df
            else:
                merged = merged.join(df, how='outer')
        return merged

    def eda_missing(self, merged_data):
        print("Visualizing missing data with missingno...")
        msno.matrix(merged_data)
        plt.title("Missing Data Matrix")
        plt.show()

        msno.heatmap(merged_data)
        plt.title("Missing Data Heatmap")
        plt.show()

    def missing_test(self, merged_data):
        print("Performing MCAR vs NMAR test...")
        missing = merged_data.isnull()
        observed = (~missing).astype(int)

        # Expectation under MCAR: same chance of being observed across rows
        expected = pd.DataFrame([observed.mean().values] * len(merged_data), columns=observed.columns)

        chi2, p, _, _ = chi2_contingency([observed.sum(), expected.sum()])
        print(f"Chi-squared test statistic: {chi2:.4f}, p-value: {p:.4f}")
        if p < 0.05:
            print("Missing data is likely *not* missing completely at random (MCAR rejected).")
        else:
            print("Missing data is likely missing completely at random (MCAR accepted).")

    def impute_data(self, merged_data):
        print("Imputing missing values with forward fill and backward fill...")
        imputed = merged_data.ffill().bfill()
        return imputed