# scripts/download_data.py

import logging
from pathlib import Path
from srcgpt.data_loader_gpt import DataLoader, DataLoaderConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/prices_raw.csv")

TICKERS = ["GC=F", "SI=F", "EURUSD=X", "JPYUSD=X", "^TNX"]
START_DATE = "2022-01-01"
END_DATE   = "2023-12-31"


def main():
    logger.info("Downloading raw data from Yahoo Finance...")

    cfg = DataLoaderConfig(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        price_column="Adj Close",
        resample_rule=None,
    )
    loader = DataLoader(cfg)

    raw = loader.fetch_data()
    merged = loader.merge_data(raw)

    Path("data").mkdir(exist_ok=True)
    merged.to_csv(DATA_PATH, index=True)

    logger.info(f"Saved dataset to {DATA_PATH} — shape={merged.shape}")


if __name__ == "__main__":
    main()
