# 📈 Algo Trading Pipeline

This project provides a modular pipeline for fetching, cleaning, analyzing, and visualizing financial time series data.

## 📦 Setup Instructions

1. **Create a Virtual Environment**:
    ```bash
    python -m venv algotrading_env
    ```

2. **Activate the Virtual Environment**:

    - **Windows**:
        ```bash
        .\algotrading_env\Scripts\Activate.ps1
        ```

    - **macOS/Linux**:
        ```bash
        source algotrading_env/bin/activate
        ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Run the Trading Algorithm

Once the environment is set up, you can run the main trading algorithm by executing:

```bash
python tradingalgo.py
```

## 📁 Project Structure
- **`src/`**: Contains the core Python modules such as `DataLoader`, `Preprocessing`, and `EDA` for data processing and analysis.
  - **`data_loader.py`**: The `DataLoader` class.
  - **`preprocessing.py`**: The `Preprocessing` class.
  - **`eda.py`**: The `EDA` class.
