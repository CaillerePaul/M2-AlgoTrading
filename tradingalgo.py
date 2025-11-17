import pandas as pd
import numpy as np
from src.data_loader import DataLoader
from src.preprocessing import Preprocessing
from src.eda import EDA
from src.LeadLagDTW import LeadLagDTW
from src.ModelTuningValidation import ModelTuningValidation
from datetime import datetime
from pprint import pprint

# --- Step 1: Initialize and Fetch Data ---
tickers = ["GC=F", "SI=F", "EURUSD=X", "JPYUSD=X", "^TNX"] # Gold, Silver, EUR/USD, JPY/USD, 10YUSTreasury
start_date = "2022-01-01"
end_date = "2023-12-31"

data_loader = DataLoader(tickers, start_date, end_date)
raw_data_dict = data_loader.fetch_data()

# --- Step 2: Merge Data ---
merged_data = data_loader.merge_data(raw_data_dict)

# --- Step 3: Perform EDA on Missing Values ---
data_loader.eda_missing(merged_data)
data_loader.missing_test(merged_data)

# --- Step 4: Clean and Impute Missing Values ---
cleaned_data = data_loader.impute_data(merged_data)

# --- Step 5: Preprocess the Data ---
preprocessor = Preprocessing(cleaned_data)

# Clean again (in case additional preprocessing is needed)
preprocessor.clean_data()

# --- Step 6: Feature Engineering (log returns, moving avg, scaling) ---
features = preprocessor.transform_data()
pprint(features.keys())  # Returns: 'returns', 'scaled_returns', 'moving_avg'

# --- Step 7: Apply a Filter (e.g., Savitzky-Golay) ---
smoothed_data = preprocessor.apply_filter(filter_type='savgol', window=11, polyorder=3)

# --- Step 8: Anomaly Detection ---
anomalies = preprocessor.anomaly_detection()

# --- Step 9: View Final Outputs ---
print("\n📊 Sample of Scaled Returns:")
print(features['scaled_returns'].head())

print("\n📉 Sample of Filtered Data (Savitzky-Golay):")
print(smoothed_data.head())

print("\n🚨 Detected Anomalies:")
print(anomalies.head())



# Final processed data (e.g., smoothed_data or scaled returns)
processed_data = features['scaled_returns']  # from Preprocessing.transform_data()

# Initialize EDA
eda = EDA(processed_data)

# Step-by-step analysis
eda.plot_timeseries()
eda.correlation_matrix()
eda.dtw_clustermap()
eda.seasonality_tracker()

dtw_model = LeadLagDTW(processed_data)
results = dtw_model.identify_lead_lag()
print(results)


# Hyperparameter tuning and model validation function
def tune_and_validate_model():
    """
    Tune and validate the Lead-Lag forecasting model using ModelTuningValidation.
    """
    # Load and preprocess data
    data = processed_data

    model = LeadLagDTW(data)

    results = []

    # Initialize ModelTuningValidation with model and validation data
    validation_data = data[['GC=F', 'SI=F']]  # Choose your validation columns or create a validation dataset
    tuning_validator = ModelTuningValidation(model, validation_data)
    
    # Hyperparameter tuning with Optuna (adjust parameter grid as necessary)
    param_grid = {
        'window_size': [10, 20, 30, 50],  # Example values, adjust based on model behavior
        'smoothing_factor': [0.1, 0.5, 1.0]  # Example values for smoothing factor
    }
    
    # Tune the model (this will run the optimization)
    best_params = tuning_validator.tune_model(param_grid)
    print(f"Best Hyperparameters: {best_params}")
    
    # Update the model with the best parameters
    model.set_params(**best_params)
    
    # Validate the model's performance
    validation_results = tuning_validator.validate_model()
    print(f"Model Validation Results: {validation_results}")
    
    # Visualize the results
    tuning_validator.visualize_validation_results(validation_results)

# Main function
def main():
    """
    The main function to run the trading algorithm, model tuning, and validation.
    """
    tune_and_validate_model()  # Tune and validate the model

if __name__ == "__main__":
    main()
