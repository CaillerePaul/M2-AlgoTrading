import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import wasserstein_distance
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

class ModelTuningValidation:
    def __init__(self, model, validation_data):
        """
        Initialize the ModelTuningValidation instance with the forecasting model and validation dataset.
        
        Parameters:
        - model: The forecasting model instance (e.g., LeadLagDTW with chosen hyperparameters).
        - validation_data: DataFrame containing the validation dataset for performance evaluation.
        """
        self.model = model
        self.validation_data = validation_data

    def tune_model(self):
        """
        Optimize model parameters using a hyperparameter search technique.
        
        Parameters:
        - param_grid: Dictionary with parameter names as keys and lists of possible values to explore.
        
        Returns:
        - best_params: Best hyperparameters found by the optimization process.
        """
        # Define the objective function for Optuna
        def objective(trial):
            # Example: Optuna optimization for model parameters
            # Here, we could tune hyperparameters like window size, constraints, etc.
            window_size = trial.suggest_int('window_size', 10, 50)
            smoothing_factor = trial.suggest_uniform('smoothing_factor', 0.1, 1.0)
            
            # Example: Set model parameters
            self.model.set_params(window_size=window_size, smoothing_factor=smoothing_factor)
            
            # Perform cross-validation or single validation run
            score = self.cross_validate_model()
            return score
        
        # Set up the Optuna study and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        
        # Return the best hyperparameters found
        best_params = study.best_params
        return best_params
    
    def cross_validate_model(self):
        """
        Perform cross-validation to test the model's robustness over different time periods or data subsets.
        
        Returns:
        - average_error: The average error metric (e.g., MSE, MAE) across all cross-validation folds.
        """
        # Implement TimeSeriesSplit cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        errors = []
        
        for train_index, test_index in tscv.split(self.validation_data):
            train, test = self.validation_data.iloc[train_index], self.validation_data.iloc[test_index]
            # Fit model on training data and make predictions on test data
            self.model.fit(train)
            predictions = self.model.predict(test)
            
            # Compute error metrics (MSE, MAE, RMSE)
            mse = mean_squared_error(test, predictions)
            mae = mean_absolute_error(test, predictions)
            rmse = np.sqrt(mse)
            
            # Store error metrics for cross-validation evaluation
            errors.append({'mse': mse, 'mae': mae, 'rmse': rmse})
        
        # Average errors over all folds
        avg_errors = pd.DataFrame(errors).mean()
        return avg_errors['mse']  # Objective: Minimize MSE
    
    def validate_model(self):
        """
        Compute performance metrics to validate the model's forecasts.
        
        Returns:
        - results: Dictionary with error metrics (MSE, MAE, RMSE, Wasserstein distance).
        """
        # Split validation data into features and target (e.g., leading and lagging asset)
        features, target = self.validation_data.iloc[:, :-1], self.validation_data.iloc[:, -1]
        
        # Predict using the model
        self.model.fit(features)
        predictions = self.model.predict(target)
        
        # Traditional error metrics
        mse = mean_squared_error(target, predictions)
        mae = mean_absolute_error(target, predictions)
        rmse = np.sqrt(mse)
        
        # Compute Wasserstein distance (distributional difference)
        wasserstein_dist = wasserstein_distance(target, predictions)
        
        # Return the error metrics
        results = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'wasserstein_distance': wasserstein_dist
        }
        
        return results
    
    def visualize_validation_results(self, results):
        """
        Visualize the validation results with diagnostic plots comparing forecast errors.
        
        Parameters:
        - results: Dictionary with error metrics (MSE, MAE, RMSE, Wasserstein distance).
        """
        # Plot error distributions (MSE, MAE, RMSE)
        errors = pd.DataFrame(results)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(errors['mse'], bins=20, color='blue', alpha=0.7, label='MSE')
        plt.title('MSE Distribution')
        plt.xlabel('MSE')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.hist(errors['wasserstein_distance'], bins=20, color='green', alpha=0.7, label='Wasserstein Distance')
        plt.title('Wasserstein Distance Distribution')
        plt.xlabel('Wasserstein Distance')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()

# Example usage:
# Assuming you have a model instance and validation data
# model = LeadLagDTW(window_size=20, smoothing_factor=0.5)
# validation_data = your_validation_data
# tuning_validator = ModelTuningValidation(model, validation_data)
# best_params = tuning_validator.tune_model(param_grid)
# validation_results = tuning_validator.validate_model()
# tuning_validator.visualize_validation_results(validation_results)
