from typing import Callable, Dict, Optional

import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import wasserstein_distance
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt


class ModelTuningValidation:
    """
    Tuning + validation générique d'un modèle de forecast type scikit-learn.
    """

    def __init__(
        self,
        model,
        validation_data: pd.DataFrame,
        target_col: str,
        n_splits: int = 5,
        scoring: Callable[[np.ndarray, np.ndarray], float] = mean_squared_error,
    ):
        self.model = model
        self.validation_data = validation_data
        self.target_col = target_col
        self.n_splits = n_splits
        self.scoring = scoring

    def tune_model(self, n_trials: int = 50, param_space: Optional[Dict] = None) -> Dict:
        if param_space is None:
            raise ValueError("You must provide a param_space for tune_model().")

        X, y = self._get_X_y()

        def objective(trial):
            params = {}
            for name, spec in param_space.items():
                kind = spec[0]
                if kind == "int":
                    params[name] = trial.suggest_int(name, spec[1], spec[2])
                elif kind == "float":
                    params[name] = trial.suggest_float(name, spec[1], spec[2])
                elif kind == "categorical":
                    params[name] = trial.suggest_categorical(name, spec[1])
                else:
                    raise ValueError(f"Unknown param kind: {kind}")

            if hasattr(self.model, "set_params"):
                self.model.set_params(**params)

            score = self.cross_validate_model(X, y)
            return score

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        return study.best_params

    def cross_validate_model(self, X: pd.DataFrame, y: pd.Series) -> float:
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        scores = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)
            score = self.scoring(y_test, preds)
            scores.append(score)

        return float(np.mean(scores))

    def validate_model(self) -> Dict[str, float]:
        """
        Fit sur tout le dataset et calcul des métriques.
        Nettoie y/preds en vecteurs 1D avant Wasserstein.
        """
        X, y = self._get_X_y()

        self.model.fit(X, y)
        preds = self.model.predict(X)

        y_arr = np.asarray(y).ravel()
        preds_arr = np.asarray(preds).ravel()

        mse = mean_squared_error(y_arr, preds_arr)
        mae = mean_absolute_error(y_arr, preds_arr)
        rmse = float(np.sqrt(mse))
        wasserstein_dist = wasserstein_distance(y_arr, preds_arr)

        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": rmse,
            "wasserstein_distance": float(wasserstein_dist),
        }

    def visualize_validation_results(self, results: Dict[str, float]) -> None:
        metrics = list(results.keys())
        values = list(results.values())

        plt.figure(figsize=(8, 4))
        plt.bar(metrics, values)
        plt.title("Model Validation Metrics")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.show()

    def _get_X_y(self) -> tuple[pd.DataFrame, pd.Series]:
        if self.target_col not in self.validation_data.columns:
            raise ValueError(f"Target column '{self.target_col}' not in validation_data.")

        X = self.validation_data.drop(columns=[self.target_col])
        y = self.validation_data[self.target_col]
        return X, y
