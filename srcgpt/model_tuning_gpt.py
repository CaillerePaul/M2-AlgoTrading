# srcgpt/model_tuning_gpt.py

import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import wasserstein_distance

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Utility : build parameter dict from Optuna suggestion
# ------------------------------------------------------------

def sample_params(trial: optuna.Trial, param_space: Dict[str, tuple]) -> Dict[str, Any]:
    """
    Convertit un espace de recherche de type :
        { "max_depth": ("int", 2, 6),
          "learning_rate": ("float", 0.01, 0.2),
          "subsample": ("float", 0.5, 1.0),
          "booster": ("categorical", ["gbtree", "dart"])
        }

    En un dict de paramètres :
        { "max_depth": 4,
          "learning_rate": 0.13,
          "subsample": 0.91,
          "booster": "gbtree" }
    """
    params = {}

    for name, spec in param_space.items():
        ptype = spec[0]

        if ptype == "int":
            _, low, high = spec
            params[name] = trial.suggest_int(name, low, high)

        elif ptype == "float":
            _, low, high = spec
            params[name] = trial.suggest_float(name, low, high)

        elif ptype == "logfloat":
            _, low, high = spec
            params[name] = trial.suggest_float(name, low, high, log=True)

        elif ptype == "categorical":
            _, choices = spec
            params[name] = trial.suggest_categorical(name, choices)

        else:
            raise ValueError(f"Unknown param type '{ptype}' for param '{name}'")

    return params


# ------------------------------------------------------------
# MAIN CLASS : Model Tuning + Validation
# ------------------------------------------------------------

class ModelTuningValidation:
    """
    - Validation croisée temporelle (TimeSeriesSplit)
    - Tuning d'hyperparamètres via Optuna
    - Évaluation out-of-sample

    Paramètres
    ----------
    model : objet sklearn-like
        Modèle de base (XGBRegressor, Ridge, etc.)
    validation_data : pd.DataFrame
        Dataset supervisé : features + colonne de target.
    target_col : str
        Nom de la colonne cible dans validation_data.
    n_splits : int
        Nombre de splits TimeSeriesSplit pour la CV.
    base_params : dict, optionnel
        Paramètres de base **fixes** du modèle (objective, random_state, etc.)
        qui seront combinés avec les paramètres proposés par Optuna.
    model_name : str, optionnel
        Juste pour le logging (xgboost, ridge, etc.).
    """

    def __init__(
        self,
        model,
        validation_data: pd.DataFrame,
        target_col: str,
        n_splits: int = 5,
        base_params: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
    ):
        self.model = model
        self.data = validation_data
        self.target_col = target_col
        self.n_splits = n_splits
        self.base_params: Dict[str, Any] = base_params or {}
        self.model_name = model_name or model.__class__.__name__

        if target_col not in validation_data.columns:
            raise ValueError(f"Target column '{target_col}' not in provided dataset.")

    # --------------------------------------------------------
    # Function: Cross-validation score
    # --------------------------------------------------------
    def cross_val_score(self, params: Dict[str, Any]) -> float:
        """
        Retourne MSE moyen sur toutes les splits de TimeSeriesSplit.

        On combine :
        - les paramètres fixes de base (self.base_params),
        - les hyperparamètres proposés par Optuna (params).
        """
        X = self.data.drop(columns=[self.target_col]).values
        y = self.data[self.target_col].values

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        mse_scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # On merge base_params + params (les params Optuna
            # écrasent les valeurs de base s'il y a conflit).
            all_params = {**self.base_params, **params}

            model = self.model.__class__(**all_params)
            model.fit(X_train, y_train)

            preds = model.predict(X_val)
            mse_scores.append(mean_squared_error(y_val, preds))

        mean_mse = float(np.mean(mse_scores))
        logger.info(
            "CV MSE (model=%s): %.6f", self.model_name, mean_mse
        )
        return mean_mse

    # --------------------------------------------------------
    # Function: Hyperparameter tuning
    # --------------------------------------------------------
    def tune_model(
        self,
        n_trials: int,
        param_space: Dict[str, tuple],
        show_progress_bar: bool = False,
    ) -> Dict[str, Any]:
        """
        Tune les hyperparamètres via Optuna pour minimiser le MSE.
        """

        def objective(trial: optuna.Trial) -> float:
            params = sample_params(trial, param_space)
            return self.cross_val_score(params)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)

        logger.info("Best trial (model=%s): %.6f", self.model_name, study.best_value)
        logger.info("Best hyperparameters (model=%s): %s", self.model_name, study.best_params)

        return study.best_params

    # --------------------------------------------------------
    # Function: Final walk-forward validation report
    # --------------------------------------------------------
    def validate_model(self, model=None) -> Dict[str, float]:
        """
        Évalue les performances finales du modèle (déjà tuné) sur une
        dernière séparation temporelle 80% / 20%.
        """
        if model is None:
            model = self.model

        X = self.data.drop(columns=[self.target_col]).values
        y = self.data[self.target_col].values

        # Split temporel final : 80% train, 20% test
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        mse = float(mean_squared_error(y_val, preds))
        mae = float(mean_absolute_error(y_val, preds))
        rmse = float(np.sqrt(mse))
        wass = float(wasserstein_distance(y_val, preds))

        # Prédiction directionnelle (sign accuracy)
        sign_acc = float(np.mean(np.sign(preds) == np.sign(y_val)))

        results = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "wasserstein_distance": wass,
            "sign_accuracy": sign_acc,
        }

        logger.info("Validation results (model=%s): %s", self.model_name, results)
        return results
