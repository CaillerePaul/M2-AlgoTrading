# srcgpt/strategy_gpt.py

import numpy as np
import pandas as pd


def make_predictions_and_signals(
    model,
    supervised_df: pd.DataFrame,
    target_col: str,
    threshold: float = 0.0,
    min_holding_days: int = 1,
) -> pd.DataFrame:
    """
    Fit le modèle sur (X, y) et génère :
    - y_true
    - y_pred
    - signal brut (-1, 0, +1)
    - signal lissé (min_holding_days)

    min_holding_days:
        1 => pas de lissage (comportement classique),
        2+ => on évite de changer de position plus souvent que tous les N jours.
    """
    if target_col not in supervised_df.columns:
        raise ValueError(f"target_col '{target_col}' not in supervised_df columns.")

    X = supervised_df.drop(columns=[target_col])
    y = supervised_df[target_col]

    model.fit(X, y)
    preds = model.predict(X)

    y_arr = np.asarray(y).ravel()
    preds_arr = np.asarray(preds).ravel()

    # Signal brut
    raw_signal = np.where(
        preds_arr > threshold,
        1,
        np.where(preds_arr < -threshold, -1, 0),
    )

    # Lissage minimal : imposer min_holding_days
    if min_holding_days <= 1:
        final_signal = raw_signal
    else:
        final_signal = _apply_min_holding(raw_signal, min_holding_days)

    signals_df = pd.DataFrame(
        {
            "y_true": y_arr,
            "y_pred": preds_arr,
            "signal_raw": raw_signal,
            "signal": final_signal,
        },
        index=supervised_df.index,
    )

    return signals_df


def _apply_min_holding(signal: np.ndarray, min_holding_days: int) -> np.ndarray:
    """
    Applique une contrainte de min_holding_days sur une série de signaux (-1, 0, +1).
    L'idée :
    - On ne peut changer de position que si on a tenu la précédente
      au moins min_holding_days.
    """
    if len(signal) == 0:
        return signal

    current_pos = 0
    hold_counter = 0
    out = np.zeros_like(signal)

    for i, s in enumerate(signal):
        if current_pos == 0:
            # à plat : on peut ouvrir une position dès qu'un signal ≠ 0 apparaît
            if s != 0:
                current_pos = s
                hold_counter = 1
        else:
            # déjà en position
            if s == current_pos:
                # même direction => on continue
                hold_counter += 1
            elif s == 0:
                # signal neutre : on sort seulement si on a tenu assez longtemps
                if hold_counter >= min_holding_days:
                    current_pos = 0
                    hold_counter = 0
                else:
                    hold_counter += 1
            else:
                # changement de direction (long -> short ou l'inverse)
                # on ne change que si on a tenu assez longtemps
                if hold_counter >= min_holding_days:
                    current_pos = s
                    hold_counter = 1
                else:
                    hold_counter += 1

        out[i] = current_pos

    return out
