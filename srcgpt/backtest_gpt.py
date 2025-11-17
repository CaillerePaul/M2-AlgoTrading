from typing import Dict, Tuple

import numpy as np
import pandas as pd


def backtest_signals(
    target_returns: pd.Series | pd.DataFrame,
    signals: pd.Series | pd.DataFrame,
    initial_capital: float = 100_000.0,
    trading_cost_bps: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Backtest simple :
    - position_t = signal_{t-1}
    - ret_gross_t = position_t * return_t
    - coût à chaque changement de position.
    """

    # Assurer des Series 1D
    if isinstance(target_returns, pd.DataFrame):
        if target_returns.shape[1] != 1:
            raise ValueError(
                f"target_returns DataFrame must have 1 column, got {target_returns.shape[1]}"
            )
        target_returns = target_returns.iloc[:, 0]

    if isinstance(signals, pd.DataFrame):
        if signals.shape[1] != 1:
            raise ValueError(
                f"signals DataFrame must have 1 column, got {signals.shape[1]}"
            )
        signals = signals.iloc[:, 0]

    # Alignement index
    idx = target_returns.index.intersection(signals.index)
    r = target_returns.loc[idx].astype(float)
    s = signals.loc[idx].astype(float)

    position = s.shift(1).fillna(0.0)
    ret_gross = position * r

    cost_per_unit = trading_cost_bps / 10_000.0
    position_change = position.diff().abs().fillna(0.0)
    ret_cost = cost_per_unit * position_change

    ret_net = ret_gross - ret_cost
    equity = (1.0 + ret_net).cumprod() * initial_capital

    ret_gross = pd.Series(ret_gross, index=idx, name="ret_gross")
    ret_cost = pd.Series(ret_cost, index=idx, name="ret_cost")
    ret_net = pd.Series(ret_net, index=idx, name="ret_net")
    equity = pd.Series(equity, index=idx, name="equity")

    equity_curve = pd.DataFrame(
        {
            "ret_gross": ret_gross,
            "ret_cost": ret_cost,
            "ret_net": ret_net,
            "equity": equity,
        },
        index=idx,
    )

    stats = _compute_performance_stats(ret_net, equity)
    return equity_curve, stats


def _compute_performance_stats(ret_net: pd.Series, equity: pd.Series) -> Dict[str, float]:
    """
    Stats cohérentes :
    - total_return basé sur equity
    - CAGR basé sur equity
    - annual_return = mean_daily * 252
    - annual_vol = std_daily * sqrt(252)
    - Sharpe = annual_return / annual_vol
    - max_drawdown
    """
    n_days = len(ret_net)
    if n_days == 0:
        return {}

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    if equity.iloc[-1] > 0 and equity.iloc[0] > 0:
        cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (252 / n_days) - 1.0)
    else:
        cagr = -1.0

    mean_daily = ret_net.mean()
    std_daily = ret_net.std(ddof=0)

    ann_return = float(mean_daily * 252)
    ann_vol = float(std_daily * np.sqrt(252)) if std_daily > 0 else np.nan
    sharpe = float(ann_return / ann_vol) if ann_vol and not np.isnan(ann_vol) else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())

    stats = {
        "total_return": total_return,
        "cagr": cagr,
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
    }

    return stats
