"""Datos sintéticos compartidos por las pruebas y las pruebas de la regla de confirmación 2 de 3."""
import numpy as np
import pandas as pd

from src.backtest import BacktestParams, backtest
from src.signals import confirm_signal


def make_prices(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Precios OHLC sintéticos (caminata aleatoria) con un hueco de ~10.8 h que parte la serie en dos tramos."""
    rng = np.random.default_rng(seed)
    close = 30000 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.0005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.0005, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1.0},
        index=idx,
    )
    return df.drop(df.index[1000:1130])


def test_confirmation_rule():
    """Un indicador a favor no abre posición; dos o más sí."""
    signals = pd.DataFrame(
        [(1, 0, 0), (0, 0, -1), (1, -1, 0), (1, 1, 0), (1, 1, 1), (-1, -1, 0), (1, 1, -1)],
        columns=["a", "b", "c"],
    )
    expected = [0, 0, 0, 1, 1, -1, 1]
    assert confirm_signal(signals).tolist() == expected


def test_one_indicator_does_not_open_a_position_two_do():
    """En el motor: con un solo indicador a favor no se abre posición; con dos, sí."""
    idx = pd.date_range("2024-01-01", periods=4, freq="5min", tz="UTC")
    df = pd.DataFrame({"Open": 100.0, "High": 100.5, "Low": 99.5, "Close": 100.0}, index=idx)
    seg, atr = pd.Series(0, index=idx), pd.Series(1.0, index=idx)
    params = BacktestParams(m=2, r=3)
    one = confirm_signal(pd.DataFrame({"a": [1, 0, 0, 0], "b": 0, "c": 0}, index=idx))
    two = confirm_signal(pd.DataFrame({"a": [1, 0, 0, 0], "b": [1, 0, 0, 0], "c": 0}, index=idx))
    assert backtest(df, one, atr, seg, params).trades.empty
    assert len(backtest(df, two, atr, seg, params).trades) == 1
