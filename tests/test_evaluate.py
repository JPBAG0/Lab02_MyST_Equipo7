"""Prueba de la evaluación final: el test no usa datos futuros ni abre posiciones antes de empezar."""
import pandas as pd

from src.optimize import NEVER, evaluate_frozen
from tests.test_signals import make_prices

PARAMS = {
    "n_donchian": 30, "n_roc": 10, "n_ema": 15, "n_atr": 10, "k": 1.0,
    "m": 4.0, "r": 2.0, "max_hold": 100, "signal_exit_after": NEVER, "risk_frac": 0.01,
}
THETA = {
    "single": {"params": PARAMS},
    "rules": {"vol_crisis": 0.0016, "r2_trend": 0.3},
    "regimes": {
        "reversion": {"params": PARAMS},
        "tendencia": {"params": {**PARAMS, "n_donchian": 60, "m": 6.0}},
        "crisis": {"params": None},
    },
}
STRATEGIES = ("theta_unico", "theta_por_regimen")


def closed_before(result, t_time):
    """Operaciones ya cerradas antes de t_time (vacío si la estrategia no operó)."""
    trades = result.trades
    return trades if trades.empty else trades[trades["exit_time"] < t_time].reset_index(drop=True)


def test_evaluation_does_not_use_data_after_t():
    """Recortar el test en t no cambia el equity previo a t ni las operaciones ya cerradas, y
    ninguna operación se abre antes del primer dato de test."""
    df = make_prices(12000)
    train, test = df.iloc[:7000], df.iloc[7000:]
    full = evaluate_frozen(train, test, THETA)
    for key in STRATEGIES:
        assert len(full[key].trades) > 0, f"la prueba sería vacía sin operaciones ({key})"
        assert (full[key].trades["entry_time"] >= test.index[0]).all()
    for t in (800, 2000, 3500):
        part = evaluate_frozen(train, test.iloc[: t + 1], THETA)
        for key in STRATEGIES:
            pd.testing.assert_series_equal(part[key].equity.iloc[:-1], full[key].equity.iloc[:t])
            done_full, done_part = closed_before(full[key], test.index[t]), closed_before(part[key], test.index[t])
            assert len(done_part) == len(done_full)
            if len(done_full):
                pd.testing.assert_frame_equal(done_part, done_full)
