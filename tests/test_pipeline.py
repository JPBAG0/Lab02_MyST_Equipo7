"""Prueba de truncamiento del pipeline completo (indicadores -> señal -> backtest)."""
import pandas as pd

from src.backtest import BacktestParams, run_strategy
from src.data import segment_ids
from src.optimize import run_regime_strategy, to_params
from src.signals import SignalParams, strategy_signal
from tests.test_evaluate import PARAMS
from tests.test_signals import make_prices

SP = SignalParams(n_donchian=30, n_roc=10, n_ema=15, n_atr=10, k=1.0)
BP = BacktestParams(m=4, r=2, max_hold=100)


def test_pipeline_is_causal():
    """Recalcular sobre df[:t+1] no cambia nada de lo ocurrido antes de t ni la señal en t."""
    df = make_prices()
    full = run_strategy(df, SP, BP)
    assert len(full.trades) >= 3, "la prueba sería vacía sin operaciones"
    full_signal, _ = strategy_signal(df, segment_ids(df), SP)

    for t in (400, 900, 1100, 1500, 1800):
        cut = df.iloc[: t + 1]
        cut_signal, _ = strategy_signal(cut, segment_ids(cut), SP)
        assert cut_signal.iloc[-1] == full_signal.iloc[t], f"señal distinta en t={t}"

        part = run_strategy(cut, SP, BP)
        pd.testing.assert_series_equal(part.equity.iloc[:-1], full.equity.iloc[:t])

        t_time = df.index[t]
        done_full = full.trades[full.trades["exit_time"] < t_time].reset_index(drop=True)
        done_cut = part.trades[part.trades["exit_time"] < t_time].reset_index(drop=True)
        pd.testing.assert_frame_equal(done_cut, done_full)


def test_regime_strategy_with_one_regime_matches_single_theta():
    """Con un solo régimen, run_regime_strategy reproduce exactamente el motor de un solo theta."""
    df = make_prices()
    p = {**PARAMS, "signal_exit_after": 0}
    seg, labels = segment_ids(df), pd.Series(0, index=df.index)
    by_regime = run_regime_strategy(df, seg, labels, {0: p})
    sp, bp = to_params(p)
    single = run_strategy(df, sp, bp)
    assert len(single.trades) > 0, "la prueba sería vacía sin operaciones"
    pd.testing.assert_series_equal(by_regime.equity, single.equity)
