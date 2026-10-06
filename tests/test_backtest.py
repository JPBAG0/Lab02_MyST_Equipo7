"""Pruebas del motor: convención stop-primero y contabilidad."""
import pandas as pd
import pytest

from src.backtest import FEE, Portfolio, Position, check_exit

T0 = pd.Timestamp("2024-01-01", tz="UTC")


def test_stop_wins_when_both_levels_in_same_bar():
    """Largo con stop y target dentro de la misma barra: se cierra como stop_loss."""
    long_pos = Position(1, 1.0, 100.0, 95.0, 105.0, T0)
    assert check_exit(long_pos, 100.0, 106.0, 94.0) == ("stop_loss", 95.0)
    short_pos = Position(-1, 1.0, 100.0, 105.0, 95.0, T0)
    assert check_exit(short_pos, 100.0, 106.0, 94.0) == ("stop_loss", 105.0)


def test_accounting_matches_trades():
    """Sin posición abierta, el valor es el efectivo y refleja el P&L neto de comisiones;
    los costos totales son la comisión por el nocional operado."""
    p = Portfolio(1_000_000)
    p.open(Position(1, 2.0, 100.0, 95.0, 105.0, T0))
    p.close(110.0, T0, "take_profit")
    expected_pnl = 2.0 * 10.0 - FEE * (2.0 * 100.0 + 2.0 * 110.0)
    assert p.trades[0]["pnl"] == pytest.approx(expected_pnl)
    assert p.equity(110.0) == pytest.approx(1_000_000 + expected_pnl)
    assert p.total_costs == pytest.approx(FEE * p.traded_notional)


def test_equity_with_open_position_and_costs_per_fill():
    """Con una posición abierta, el valor es efectivo + side*qty*precio; los costos suman
    la comisión de cada apertura y cada cierre."""
    p = Portfolio(1_000_000)
    p.open(Position(-1, 3.0, 100.0, 105.0, 95.0, T0))
    assert p.equity(102.0) == pytest.approx(p.cash - 3.0 * 102.0)
    assert p.equity(102.0) == pytest.approx(1_000_000 - FEE * 300.0 - 3.0 * 2.0)
    p.close(102.0, T0, "signal")
    p.open(Position(1, 1.0, 102.0, 97.0, 110.0, T0))
    assert p.total_costs == pytest.approx(FEE * (300.0 + 306.0 + 102.0))
