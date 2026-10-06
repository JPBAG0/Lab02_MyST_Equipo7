"""Pruebas del recorte de periodos: intervalo semiabierto y rechazo de huecos."""
import pandas as pd
import pytest

from src.data import PERIODS, check_continuous, select_period
from tests.test_signals import make_prices


def test_select_period_is_half_open():
    """Incluye la barra de inicio y excluye la barra de fin."""
    df = make_prices()
    start, end = df.index[100], df.index[112]
    part = select_period(df, str(start), str(end))
    assert part.index[0] == start and part.index[-1] == df.index[111] and len(part) == 12


def test_check_continuous_rejects_gaps():
    """make_prices tiene un hueco de ~10.8 h: el periodo completo se rechaza y un tramo sin hueco pasa."""
    df = make_prices()
    with pytest.raises(ValueError):
        check_continuous(df, "con hueco")
    check_continuous(df.iloc[:900], "sin hueco")


def test_work_periods_are_ordered_and_disjoint():
    """Train, test y validación van en ese orden y no se traslapan."""
    bounds = [(pd.Timestamp(start), pd.Timestamp(end)) for _, start, end in PERIODS.values()]
    assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(bounds, bounds[1:]))
