"""Métricas de desempeño: retorno, riesgo, Sharpe, Sortino, Calmar, drawdown, win rate, turnover."""
import numpy as np
import pandas as pd
from scipy import stats

from src.backtest import FEE

BARS_PER_YEAR = 365 * 288  # BTC opera 24/7; barras de 5 minutos


def bar_returns(equity: pd.Series) -> pd.Series:
    """Retorno simple por barra."""
    return equity.pct_change().dropna()


def years_of_data(equity: pd.Series) -> float:
    """Tiempo operado en años, contando barras (no días de calendario), para no
    contar los huecos de datos como tiempo transcurrido."""
    return (len(equity) - 1) / BARS_PER_YEAR


def annualized_return(equity: pd.Series) -> float:
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years_of_data(equity)) - 1


def annualized_vol(equity: pd.Series) -> float:
    return bar_returns(equity).std() * np.sqrt(BARS_PER_YEAR)


def sharpe(equity: pd.Series) -> float:
    r = bar_returns(equity)
    return r.mean() / r.std() * np.sqrt(BARS_PER_YEAR)


def sortino(equity: pd.Series) -> float:
    r = bar_returns(equity)
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return r.mean() / downside * np.sqrt(BARS_PER_YEAR)


def drawdown_curve(equity: pd.Series) -> pd.Series:
    """Caída porcentual respecto al máximo previo (valores <= 0)."""
    return equity / equity.cummax() - 1


def max_drawdown(equity: pd.Series) -> float:
    return drawdown_curve(equity).min()


def calmar(equity: pd.Series) -> float:
    """Retorno anualizado entre |máximo drawdown|; NaN si no hubo drawdown."""
    mdd = abs(max_drawdown(equity))
    return annualized_return(equity) / mdd if mdd > 0 else float("nan")


def win_rate(trades: pd.DataFrame) -> float:
    """Fracción de operaciones con P&L neto positivo."""
    return float((trades["pnl"] > 0).mean()) if len(trades) else float("nan")


def annual_turnover(equity: pd.Series, traded_notional: float) -> float:
    """Nocional operado (entradas y salidas) por año, en múltiplos del equity medio."""
    return traded_notional / equity.mean() / years_of_data(equity)


def periodic_returns(equity: pd.Series, freq: str) -> pd.Series:
    """Retorno por periodo calendario ('M' mensual, 'Q' trimestral, 'Y' anual). Los periodos
    sin datos se saltan: cada retorno se mide contra el cierre del periodo anterior con datos,
    y el primero contra el valor inicial."""
    e = equity.copy()
    e.index = e.index.tz_localize(None)
    last = e.groupby(e.index.to_period(freq)).last()
    previous = np.r_[e.iloc[0], last.to_numpy()[:-1]]
    return pd.Series(last.to_numpy() / previous - 1, index=last.index, name=f"retorno_{freq}")


def trade_returns(trades: pd.DataFrame, fee: float = FEE) -> pd.DataFrame:
    """Agrega a la bitácora el retorno bruto y neto de comisiones de cada operación, en % del
    precio de entrada (fee es la comisión por lado; el lab fija 0.125%)."""
    t = trades.copy()
    t["bruto_pct"] = t["side"] * (t["exit_price"] / t["entry_price"] - 1) * 100
    t["neto_pct"] = t["bruto_pct"] - fee * 100 * (1 + t["exit_price"] / t["entry_price"])
    return t


def compare_groups(a, b, n_boot: int = 5000, seed: int = 42) -> dict:
    """Diferencia de medias entre dos muestras, con prueba de Welch e intervalo bootstrap del 95%."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    _, p_value = stats.ttest_ind(a, b, equal_var=False)
    rng = np.random.default_rng(seed)
    diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)]
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"diferencia_media": a.mean() - b.mean(), "p_welch": p_value, "ic95_bajo": low, "ic95_alto": high}


def market_exposure(result, close: pd.Series) -> dict:
    """Fracción de las barras con posición abierta y exposición (nocional / valor del portafolio)
    mientras la hay. Sirve para interpretar la volatilidad de una estrategia."""
    equity = result.equity
    price = close.reindex(equity.index)
    notional = pd.Series(0.0, index=equity.index)
    for t in result.trades.itertuples():
        held = (equity.index >= t.entry_time) & (equity.index <= t.exit_time)
        notional[held] = t.qty * price[held]
    invested = notional > 0
    ratio = (notional / equity)[invested]
    return {
        "barras_con_posicion_%": 100 * invested.mean(),
        "exposicion_media_%": 100 * ratio.mean() if invested.any() else 0.0,
        "exposicion_maxima_%": 100 * ratio.max() if invested.any() else 0.0,
    }


def summary(result) -> dict:
    """Todas las métricas de un BacktestResult en un diccionario."""
    eq = result.equity
    return {
        "retorno_total": eq.iloc[-1] / eq.iloc[0] - 1,
        "retorno_anual": annualized_return(eq),
        "volatilidad_anual": annualized_vol(eq),
        "sharpe": sharpe(eq),
        "sortino": sortino(eq),
        "max_drawdown": max_drawdown(eq),
        "calmar": calmar(eq),
        "n_operaciones": len(result.trades),
        "win_rate": win_rate(result.trades),
        "turnover_anual": annual_turnover(eq, result.traded_notional),
        "costos_totales": result.total_costs,
    }
