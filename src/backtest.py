"""Motor de backtest: posición, sizing, regla de salida, estado del portafolio y backtest()."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data import segment_ids
from src.signals import SignalParams, strategy_signal

FEE = 0.00125  # comisión por operación (entrada y salida), fijada por el lab
INITIAL_CASH = 1_000_000.0  # capital inicial, fijado por el lab


@dataclass(frozen=True)
class Position:
    """Posición abierta. side: +1 largo, -1 corto. qty en unidades de BTC."""

    side: int
    qty: float
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: pd.Timestamp
    regime: int = 0  # régimen vigente cuando se abrió la posición

    @classmethod
    def from_atr(cls, side, qty, entry_price, atr, m, r, entry_time, regime=0):
        """SL = entrada -/+ m*ATR y TP = entrada +/- r*m*ATR (según el lado)."""
        stop_dist = m * atr
        return cls(
            side,
            qty,
            entry_price,
            entry_price - side * stop_dist,
            entry_price + side * r * stop_dist,
            entry_time,
            regime,
        )


def size_position(equity, price, stop_distance, risk_frac, fee=FEE) -> float:
    """Unidades a operar: el riesgo en dólares (risk_frac * equity) dividido entre la
    distancia al stop, limitado para que el nocional (con comisión) no exceda el
    capital, porque el lab no permite apalancamiento."""
    qty_risk = risk_frac * equity / stop_distance
    qty_cap = equity / (price * (1 + fee))
    return min(qty_risk, qty_cap)


def check_exit(pos: Position, bar_open, bar_high, bar_low):
    """Devuelve (motivo, precio de salida) si la barra cierra la posición, o None.

    Primero se revisan los gaps (si la barra abre más allá de un nivel, se llena a la
    apertura). Si el stop y el target caen dentro de la misma barra, gana el stop
    (convención conservadora)."""
    if pos.side == 1:
        if bar_open <= pos.stop_loss:
            return "stop_loss", bar_open
        if bar_open >= pos.take_profit:
            return "take_profit", bar_open
        if bar_low <= pos.stop_loss:
            return "stop_loss", pos.stop_loss
        if bar_high >= pos.take_profit:
            return "take_profit", pos.take_profit
    else:
        if bar_open >= pos.stop_loss:
            return "stop_loss", bar_open
        if bar_open <= pos.take_profit:
            return "take_profit", bar_open
        if bar_high >= pos.stop_loss:
            return "stop_loss", pos.stop_loss
        if bar_low <= pos.take_profit:
            return "take_profit", pos.take_profit
    return None


class Portfolio:
    """Estado explícito: efectivo, a lo más una posición abierta y bitácora de operaciones.

    Contabilidad simétrica: abrir un largo resta el nocional del efectivo y abrir un
    corto lo suma; la posición se valúa en side * qty * precio. La comisión se cobra
    en cada apertura y cada cierre."""

    def __init__(self, cash: float, fee: float = FEE):
        self.cash = cash
        self.fee = fee
        self.position: Position | None = None
        self.trades: list[dict] = []
        self.total_costs = 0.0
        self.traded_notional = 0.0
        self._entry_cost = 0.0

    def _trade_cash(self, side: int, qty: float, price: float, opening: bool) -> float:
        """Mueve el efectivo por una operación y devuelve la comisión pagada."""
        notional = qty * price
        cost = self.fee * notional
        direction = -side if opening else side
        self.cash += direction * notional - cost
        self.total_costs += cost
        self.traded_notional += notional
        return cost

    def open(self, pos: Position) -> None:
        if self.position is not None:
            raise RuntimeError("Ya hay una posición abierta")
        self._entry_cost = self._trade_cash(pos.side, pos.qty, pos.entry_price, True)
        self.position = pos

    def close(self, price: float, time: pd.Timestamp, reason: str) -> None:
        pos = self.position
        if pos is None:
            raise RuntimeError("No hay posición abierta")
        exit_cost = self._trade_cash(pos.side, pos.qty, price, False)
        self.trades.append(
            {
                "entry_time": pos.entry_time,
                "exit_time": time,
                "side": pos.side,
                "qty": pos.qty,
                "entry_price": pos.entry_price,
                "exit_price": price,
                "reason": reason,
                "regime": pos.regime,
                "pnl": pos.side * pos.qty * (price - pos.entry_price)
                - self._entry_cost
                - exit_cost,
            }
        )
        self.position = None

    def equity(self, mark_price: float) -> float:
        """Valor del portafolio: efectivo más el valor de la posición abierta."""
        pos_value = self.position.side * self.position.qty * mark_price if self.position else 0.0
        return self.cash + pos_value


@dataclass(frozen=True)
class BacktestParams:
    """Parámetros del motor (no de los indicadores)."""

    m: float  # stop = m * ATR
    r: float  # target = r * stop
    risk_frac: float = 0.01  # presupuesto de riesgo por operación (fracción del equity)
    max_hold: int = 288  # holding máximo en barras (288 = 24 h de barras de 5 min)
    signal_exit_after: int = 0  # la señal opuesta solo cierra tras esta cantidad de barras
    min_tp_pct: float = 0.005  # filtro de viabilidad: distancia al target >= 0.5% del precio
    cash: float = INITIAL_CASH
    fee: float = FEE


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    total_costs: float
    traded_notional: float


@dataclass(frozen=True)
class RegimeSpec:
    """Señal, ATR y parámetros del motor que se usan mientras el régimen vigente sea éste."""

    signal: pd.Series
    atr: pd.Series
    params: BacktestParams


def backtest_regimes(
    df,
    seg,
    labels,
    specs: dict,
    start: int = 0,
) -> BacktestResult:
    """Backtest event-driven, sin estado global, con parámetros distintos por régimen.

    `labels` es la etiqueta de régimen al cierre de cada barra y `specs` asigna a cada
    régimen su RegimeSpec; un régimen sin spec es un régimen en el que no se opera.
    La señal y la etiqueta de la barra t-1 (calculadas al cierre) se ejecutan en la
    apertura de t. Orden de eventos dentro de cada barra t:
      1. Salida a la apertura: si cambió el régimen, por señal opuesta o por holding máximo. La posición conserva los parámetros del régimen
         en que se abrió.
      2. Entrada a la apertura si no hay posición, el régimen vigente tiene spec y su
         señal es distinta de 0, con los parámetros de ese régimen.
      3. Stop-loss / take-profit intrabarra (el stop gana si ambos caen en la barra).
      4. Cierre forzado al cierre de la última barra de cada tramo continuo.
    Ninguna entrada usa señal ni etiqueta de un tramo anterior. No se abre ninguna
    posición antes de la barra `start` (sirve para dejar un periodo de calentamiento)."""
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    seg_v = seg.to_numpy()
    lab = labels.to_numpy()
    sig_by = {r: sp.signal.to_numpy() for r, sp in specs.items()}
    atr_by = {r: sp.atr.to_numpy(dtype=float) for r, sp in specs.items()}
    times = df.index
    n = len(df)
    first = next(iter(specs.values())).params
    pf = Portfolio(first.cash, first.fee)
    equity = np.empty(n)
    entry_i, pos_regime = -1, None

    for i in range(n):
        same_segment = i > 0 and seg_v[i] == seg_v[i - 1]
        reg = lab[i - 1] if same_segment else None

        pos = pf.position
        if pos is not None:
            pp = specs[pos_regime].params
            own_sig = sig_by[pos_regime][i - 1] if same_segment else 0
            if reg != pos_regime:
                pf.close(o[i], times[i], "regime_change")
            elif own_sig == -pos.side and i - entry_i >= pp.signal_exit_after:
                pf.close(o[i], times[i], "signal")
            elif i - entry_i >= pp.max_hold:
                pf.close(o[i], times[i], "max_hold")

        if i >= start and pf.position is None and reg in specs:
            sp = specs[reg].params
            prev_sig = sig_by[reg][i - 1]
            prev_atr = atr_by[reg][i - 1]
            if prev_sig != 0 and np.isfinite(prev_atr) and prev_atr > 0:
                stop_dist = sp.m * prev_atr
                if sp.r * stop_dist / o[i] >= sp.min_tp_pct:
                    qty = size_position(pf.cash, o[i], stop_dist, sp.risk_frac, sp.fee)
                    pf.open(
                        Position.from_atr(int(prev_sig), qty, o[i], prev_atr, sp.m, sp.r, times[i], reg)
                    )
                    entry_i, pos_regime = i, reg

        pos = pf.position
        if pos is not None:
            hit = check_exit(pos, o[i], h[i], l[i])
            if hit is not None:
                pf.close(hit[1], times[i], hit[0])

        if pf.position is not None and (i == n - 1 or seg_v[i + 1] != seg_v[i]):
            pf.close(c[i], times[i], "segment_end")

        equity[i] = pf.equity(c[i])

    return BacktestResult(
        equity=pd.Series(equity, index=df.index, name="equity"),
        trades=pd.DataFrame(pf.trades),
        total_costs=pf.total_costs,
        traded_notional=pf.traded_notional,
    )


def backtest(df, signal, atr, seg, params: BacktestParams, start: int = 0) -> BacktestResult:
    """Backtest con un único conjunto de parámetros (un solo régimen). Ver backtest_regimes."""
    labels = pd.Series(0, index=df.index)
    return backtest_regimes(df, seg, labels, {0: RegimeSpec(signal, atr, params)}, start)


def run_strategy(
    df, signal_params: SignalParams, params: BacktestParams, start: int = 0
) -> BacktestResult:
    """Pipeline completo: tramos -> indicadores -> señal 2 de 3 -> backtest."""
    seg = segment_ids(df)
    signal, atr_series = strategy_signal(df, seg, signal_params)
    return backtest(df, signal, atr_series, seg, params, start)

