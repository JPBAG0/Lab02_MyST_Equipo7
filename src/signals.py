import numpy as np
import pandas as pd

from src.data import segment_ids
from dataclasses import dataclass

MIN_AGREE = 2  # regla de confirmación fijada por el lab: 2 de 3


def _direction(index, up, down) -> pd.Series:
    """Convierte dos condiciones booleanas en una señal +1 / -1 / 0."""
    return pd.Series(np.select([up, down], [1, -1], default=0), index=index)


def _per_segment(series: pd.Series, seg: pd.Series, fn) -> pd.Series:
    """Aplica fn a cada tramo por separado."""
    return series.groupby(seg).transform(fn)


def ema(close: pd.Series, seg: pd.Series, span: int) -> pd.Series:
    """Media móvil exponencial; NaN hasta tener `span` barras en el tramo."""
    return _per_segment(
        close, seg, lambda s: s.ewm(span=span, adjust=False, min_periods=span).mean()
    )


def atr(df: pd.DataFrame, seg: pd.Series, n: int) -> pd.Series:
    """Average True Range (media simple de n barras) en unidades de precio."""
    prev_close = df["Close"].groupby(seg).shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _per_segment(true_range, seg, lambda s: s.rolling(n).mean())


def rsi(close: pd.Series, seg: pd.Series, n: int) -> pd.Series:
    """RSI de Wilder, calculado dentro de cada tramo."""
    delta = close.groupby(seg).diff()
    wilder = lambda s: s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_gain = _per_segment(delta.clip(lower=0), seg, wilder)
    avg_loss = _per_segment(-delta.clip(upper=0), seg, wilder)
    return 100 - 100 / (1 + avg_gain / avg_loss)


def ema_cross_signal(df, seg, fast: int = 12, slow: int = 48) -> pd.Series:
    """Tendencia: +1 si la EMA rápida está sobre la lenta, -1 si está debajo."""
    f, s = ema(df["Close"], seg, fast), ema(df["Close"], seg, slow)
    return _direction(df.index, f > s, f < s)


def donchian_signal(df, seg, n: int = 48) -> pd.Series:
    """Tendencia: dirección de la última ruptura del canal de n barras previas."""
    hi = _per_segment(df["High"], seg, lambda s: s.rolling(n).max().shift(1))
    lo = _per_segment(df["Low"], seg, lambda s: s.rolling(n).min().shift(1))
    breakout = pd.Series(
        np.select([df["Close"] > hi, df["Close"] < lo], [1.0, -1.0], default=np.nan),
        index=df.index,
    )
    return breakout.groupby(seg).ffill().fillna(0).astype(int)


def rsi_signal(df, seg, n: int = 14, upper: float = 55, lower: float = 45) -> pd.Series:
    """Momento: +1 si RSI > upper, -1 si RSI < lower, 0 en medio."""
    r = rsi(df["Close"], seg, n)
    return _direction(df.index, r > upper, r < lower)


def roc_signal(df, seg, n: int = 12) -> pd.Series:
    """Momento: signo del retorno de las últimas n barras."""
    roc = df["Close"].groupby(seg).pct_change(n)
    return _direction(df.index, roc > 0, roc < 0)


def keltner_signal(df, seg, n_ema: int = 20, n_atr: int = 14, k: float = 2.0) -> pd.Series:
    """Volatilidad: +1 sobre la banda EMA + k*ATR, -1 bajo EMA - k*ATR, 0 dentro."""
    mid = ema(df["Close"], seg, n_ema)
    band = k * atr(df, seg, n_atr)
    return _direction(df.index, df["Close"] > mid + band, df["Close"] < mid - band)


def candidate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Las 5 señales candidatas con parámetros por defecto (sin optimizar)."""
    seg = segment_ids(df)
    return pd.DataFrame(
        {
            "ema_cross": ema_cross_signal(df, seg),
            "donchian": donchian_signal(df, seg),
            "rsi": rsi_signal(df, seg),
            "roc": roc_signal(df, seg),
            "keltner": keltner_signal(df, seg),
        }
    )


def confirm_signal(signals: pd.DataFrame, min_agree: int = MIN_AGREE) -> pd.Series:
    """Regla de confirmación: +1 si al menos min_agree indicadores son +1,
    -1 si al menos min_agree son -1, 0 en otro caso. Exige exactamente 3 indicadores."""
    if signals.shape[1] != 3:
        raise ValueError("La regla de confirmación se define para 3 indicadores")
    longs = (signals == 1).sum(axis=1)
    shorts = (signals == -1).sum(axis=1)
    return pd.Series(
        np.select([longs >= min_agree, shorts >= min_agree], [1, -1], default=0),
        index=signals.index,
    )


@dataclass(frozen=True)
class SignalParams:
    """Ventanas de los tres indicadores elegidos (valores por defecto, sin optimizar)."""

    n_donchian: int = 48
    n_roc: int = 12
    n_ema: int = 20
    n_atr: int = 14
    k: float = 2.0


def strategy_signal(df: pd.DataFrame, seg: pd.Series, p: SignalParams):
    """Pipeline de señal: tres indicadores -> regla 2 de 3. Devuelve (señal, ATR)."""
    three = pd.DataFrame(
        {
            "donchian": donchian_signal(df, seg, p.n_donchian),
            "roc": roc_signal(df, seg, p.n_roc),
            "keltner": keltner_signal(df, seg, p.n_ema, p.n_atr, p.k),
        }
    )
    return confirm_signal(three), atr(df, seg, p.n_atr)

