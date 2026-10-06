"""Detección de régimen de mercado: variables sobre ventana móvil de 1 semana y clasificador por reglas."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

WEEK = 7 * 288  # 1 semana en barras de 5 minutos
HOUR = 12  # 1 hora en barras de 5 minutos


def regime_features(df: pd.DataFrame, seg: pd.Series, window: int = WEEK, lag: int = HOUR) -> pd.DataFrame:
    """Tres variables de régimen, causales y calculadas dentro de cada tramo continuo:

    vol          volatilidad realizada: desviación estándar de los retornos logarítmicos.
    trend_r2     fuerza de tendencia: R^2 de la regresión lineal del log-precio contra el
                 tiempo (cercano a 1 si el precio avanza en línea recta).
    autocorr_1h  autocorrelación entre el retorno de la última hora y el de la hora anterior
                 (negativa si los movimientos tienden a revertirse). Se usa retorno horario y
                 no de 5 minutos porque en estos datos la autocorrelación de 5 minutos es
                 positiva solo en el rezago 1, un efecto de cómo se construyen las barras.
                 Solo se usa para describir los regímenes, no para clasificarlos.

    Son NaN hasta completar la primera ventana de cada tramo."""
    parts = []
    for _, d in df.groupby(seg, sort=False):
        logp = np.log(d["Close"])
        ret = logp.diff()
        ret_h = logp.diff(lag)
        t = pd.Series(np.arange(len(d), dtype=float), index=d.index)
        parts.append(
            pd.DataFrame(
                {
                    "vol": ret.rolling(window).std(),
                    "trend_r2": logp.rolling(window).corr(t) ** 2,
                    "autocorr_1h": ret_h.rolling(window).corr(ret_h.shift(lag)),
                }
            )
        )
    return pd.concat(parts)


# ---------------------------------------------------------------------------
# Clasificador por reglas, actualización de etiquetas y validación
# ---------------------------------------------------------------------------
REVERSION, TREND, CRISIS, UNLABELED = 0, 1, 2, -1
NAMES = {REVERSION: "reversion", TREND: "tendencia", CRISIS: "crisis"}
UPDATE_EVERY = 12  # la etiqueta se actualiza cada hora (12 barras de 5 min)
BAR_HOURS = 5 / 60


@dataclass(frozen=True)
class RuleRegimes:
    """Régimen por reglas con umbrales fijos, ajustados una sola vez en train:

        crisis     si vol >= vol_crisis (cuantil alto de la volatilidad de train)
        tendencia  si no es crisis y trend_r2 >= r2_trend
        reversion  en otro caso (comportamiento de rango)

    Solo usa vol y trend_r2."""

    vol_crisis: float
    r2_trend: float = 0.5

    @classmethod
    def fit(cls, feats: pd.DataFrame, q_crisis: float = 0.90, r2_trend: float = 0.5) -> "RuleRegimes":
        return cls(float(feats["vol"].quantile(q_crisis)), r2_trend)

    def classify(self, feats: pd.DataFrame) -> pd.Series:
        label = np.where(
            feats["vol"] >= self.vol_crisis,
            CRISIS,
            np.where(feats["trend_r2"] >= self.r2_trend, TREND, REVERSION),
        )
        missing = feats[["vol", "trend_r2"]].isna().any(axis=1)
        return pd.Series(np.where(missing, UNLABELED, label), index=feats.index)


def hold_labels(labels: pd.Series, seg: pd.Series, every: int = UPDATE_EVERY) -> pd.Series:
    """Actualiza la etiqueta cada `every` barras (contadas desde el inicio de cada tramo)
    y la mantiene constante entre actualizaciones."""
    pos = labels.groupby(seg).cumcount()
    held = labels.where(pos % every == 0).groupby(seg).ffill()
    return held.fillna(UNLABELED).astype(int)


def run_lengths(labels: pd.Series, seg: pd.Series) -> pd.DataFrame:
    """Rachas consecutivas de una misma etiqueta (sin cruzar tramos): etiqueta y duración en barras."""
    new_run = (labels != labels.shift()) | (seg != seg.shift())
    runs = labels.groupby(new_run.cumsum()).agg(["first", "size"])
    return runs.rename(columns={"first": "label", "size": "bars"})


def regime_report(feats: pd.DataFrame, labels: pd.Series, seg: pd.Series, cols=("vol", "trend_r2"),
                  sample: int = 10_000, seed: int = 42):
    """Validación del régimen: participación y duración por régimen, transiciones por mes
    y silhouette (sobre las variables estandarizadas, con submuestra por costo)."""
    valid = labels != UNLABELED
    runs = run_lengths(labels, seg)
    runs = runs[runs["label"] != UNLABELED]
    rows = {}
    for code, name in NAMES.items():
        r = runs[runs["label"] == code]["bars"] * BAR_HOURS
        rows[name] = {
            "participacion_%": 100 * (labels[valid] == code).mean(),
            "duracion_media_h": r.mean(),
            "duracion_mediana_h": r.median(),
            "rachas": len(r),
        }
    prev = labels.shift()
    change = (labels != prev) & (seg == seg.shift()) & valid & (prev != UNLABELED)
    months = valid.sum() / (288 * 30)
    x = feats.loc[valid, list(cols)]
    x = (x - x.mean()) / x.std()
    sil = silhouette_score(x, labels[valid], sample_size=min(sample, len(x)), random_state=seed)
    return pd.DataFrame(rows).T, {"transiciones_por_mes": change.sum() / months, "silhouette": sil}
