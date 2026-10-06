"""Punto de entrada: `python main.py` corre todo el proyecto y deja tablas en docs/tables y figuras
en docs/figures. `python main.py --quick` usa pocas pruebas por ventana para comprobar que todo
corre; sus resultados no son los del reporte y no tocan docs/theta_frozen.json."""
import argparse
import itertools
import json
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import plots
from src.backtest import INITIAL_CASH
from src.data import (
    PERIODS,
    audit_prices,
    check_continuous,
    check_no_overlap,
    clean_prices,
    load_prices,
    segment_ids,
    select_period,
    validate_prices,
)
from src.metrics import compare_groups, market_exposure, periodic_returns, summary, trade_returns
from src.optimize import (
    N_TRIALS,
    SEED,
    breakeven_fee,
    confirmation_effect,
    cost_curve,
    evaluate_frozen,
    freeze_theta,
    make_folds,
    sensitivity,
    walk_forward,
    walk_forward_regimes,
)
from src.regimes import NAMES, RuleRegimes, hold_labels, regime_features, regime_report
from src.signals import candidate_signals

QUICK_TRIALS = 8
LAB_FEE_BPS = 12.5  # comisión por lado fijada por el lab
DATA, DOCS = Path("data"), Path("docs")
TABLES, FIGURES = DOCS / "tables", DOCS / "figures"
SET_TITLES = {  # título de cada conjunto en las figuras
    "train": "Entrenamiento: Jul-Nov 2023 (fuera de muestra)",
    "test": "Prueba: Dic 2023",
    "validacion": "Validación: May-Jun 2024",
}
SET_FILES = {"train": "entrenamiento", "test": "prueba", "validacion": "validacion"}
START = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - START:6.0f} s] {message}", flush=True)


def save_table(frame: pd.DataFrame, name: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / f"{name}.csv")


def _default(obj):
    return obj.item() if hasattr(obj, "item") else str(obj)


def jsonable(obj):
    """Convierte tipos de numpy y pandas a tipos que JSON acepta."""
    return json.loads(json.dumps(obj, default=_default))


def to_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_default))


def metrics_table(results: dict) -> pd.DataFrame:
    return pd.DataFrame({name: summary(r) for name, r in results.items()}).T


def save_periodic_returns(results: dict, dataset: str) -> None:
    for name, r in results.items():
        for freq, label in (("M", "mensual"), ("Q", "trimestral"), ("Y", "anual")):
            save_table(periodic_returns(r.equity, freq).to_frame("retorno"), f"retornos_{label}_{dataset}_{name}")


def prepare_data() -> dict:
    """Audita los dos archivos crudos y recorta los tres periodos de trabajo (train, test y validación),
    que deben ser continuos, estar ordenados y no traslaparse."""
    raw = {"train": load_prices(DATA / "btc_project_train.csv"), "test": load_prices(DATA / "btc_project_test.csv")}
    save_table(pd.DataFrame({name: audit_prices(df) for name, df in raw.items()}).astype(str), "auditoria_datos")
    clean = {name: clean_prices(df) for name, df in raw.items()}
    periods = {name: select_period(clean[source], start, end) for name, (source, start, end) in PERIODS.items()}
    for name, d in periods.items():
        validate_prices(d)
        check_continuous(d, name)
    for before, after in itertools.pairwise(periods.values()):
        check_no_overlap(before, after)
    summary_rows = {
        name: {"archivo": PERIODS[name][0], "primera_barra": d.index[0], "ultima_barra": d.index[-1],
               "barras": len(d), "dias_con_datos": round(len(d) / 288, 1)}
        for name, d in periods.items()
    }
    save_table(pd.DataFrame(summary_rows).T, "periodos")
    return periods


def indicator_selection(train: pd.DataFrame) -> None:
    """Elección de indicadores sin mirar retornos: correlación entre las cinco señales candidatas
    (parámetros por defecto) y correlación media y máxima de cada trío con un indicador por familia
    (tendencia, momento y volatilidad)."""
    corr = candidate_signals(train).corr()
    save_table(corr, "correlacion_indicadores")
    families = (("ema_cross", "donchian"), ("rsi", "roc"), ("keltner",))
    rows = {}
    for trio in itertools.product(*families):
        pairs = [corr.loc[a, b] for a, b in itertools.combinations(trio, 2)]
        rows[" + ".join(trio)] = {"correlacion_media": np.mean(pairs), "correlacion_maxima": np.max(pairs)}
    save_table(pd.DataFrame(rows).T, "correlacion_trios")


def train_walk_forward(train, feats, seg, n_trials):
    """Walk-forward semanal sobre train: un solo theta (con gate) y un theta por régimen."""
    folds = make_folds(seg)
    log(f"Walk-forward con un solo theta: {len(folds)} ventanas x {n_trials} pruebas")
    table_single, oos_single, bench = walk_forward(train, folds, n_trials, gate=0.0, verbose=True)
    log("Walk-forward con un theta por régimen")
    table_regimes, oos_regimes, _ = walk_forward_regimes(train, feats, seg, folds, n_trials, verbose=True)
    save_table(table_single, "walk_forward_theta_unico")
    save_table(table_regimes, "walk_forward_theta_por_regimen")
    results = {"theta_unico": oos_single, "theta_por_regimen": oos_regimes, "comprar_y_mantener": bench}
    save_table(metrics_table(results), "metricas_train")
    save_periodic_returns(results, "train")
    return table_single, table_regimes, results


def regime_differentiation(trades: pd.DataFrame, dataset: str = "train") -> dict:
    """Pregunta 5: desempeño por régimen de entrada (operaciones, win rate, retorno por operación,
    contribución al capital) y prueba entre reversión y tendencia. Guarda también cómo terminan
    las operaciones (stop, target, señal, cambio de régimen, etc.)."""
    if trades.empty:
        return {}
    t = trade_returns(trades)
    t["regimen"] = t["regime"].map(NAMES)
    t["pnl_pct_capital"] = t["pnl"] / INITIAL_CASH * 100  # % del capital con que arrancó la semana
    g = t.groupby("regimen")
    table = pd.DataFrame(
        {
            "operaciones": g.size(),
            "win_rate": g["pnl"].apply(lambda s: (s > 0).mean()),
            "bruto_medio_pct": g["bruto_pct"].mean(),
            "neto_medio_pct": g["neto_pct"].mean(),
            "neto_desv_pct": g["neto_pct"].std(),
            "razon_media_desv": g["neto_pct"].mean() / g["neto_pct"].std(),
            "contribucion_capital_pct": g["pnl_pct_capital"].sum(),
        }
    )
    save_table(table, f"desempeno_por_regimen_{dataset}")
    save_table(t["reason"].value_counts().to_frame("operaciones"), f"salidas_por_motivo_{dataset}")
    a = t.loc[t["regimen"] == "reversion", "bruto_pct"]
    b = t.loc[t["regimen"] == "tendencia", "bruto_pct"]
    return compare_groups(a, b) if min(len(a), len(b)) > 1 else {}


def freeze_and_evaluate(train, test, validation, feats, seg, n_trials, quick):
    """Optimiza theta una sola vez sobre train y lo evalúa en test y en validación. La evaluación usa
    el theta congelado y commiteado (docs/theta_frozen.json); main.py solo recalcula para verificarlo."""
    log("Recalculando theta sobre todo el train")
    recomputed = freeze_theta(train, feats, seg, n_trials=n_trials)
    to_json(recomputed, DOCS / "theta_recomputed.json")
    frozen_file = DOCS / "theta_frozen.json"
    if frozen_file.exists() and not quick:
        theta = json.loads(frozen_file.read_text())
        same = jsonable(recomputed) == theta
        log("theta recalculado " + ("IGUAL" if same else "DISTINTO") + " al congelado (docs/theta_frozen.json)")
    else:
        theta = jsonable(recomputed)
    log("Evaluando en test y en validación con theta congelado")
    evaluations = {"test": (train, test), "validacion": (pd.concat([train, test]), validation)}
    results = {}
    for name, (history, target) in evaluations.items():
        results[name] = evaluate_frozen(history, target, theta)
        save_table(metrics_table(results[name]), f"metricas_{name}")
        exposure = {k: market_exposure(r, target["Close"]) for k, r in results[name].items() if k != "comprar_y_mantener"}
        save_table(pd.DataFrame(exposure).T, f"exposicion_{name}")
        save_periodic_returns(results[name], name)
    return theta, results


def robustness(train, seg, feats, theta):
    """Preguntas 1, 3 y 4 con los theta congelados de cada régimen, sobre train (dentro de muestra).
    Se hacen con la estrategia por régimen porque el theta único no pasó el filtro de calidad."""
    thetas = {code: theta["regimes"][name]["params"] for code, name in NAMES.items()
              if theta["regimes"][name]["params"] is not None}
    if not thetas:
        log("ningún régimen tiene parámetros (gate no cumplido): se omite el análisis de robustez")
        return {}
    log("Robustez: 2 de 3 contra un indicador, sensibilidad de ±20% y curva de costos")
    rules = RuleRegimes(theta["rules"]["vol_crisis"], theta["rules"]["r2_trend"])
    labels = hold_labels(rules.classify(feats), seg)
    effect = confirmation_effect(train, seg, labels, thetas)
    sens = sensitivity(train, seg, labels, thetas)
    curve = cost_curve(train, seg, labels, thetas)
    equilibrio = breakeven_fee(curve)
    save_table(effect, "pregunta1_dos_de_tres")
    save_table(sens, "pregunta3_sensibilidad")
    save_table(curve, "pregunta4_curva_de_costos")
    plots.plot_sensitivity(sens, FIGURES / "fig4_sensibilidad.png")
    plots.plot_cost_curve(curve, equilibrio, LAB_FEE_BPS, FIGURES / "fig5_costos.png")
    return {"comision_equilibrio_bps_por_lado": equilibrio}


def regime_analysis(periods, theta):
    """Estabilidad del régimen en cada conjunto (umbrales congelados) y etiquetas para las figuras."""
    full = pd.concat(list(periods.values()))
    seg = segment_ids(full)
    feats = regime_features(full, seg)
    rules = RuleRegimes(theta["rules"]["vol_crisis"], theta["rules"]["r2_trend"])
    labels = hold_labels(rules.classify(feats), seg)
    start = 0
    for name, part in periods.items():
        rows = slice(start, start + len(part))
        start += len(part)
        table, extra = regime_report(feats.iloc[rows], labels.iloc[rows], seg.iloc[rows])
        save_table(table, f"regimenes_{name}")
        save_table(pd.Series(extra, name="valor").to_frame(), f"regimenes_{name}_global")
    return feats, labels


def make_figures(results, periods, feats, labels):
    """`results` asigna a cada conjunto (train, test, validacion) los resultados de las tres estrategias."""
    log("Generando figuras")
    names = {"theta_unico": "θ único", "theta_por_regimen": "θ por régimen", "comprar_y_mantener": "Comprar y mantener"}
    curves = {SET_TITLES[k]: {names[m]: r.equity for m, r in res.items()} for k, res in results.items()}
    plots.plot_equity(curves, FIGURES / "fig1_valor_portafolio.png")
    plots.plot_drawdown(curves, FIGURES / "fig2_drawdown.png")
    for k, res in results.items():
        plots.plot_returns_table(res["theta_por_regimen"].equity, f"θ por régimen, {SET_TITLES[k]}",
                                 FIGURES / f"fig3_retornos_{SET_FILES[k]}.png")
    plots.plot_regime_timeline({SET_TITLES[k]: d["Close"] for k, d in periods.items()}, labels,
                               FIGURES / "fig6a_regimenes_precio.png")
    plots.plot_feature_distributions(feats, labels, FIGURES / "fig6b_distribuciones.png")
    for k, res in results.items():
        plots.plot_equity_with_regimes(res["theta_por_regimen"].equity, labels, f"θ por régimen, {SET_TITLES[k]}",
                                       FIGURES / f"fig6c_portafolio_regimenes_{SET_FILES[k]}.png")


def main() -> None:
    global TABLES, FIGURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="pocas pruebas por ventana (solo para verificar)")
    quick = parser.parse_args().quick
    if quick:  # no sobrescribe las tablas ni las figuras del reporte
        out = Path(tempfile.gettempdir()) / "lab02_quick"
        TABLES, FIGURES = out / "tables", out / "figures"
    random.seed(SEED)
    np.random.seed(SEED)
    n_trials = QUICK_TRIALS if quick else N_TRIALS

    log("Cargando, limpiando y auditando datos")
    periods = prepare_data()
    train, test, validation = periods["train"], periods["test"], periods["validacion"]
    seg = segment_ids(train)
    feats = regime_features(train, seg)
    indicator_selection(train)

    table_single, table_regimes, train_results = train_walk_forward(train, feats, seg, n_trials)
    summary_numbers = {
        "pruebas_por_ventana": n_trials,
        "ventanas": len(table_single),
        "calmar_entrenamiento_mediano": float(table_single["is_calmar"].median()),
        "calmar_fuera_de_muestra": float(summary(train_results["theta_unico"])["calmar"]),
        "diferenciacion_reversion_vs_tendencia": regime_differentiation(train_results["theta_por_regimen"].trades),
    }
    theta, eval_results = freeze_and_evaluate(train, test, validation, feats, seg, n_trials, quick)
    for name, res in eval_results.items():
        regime_differentiation(res["theta_por_regimen"].trades, name)
    regimes_fitted = table_regimes[["is_reversion", "is_tendencia", "is_crisis"]].notna().sum().sum()
    frozen_fits = 1 + sum(r["calmar"] is not None for r in theta["regimes"].values())
    summary_numbers.update(
        {
            "configuraciones_walk_forward_theta_unico": n_trials * len(table_single),
            "configuraciones_walk_forward_por_regimen": n_trials * int(regimes_fitted),
            "configuraciones_theta_congelado": n_trials * frozen_fits,
        }
    )
    summary_numbers.update(robustness(train, seg, feats, theta))
    full_feats, labels = regime_analysis(periods, theta)
    make_figures({"train": train_results, **eval_results}, periods, full_feats, labels)
    summary_numbers["configuraciones_totales"] = sum(v for k, v in summary_numbers.items() if k.startswith("configuraciones_"))
    summary_numbers["tiempo_total_s"] = round(time.time() - START)
    to_json(summary_numbers, TABLES / "resumen.json")
    log("Listo. Tablas en docs/tables y figuras en docs/figures")


if __name__ == "__main__":
    main()
