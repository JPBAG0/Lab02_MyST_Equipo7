"""Prueba de causalidad de las etiquetas de régimen (nivel B)."""
from src.data import segment_ids
from src.regimes import RuleRegimes, hold_labels, regime_features
from tests.test_signals import make_prices

WINDOW = 120


def test_labels_are_causal():
    """Con umbrales fijos, la etiqueta en t no cambia al agregar datos posteriores a t."""
    df = make_prices()
    feats_full = regime_features(df, segment_ids(df), WINDOW)
    rules = RuleRegimes.fit(feats_full.iloc[:600])  # umbrales ajustados solo con el pasado
    full = hold_labels(rules.classify(feats_full), segment_ids(df))
    assert (full != -1).sum() > 500, "la prueba sería vacía sin etiquetas"
    for t in (400, 900, 1100, 1500, 1800):
        cut = df.iloc[: t + 1]
        seg_cut = segment_ids(cut)
        part = hold_labels(rules.classify(regime_features(cut, seg_cut, WINDOW)), seg_cut)
        assert part.iloc[-1] == full.iloc[t], f"etiqueta distinta en t={t}"
