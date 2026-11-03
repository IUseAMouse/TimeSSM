# Regards sur GIFT-Eval

GIFT-Eval n'entre dans aucune décision du protocole (sélection sur `dev_eval.py`).
Chaque évaluation est consignée ici, avec ce qu'elle a servi à décider.

| # | Date | Run | Réglage | CRPS | MASE | Usage |
|---|---|---|---|---|---|---|
| 1 | 2026-11-03 | base0 (20 000 pas, batch 256, corpus v5) | `+freq_delta`, sans flip | 0.5462 | 0.7834 | calibrer l'ambition ; aucune hypothèse choisie dessus |

Regard 1 contre TimeSSM-2.5M nu (0.5451) : écart total +0.2 % en log. Gains sur les horizons
longs (−1.4 %) et moyens (−0.8 %), bizitobs en tête ; perte sur les horizons courts (+2.4 %).
