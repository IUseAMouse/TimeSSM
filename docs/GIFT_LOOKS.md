# Regards sur GIFT-Eval

GIFT-Eval n'entre dans aucune décision du protocole (sélection sur `dev_eval.py`).
Chaque évaluation est consignée ici, avec ce qu'elle a servi à décider.

| # | Date | Run | Réglage | CRPS | MASE | Usage |
|---|---|---|---|---|---|---|
| 1 | 2026-11-03 | base0 (20 000 pas, batch 256, corpus v5) | `+freq_delta`, sans flip | 0.5462 | 0.7834 | calibrer l'ambition ; aucune hypothèse choisie dessus |
| 2 | 2026-11-05 | causal0 (perte de contexte, `p_full` 0.5, 20 000 pas, v5) | `+freq_delta`, sans flip | 0.5233 | 0.7534 | mesurer le transfert du dev vers GIFT ; aucune hypothèse choisie dessus |

Regard 1 contre TimeSSM-2.5M nu (0.5451) : écart total +0.2 % en log. Gains sur les horizons
longs (−1.4 %) et moyens (−0.8 %), bizitobs en tête ; perte sur les horizons courts (+2.4 %).

Regard 2 contre le regard 1 : −4.2 % en log (dev : −9.3 % contre la moyenne des bases), donc
environ la moitié du gain du dev passe sur GIFT. Gain sur les trois termes, le court en tête
(short −5.0 %, medium −3.7 %, long −2.7 %) ; D seule fréquence sans gain (+0.5 %) ;
covid_deaths/D/short (×1.47) et solar/W/short (×1.21) en perte. Couverture 80 % : 0.792.
