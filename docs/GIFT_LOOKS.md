# Regards sur GIFT-Eval

GIFT-Eval n'entre dans aucune décision du protocole (sélection sur `dev_eval.py`).
Chaque évaluation est consignée ici, avec ce qu'elle a servi à décider.

| # | Date | Run | Réglage | CRPS | MASE | Usage |
|---|---|---|---|---|---|---|
| 1 | 2026-11-03 | base0 (20 000 pas, batch 256, corpus v5) | `+freq_delta`, sans flip | 0.5462 | 0.7834 | calibrer l'ambition ; aucune hypothèse choisie dessus |
| 2 | 2026-11-05 | causal0 (perte de contexte, `p_full` 0.5, 20 000 pas, v5) | `+freq_delta`, sans flip | 0.5233 | 0.7534 | mesurer le transfert du dev vers GIFT ; aucune hypothèse choisie dessus |
| 3 | 2026-11-05 | long0 (recette de causal0, corpus v7, 40 000 pas) | `+freq_delta`, sans flip | 0.5036 | 0.7290 | mesurer le gain de la durée et de v7 sur GIFT ; la durée du run final ne se choisit pas dessus |
| 3b | 2026-11-05 | long0, même checkpoint | `+freq_delta`, `+tta_flip`, `+clip_nonneg` | 0.4990 | 0.7223 | comparaison à réglage égal avec une config publiée qui utilise flip et clip ; rien choisi dessus |

Regard 1 contre TimeSSM-2.5M nu (0.5451) : écart total +0.2 % en log. Gains sur les horizons
longs (−1.4 %) et moyens (−0.8 %), bizitobs en tête ; perte sur les horizons courts (+2.4 %).

Regard 2 contre le regard 1 : −4.2 % en log (dev : −9.3 % contre la moyenne des bases), donc
environ la moitié du gain du dev passe sur GIFT. Gain sur les trois termes, le court en tête
(short −5.0 %, medium −3.7 %, long −2.7 %) ; D seule fréquence sans gain (+0.5 %) ;
covid_deaths/D/short (×1.47) et solar/W/short (×1.21) en perte. Couverture 80 % : 0.792.

Regard 3 contre le regard 2 : −3.8 % (dev : −3.7 %, 0.5469 contre 0.5678), le gain du dev passe
presque entier, contre la moitié au regard 2. Gain sur les trois termes (short −4.4 %, medium
−3.1 %, long −2.8 %) ; D (−6.1 %) et M (−6.2 %) en tête, us_births et covid_deaths/D/short
(×0.73) regagnent ; ett2/D/short seul en perte nette (×1.11). Couverture 80 % : 0.792.

Regard 3b : flip + clip à zéro, −0.9 % sur le brut. La couverture du quantile 0.1 passe à 0.182 :
le clip pose q10 à 0 sur les séries intermittentes, et les cibles nulles comptent comme sous q10.
