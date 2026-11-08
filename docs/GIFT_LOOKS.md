# Regards sur GIFT-Eval

GIFT-Eval n'entre dans aucune décision du protocole (sélection sur `dev_eval.py`).
Chaque évaluation est consignée ici, avec ce qu'elle a servi à décider.

| # | Date | Run | Réglage | CRPS | MASE | Usage |
|---|---|---|---|---|---|---|
| 1 | 2026-11-03 | base0 (20 000 pas, batch 256, corpus v5) | `+freq_delta`, sans flip | 0.5462 | 0.7834 | calibrer l'ambition ; aucune hypothèse choisie dessus |
| 2 | 2026-11-05 | causal0 (perte de contexte, `p_full` 0.5, 20 000 pas, v5) | `+freq_delta`, sans flip | 0.5233 | 0.7534 | mesurer le transfert du dev vers GIFT ; aucune hypothèse choisie dessus |
| 3 | 2026-11-05 | long0 (recette de causal0, corpus v7, 40 000 pas) | `+freq_delta`, sans flip | 0.5036 | 0.7290 | mesurer le gain de la durée et de v7 sur GIFT ; la durée du run final ne se choisit pas dessus |
| 3b | 2026-11-05 | long0, même checkpoint | `+freq_delta`, `+tta_flip`, `+clip_nonneg` | 0.4990 | 0.7223 | comparaison à réglage égal avec une config publiée qui utilise flip et clip ; rien choisi dessus |
| 4 | 2026-11-07 | ctrl_loop (S4D + attention, 2 passes, 9.98M, v8, WSD, 20 000 pas) | `+freq_delta`, sans flip | 0.5099 | 0.7322 | estimer le run final TimeSSM-10M, déjà lancé sur cette architecture ; rien choisi dessus |
| 5 | 2026-11-07 | 20 ablations à 20 000 pas (liste ci-dessous) | `+freq_delta`, sans flip | 0.5077 à 0.5409 | 0.7297 à 0.7828 | mesurer le lien dev → GIFT ; aucune hypothèse choisie dessus, le run final était déjà lancé |
| 6 | 2026-11-08 | TimeSSM-10M, checkpoints de plateau (LR constant) à 20 000 et 35 000 pas | `+freq_delta`, sans flip | 0.5132 / 0.5055 | 0.7369 / 0.7257 | mesurer l'apport de la décroissance WSD et de la durée au plateau ; run en cours, rien choisi dessus |
| 7 | 2026-11-08 | ctrl_loop | `+freq_delta` + flip ; puis + pile RateIN publiée (mix, pool, up 2x3x4, 4 fenêtres) | 0.5076 / 0.4986 | 0.7286 / 0.7213 | mesurer ce que RateIN apporte au modèle patché avant le run final ; rien choisi dessus |

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

Regard 4 contre le regard 3 : +1.3 % (dev : −0.8 %, 0.5427 contre 0.5469) ; contre le regard 2 :
−2.6 % pour −4.4 % au dev, environ 60 % du gain du dev passe. À 20 000 pas, l'architecture et v8 ne
rattrapent pas le doublement des pas de long0 : la durée passe mieux sur GIFT que le reste. Perte
sur les trois termes face à long0 (short +1.0 %, medium +1.5 %, long +1.6 %) ; 10S (+7.4 %) et 5T
(+3.4 %) en tête des pertes (bizitobs, bitbrains), A (−3.5 %) et 10T en gain. Couverture 80 % : 0.800,
q10 0.091 : la meilleure calibration des quatre regards.

Regard 5, CRPS GIFT (dev entre parenthèses) : ctrl_delta 0.5077 (0.5481), wsd0 0.5115 (0.5586),
lr1e3 0.5118 (0.5640), c7_0 0.5124 (0.5574), cl2_0 0.5129 (0.5608), c8_0 0.5134 (0.5567),
delta0 0.5141 (0.5599), s4d352_0 0.5150 (0.5597), dloop0 0.5157 (0.5739), c7_1 0.5161 (0.5667),
cm7_0 0.5164 (0.5651), delta1 0.5165 (0.5594), s4d5h6_0 0.5171 (0.5637), loop0 0.5174 (0.5572),
half0 0.5179 (0.5564), loop4_0 0.5195 (0.5609), causal1 0.5223 (0.5731), selh0 0.5234 (0.5649),
sel0 0.5286 (0.5790), cfree0 0.5409 (0.6239). Sur les 21 runs causaux à 20 000 pas avec perte de
contexte (regards 2, 4, 5) : corrélation 0.74 en log, pente 0.49 : 1 % de dev vaut 0.5 % de GIFT.
Écart de graine sur GIFT 0.2 à 0.7 % (causal0/1, delta0/1, c7_0/1), sous celui du dev. ctrl_loop et
ctrl_delta à égalité (0.4 %). Boucles −1.0 % sur causal0/1 (loop0), delta −1.5 % ; dloop0, rejeté
au dev, à 0.5157 : sa pénalité au dev ne passe pas. Sélectif mauvais sur les deux, rejet solide.
half0 +1.1 % sur c7_0 alors qu'il gagne au dev : le nombre de séries compte sur GIFT. La perte de
contexte reste le plus gros levier (cfree0 +3.4 % sur causal0). La durée seule (long0, 40 000 pas,
0.5036) bat toutes les variantes à 20 000 pas : c'est le levier que le run final exploite.

Regard 6 : le plateau de 20 000 pas et ctrl_loop sont le même run jusqu'à 16 000 pas (même graine,
même recette) ; la décroissance de ctrl_loop vaut −0.6 % sur GIFT (0.5132 → 0.5099) pour −1.7 % au
dev (0.5523 → 0.5427), le ratio de transfert habituel. Le plateau de 35 000 pas fait 0.5055, −1.5 %
sur celui de 20 000, alors que son dev (0.5520) est un creux du plateau : le dev bruité du plateau ne
dit rien du GIFT à ce stade. La durée passe mieux que la décroissance. Couverture 80 % 0.801-0.803.

Regard 7 : flip −0.45 % sur le brut (0.5099 → 0.5076). RateIN −1.8 % sur le flip (0.4986), contre −6 %
sur TimeSSM-2.5M : le contexte de 4096 pas laisse moins à gagner. Gain nul sur le court (+0.1 %, 55
configs), −4.5 % sur le moyen et −3.7 % sur le long ; 5T et 10S à −10 % (bizitobs_l2c/5T/long ×0.47),
W +6 % et A +2.8 % en perte (solar/W/short ×1.68, ett1/H ×1.12-1.15). 29 configs sur 97 surtout
décimées. Couverture 80 % 0.786, q10 0.104. Durée : 3 h 35 contre 26 min pour le flip seul, M4
en tête.
