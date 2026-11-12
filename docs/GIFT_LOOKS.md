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
| 8 | 2026-11-08 | TimeSSM-10M, plateau à 60 000 et 100 000 pas | `+freq_delta`, sans flip | 0.5002 / 0.4946 | 0.7215 / 0.7173 | pente GIFT du plateau, pour décider de prolonger le run avant la décroissance (160 000 pas) |
| 9 | 2026-11-09 | TimeSSM-10M, plateau à 120 000, 140 000 et 160 000 pas | `+freq_delta`, sans flip | 0.4992 / 0.5005 / 0.4963 | 0.7201 / 0.7205 / 0.7184 | pente GIFT du plateau ouvert, pour décider quand l'arrêter (choix de l'utilisateur : arrêt au vu de cette courbe) |
| 10 | 2026-11-09 | TimeSSM-10M, branche d160k (plateau jusqu'à 160 000 pas, décroissance WSD de 40 000 pas, modèle à 200 000) | brut ; flip ; flip + clip ; pile RateIN publiée ; pile RateIN + clip | 0.4934 / 0.4904 / 0.4896 / 0.4763 / 0.4759 | 0.7154 / 0.7118 / 0.7112 / 0.6948 / 0.6947 | série complète du modèle livrable ; rien choisi dessus, le corpus v9 était déjà en préparation |
| 11 | 2026-11-10 | Hypothèse 13 à 240 000 pas : bras v9 ; bras v8 (d200k) ; un modèle externe de 7M paramètres passé par le même harnais, comme repère | v9 : brut ; flip ; pile RateIN publiée. d200k : brut. Externe : brut ; flip | 0.4933 / 0.4905 / 0.4804 ; 0.4922 ; 0.4849 / 0.4824 | 0.7160 / 0.7123 / 0.6995 ; 0.7153 ; 0.7047 / 0.7011 | l'hypothèse 13 était déjà tranchée au dev (v9 retenu) ; rien choisi dessus. Le levier suivant (contexte 8192) est choisi sur le dev |
| 12 | 2026-11-11 | Contexte 8192 (c8k, branche de décroissance de v9 à 200k, 40k pas) ; la pile RateIN du modèle externe de 7M | c8k : brut ; flip ; pile RateIN publiée. Externe : pile RateIN publiée | 0.4871 / 0.4855 / 0.4809 ; 0.4814 | 0.7081 / 0.7043 / 0.6992 ; — | le dev (0.5302 contre 0.5268 pour v9) ne retenait pas c8k ; rien choisi sur ce regard. La décision 8192/4096 reste au dev et au coût |
| 13 | 2026-11-12 | Bras corpus v10 (v9 à 200k, décroissance de 40k pas sur v10) | Pile RateIN publiée | 0.4782 | 0.6974 | v10 retenu au dev avant ce regard (0.5263 contre 0.5268 pour v9, critère du déclencheur) ; rien choisi sur GIFT |

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

Regard 8 : courbe du plateau 0.5132 / 0.5055 / 0.5002 / 0.4946 à 20k / 35k / 60k / 100k, soit −1.9 %,
−1.35 % et −1.5 % par doublement des pas : pas de saturation visible. Couverture 80 % 0.803-0.806.

Regard 9 : 0.4946 / 0.4992 / 0.5005 / 0.4963 à 100k / 120k / 140k / 160k. La pente de −1.5 % par
doublement s'arrête après 100k : de 100k à 160k, l'écart reste dans ±1 %, ce qui correspond au bruit
d'un itéré pris à LR constant. Le dev du plateau stagne aussi depuis 105k (0.5265 puis 0.5315 à
0.5347). Au LR de 5e-4, le plateau n'apprend plus rien que GIFT voie. La suite de la courbe se lit
sur les branches décrues, pas sur le plateau. Couverture 80 % 0.801-0.802.

Regard 10 : la décroissance vaut −0.6 % sur le plateau de 160k (0.4963 → 0.4934), comme à 20k ; elle ne
gagne que 0.2 % sur le meilleur plateau (100k, 0.4946). Le flip apporte −0.6 % et RateIN −2.9 % sur le flip
(0.4904 → 0.4763), contre −1.8 % sur ctrl_loop (regard 7) : le modèle long tire plus de la pile. Le clip
apporte −0.1 %, mais q10 passe à 0.188 et l'intervalle 80 % à 0.706. Couverture de la pile sans clip :
q10 0.101, q90 0.894, intervalle 80 % 0.793. Repères de la copie du classement du 2026-09-06 : TiRex-2-Pretrained
(82.5M) 0.4669, TiRex-2-Zeroshot (82.5M) 0.4781, TiRex (35M) 0.4885, FlowState-9.1M 0.5019. Timestral-5M
(0.479) a été relevé plus tard sur le classement en ligne et ne figure pas dans cette copie.

Regard 11 : à 240k, les trois modèles sont à égalité en brut : v9 0.4933, d200k 0.4922, d160k 0.4934 (regard 10).
Le gain de v9 au dev (−1.6 % sur d200k) ne passe pas sur GIFT, pas plus que les 40k pas de plus de d200k.
Le flip donne le même résultat sur v9 et sur d160k (0.4905 contre 0.4904). Mais la pile RateIN rapporte moins sur
v9 : 0.4804 contre 0.4763, +0.9 %. Une seule config explique les trois quarts de cet écart, solar/W/short, que
RateIN multiplie par 1.69 sur v9.

Effet de RateIN sur v9, config par config, contre le flip seul : −3.8 % de gains et +1.8 % de pertes, soit −2.0 %
net. Sur d160k, c'était −4.0 % et +1.1 %. Les pertes viennent de W, A et ett1/H. Les gains viennent de 5T
(−11.6 %) et de 10S (−8.8 %), sur les termes moyen et long.

Le modèle externe de 7M paramètres fait 0.4849 en brut par notre harnais, à égalité exacte avec sa propre boucle,
et 0.4824 avec le flip : environ 1.7 % devant v9. Sa pile RateIN est encore en cours.

Couverture de la pile v9 : q10 0.098, q90 0.896, intervalle 80 % 0.798.

Regard 12 : en brut, c8k gagne 1.3 % sur v9 (0.4871 contre 0.4933), surtout sur 10S (−7.4 %) et 5T (−5.0 %),
aux termes moyen et long. Le flip garde l'écart (0.4855 contre 0.4905). Avec la pile RateIN, l'écart disparaît :
0.4809 contre 0.4804 (+0.1 %), 0.4763 pour d160k. RateIN rapporte −0.9 % sur c8k contre −2.0 % sur v9 : sur 5T
−7.9 % contre −11.6 %, sur 10S +1.5 % contre −8.8 %, sur les configs multivariées −3.1 % contre −5.2 %. Le
contexte long et la remontée de cadence de RateIN prennent la même chose : voir plus de passé à la cadence
du modèle. Contre le modèle externe de 7M avec la même pile (0.4814) : égalité globale, c8k devant sur 10T,
5T et H (−3 à −4 %) et sur les termes moyen et long, derrière sur W (+15 %), A (+7 %), 10S (+3 %) et le terme
court (+2.3 %). Couverture de c8k avec la pile : q10 0.097, q90 0.895, intervalle 80 % 0.797. Celle du modèle
externe : intervalle 80 % 0.784.

Regard 13 : v10 fait 0.4782 avec la pile, contre 0.4804 pour v9 (−0.5 %) et 0.4763 pour d160k (+0.4 %).
Contre v9, le gain est réparti : M −2.0 %, W −1.8 %, 10S −1.5 %, Q −1.1 %, A +1.3 %, terme court −0.7 %.
Contre d160k, W reste derrière (+8.1 %), comme pour tous les bras repris de v9. Contre le modèle externe
de 7M avec la même pile (0.4814) : −0.7 %, devant sur 5T, 10T et H (−3 à −5 %), derrière sur W (+13 %)
et A (+4 %). Couverture : q10 0.101, q90 0.895, intervalle 80 % 0.795.
