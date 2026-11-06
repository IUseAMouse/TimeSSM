# Hypothèses pour PatchedSSM

Classées par gain espéré sur coût. Les gains sont des paris, rien ici n'est mesuré. Base de
référence : `configs/patched/base.yaml` (patch 16, D 320, 4 paires SSM + attention, ~8.3M,
contexte 4096). Budget total : 10M paramètres, ensembles compris.

## Protocole commun

- Run court : 20 000 pas, même corpus, score `dev_eval.py` sur les sous-ensembles mis de côté.
- Bruit : la base tourne avec deux graines ; leur écart `b` sur le score `all` est le seuil.
- Une hypothèse passe si elle gagne plus de `2b` et si elle ne dégrade pas plus de deux sous-ensembles
  de plus de 2 %. Sinon on l'abandonne sans variante supplémentaire.
- GIFT-Eval n'entre dans aucune de ces décisions.

## 1. Perte en log-ratio au naïf saisonnier

**Mécanisme.** Le score final est une moyenne géométrique de ratios `WQL / WQL_naïf`, alors qu'on
minimise une pinball moyenne dans le repère asinh. La pinball est minimisée par le quantile
conditionnel, et un quantile commute avec une transformation croissante : le repère ne change pas
l'optimum à capacité infinie, seulement ce que le modèle sacrifie à capacité finie. L'asinh écrase
les pics, donc les fenêtres à rafales pèsent peu. Variantes, en unités brutes de la fenêtre :
(a) `L_raw = pinball(q, y) / mean|y|` ; (b) `log(L_raw / L_naïf + ε)`, moyenne par batch, ce qui vise
directement la moyenne géométrique.

**Pourquoi.** Aligne la perte sur le score pour les rafales et les cadences rapides, où l'erreur brute
vit dans les pics. (b) rééquilibre aussi les séries faciles et difficiles sans table de poids.

**Test.** Remplacer `pinball` dans `train_patched.py`, 3 runs (base, a, b).
**Arrêt.** Critère commun ; attention à (b) : borner `ε` si le naïf est nul sur une fenêtre.
**Coût.** 0 paramètre.

## 2. Normalisation robuste et statistiques en entrée

**Mécanisme.** `mean/std` sur tout le contexte est dominé par un pic, et après normalisation le modèle
ne sait plus quelle échelle il a retirée. Deux changements : médiane et MAD comme centre et échelle
(MAD nul : repli sur `std`), et un token de statistiques `(log échelle, moyenne/échelle, part de
zéros, part manquante, dernière valeur normalisée, log longueur)` injecté par FiLM ou préfixé.

**Pourquoi.** Rafales et intermittence : la part de zéros et l'échelle relative décident de la largeur
du fan. Séries courtes : la longueur dit au modèle combien faire confiance à l'historique.

**Test.** Deux runs, normalisation seule puis normalisation + token. Les tests d'équivariance
d'échelle et de translation restent valables pour médiane et MAD.
**Arrêt.** Critère commun.
**Coût.** ~2k paramètres (D 320).

## 3. Moyenne exponentielle des poids

**Mécanisme.** Évaluer `θ̄_t = β θ̄_{t-1} + (1-β) θ_t`, avec `β` = 0.999 à 0.9999. La moyenne réduit la
variance de l'itéré en fin de run et tend vers des minima plus plats (SWA [1], EMA [2]). Ce n'est pas
acquis : un cosinus qui descend à presque zéro capture déjà une partie de l'effet.

**Pourquoi.** Gain général, sans coût d'inférence. Bon marché donc à essayer tôt, y compris pour tous
les runs suivants si le gain se confirme.

**Test.** Une seule run : scorer les poids courants et l'EMA au même pas.
**Arrêt.** Écart inférieur à `b` : on retire l'EMA.
**Coût.** 0 paramètre à l'inférence, double mémoire à l'entraînement.

## 4. Structure sous le patch : patch 8 et embedding convolutif

**Mécanisme.** Avec `p = 16`, le SSM voit un token tous les 16 pas : sa limite de Nyquist est de
32 pas. Une saisonnalité de 24 (H) ou 12 (M) tient entière dans un patch ou deux, et seule la couche
linéaire d'embedding et la tête peuvent la représenter, avec des poids figés par rapport à la phase.
Variantes : (a) `p = 8` ; (b) patch 16 stride 8 chevauchant [3] ; (c) petite conv 1D causale avant
le patch, qui garde le coût de `p = 16`. Le gain de calcul du patching vient pour moitié d'un
compromis sur ces périodes.

**Pourquoi.** Séries courtes basses fréquences : 60 points font 4 tokens à `p = 16`, 8 à `p = 8`.
Limite : (c) filtre en pas et non en temps physique, donc rompt l'équivariance à w ; `w` reste une
entrée utile, mais l'équivalence exacte avec la décimation disparaît.

**Test.** 3 runs. Lire surtout `tourism_*`, `cif_2016_12`, `fred_md`.
**Arrêt.** Critère commun ; `p = 8` coûte environ 2× le temps par pas.
**Coût.** Moins de 50k paramètres.

## 5. Contexte 8192

**Mécanisme.** La convolution FFT est en O(L log L) : passer de 256 à 512 tokens coûte ~2× sur les blocs
SSM, 4× sur l'attention. Mesure interne : le gain de RateIN sur TimeSSM-2.5M venait surtout de
l'historique plus long [4]. Pour contenir l'attention, la limiter aux 256 derniers tokens plus des
tokens grossiers (moyenne par blocs de 4) du passé lointain, à la façon des modèles multi-échelles
[5, 6].

**Pourquoi.** Saisonnalités longues des cadences rapides : 8192 pas à 10S couvrent moins de 23 h, à
5T un peu moins d'un mois.

**Test.** Run à 8192 avec attention pleine ; si le temps est prohibitif, version à tokens grossiers.
**Arrêt.** Critère commun. La couverture du corpus limite ce gain : peu de séries courtes en profitent.
**Coût.** 0 paramètre, ~2 à 4× en temps.

## 6. Lecture multi-cadence : RateIN externe ou gate interne

**Mécanisme.** Les deux visent la même erreur de cadence, leurs gains ne s'additionnent donc pas. Le
sélecteur appris sur une seule fenêtre a déjà échoué [4] : la cadence est une propriété de la série.
Un gate interne devrait donc lire au moins une statistique multi-fenêtres ou un spectre.

Ordre proposé : (a) règle fixe par classe de cadence, sans paramètre ; (b) RateIN par backtest ;
(c) gate interne sur spectre et fan (deux lectures, softmax par horizon sur une base lisse en
`log(1 + h)`), seulement si (b) gagne nettement sur (a).

**Pourquoi.** Le corpus équilibré et la décimation aléatoire devraient déjà réduire ce gain : c'est
la première chose à mesurer après la base.

**Test.** `dev_eval.py` à poids fixés pour (a) et (b), sans entraînement. Nécessite d'y brancher les
candidats de `k`. (c) demande un run.
**Arrêt.** Si (b) gagne moins de `2b`, on abandonne (c).
**Coût.** 0 pour (a) et (b), ~50k paramètres et 2× passes pour (c).

## 7. Ensemble de deux petits modèles

**Mécanisme.** Moyenner les quantiles de deux modèles réduit la pinball par convexité : `ρ_τ` est
convexe, donc `ρ_τ(y - (q1+q2)/2) ≤ (ρ_τ(y - q1) + ρ_τ(y - q2)) / 2`. Le gain dépend de la
décorrélation des erreurs. Comparer 1 × 8M à 2 × 4M entraînés avec des patchs différents (8 et 16),
ce qui décorrèle plus qu'une graine. Les snapshots d'un même run [7] sont une version moins chère,
mais ils comptent aussi dans les 10M.

**Pourquoi.** Gain général, et deux patchs couvrent les deux régimes de l'hypothèse 4.

**Test.** Deux runs, score de la moyenne sur `dev_eval.py`.
**Arrêt.** Critère commun contre le meilleur 8M.
**Coût.** 10M au total, 2× passes à l'inférence.

## 8. Rééquilibrage par difficulté

**Mécanisme.** Group DRO [8] : poids exponentiels `g_c ∝ exp(η L_c)` sur les classes de cadence, mis
à jour avec la perte d'entraînement. Remplace la table `RATE_WEIGHT`, fixée à la main.

**Pourquoi.** Si la perte de l'hypothèse 1(b) reste déséquilibrée, les classes M, Q et S pourraient
rester sous-apprises. Mais [8] prévient qu'une forte régularisation est nécessaire, et une classe
intrinsèquement bruitée absorbe alors tout le budget : d'où un plancher sur chaque poids.

**Test.** Un run, `η` petit.
**Arrêt.** Critère commun, sans réglage de `η` en cas d'échec.
**Coût.** 0 paramètre.

## 9. État dépendant de l'entrée, bouton w conservé

**Mécanisme.** Le S4D est LTI : son oubli ne dépend pas du signal, alors qu'une rafale ou une rupture
devrait effacer la mémoire. Remplacer le taux par `a_n(u_t) = a_n · exp(g(u_t))` (`g` borné) donne
un système linéaire à temps variable, stable car `Re a_n(u_t) < 0`, avec en ZOH
`x_t = exp(Δ a_n(u_t)) x_{t-1} + B(u_t) u_t`. `Δ = w p` reste dans l'exponentielle : le bouton survit
au sens où un pas deux fois plus long oublie deux fois plus. L'équivalence exacte avec la décimation
est perdue, l'équivariance n'étant que du premier ordre. Mamba [9], le delta gating [10] et Liquid S4
[11] sont des voisins ; ce choix est le plus proche du S4D existant. La convolution FFT disparaît :
récurrence en O(L·H·N), acceptable à ~300 tokens.

**Pourquoi.** C'est la cible directe des séries à rafales.

**Test.** Remplacer un bloc SSM sur deux. Run court, lire `alibaba_cluster_trace_2018`, `LOS_LOOP`.
**Arrêt.** Critère commun. Coût en temps à mesurer avant : si plus de 3× par pas, abandon.
**Coût.** ~20k paramètres par bloc avec `g` de rang faible (projection vers 32 facteurs par état).

## 10. Perte auxiliaire à chaque token du contexte

**Mécanisme.** Le SSM est causal : chaque token du contexte peut prévoir le patch suivant, ce qui
multiplie les cibles par fenêtre par la longueur du contexte. Une tête auxiliaire sur les sorties
SSM d'avant l'attention (non causale), avec un poids faible.

**Pourquoi.** Améliore les séries courtes et accélère l'apprentissage. Le risque est de favoriser
l'horizon court aux dépens du long.

**Test.** Un run, poids 0.1.
**Arrêt.** Critère commun, en vérifiant les horizons longs.
**Coût.** Tête partagée, ~0 paramètre.

## 11. Calibration causale par série

**Mécanisme.** La CQR [12] élargit les quantiles d'un terme calibré, garantie de couverture marginale.
Sur une série, l'adaptive conformal inference [13] ajuste le niveau en ligne sous dérive. Ici : corriger
le fan à partir de résidus du backtest causal sur le passé de la configuration. La couverture n'est pas
la pinball, donc le gain peut être nul.

**Pourquoi.** Fans mal calibrés sur les rafales. Gain probablement petit, coût presque nul.

**Test.** Sans entraînement, sur `dev_eval.py`.
**Arrêt.** Critère commun.
**Coût.** 0 paramètre, passes de backtest en plus.

## 12. Tête de sortie : spline ou mélange

**Mécanisme.** Une spline monotone de la fonction quantile [14] ou des quantiles implicites [15]
apprennent toute la distribution. Mais le benchmark ne lit que neuf niveaux, et notre tête les régresse
déjà sans croisement : le gain ne peut venir que d'un partage statistique entre niveaux. À tester en
dernier, sauf pour des niveaux extrêmes hors benchmark.

**Test.** Un run. **Arrêt.** Critère commun. **Coût.** ~10k paramètres.

## Résultats

Score `dev_eval.py` (4 origines par série), runs de 20 000 pas, batch 256, corpus v5.

| Run | dev | Remarque |
|---|---|---|
| base, graine 0 | 0.6155 | |
| base, graine 1 | 0.6366 | écart de graine 0.021, soit σ ≈ 0.015 par run |
| perte log-ratio (1b) | 0.6279 | entre les deux graines de base : non concluant |
| base 0, règle fixe 24 points par cycle (6a) | 0.6377 | pire sur alibaba (0.38 → 0.55), kdd2022, LOS_LOOP ; rejeté |
| base 1, règle fixe 24 points par cycle (6a) | 0.6616 | idem |
| normalisation médiane/MAD (2a) | 0.6582 | pire, tourism_yearly 1.42 ; rejeté |
| patch 8 (4a) | 0.5932 | −5 % sur la moyenne des bases, surtout tourism_yearly, fred_md, cif |
| patchs 8 et 32, un tiré par pas, choisi par série à l'inférence, graine 0 | 0.6241 | au niveau de la base, loin du patch 8 seul (0.5932) : rejeté, graine 1 non lancée |
| patch 8 (4a), graine 1 | 0.5788 | −0.058 sur la base de même graine ; moyenne 0.5860 contre 0.6261 (−6.4 %) : retenu |
| sans attention, 6 blocs SSM | 0.6178 | au niveau de la base |
| base, graine 2 | 0.6185 | trois graines : moyenne 0.6235, écart-type 0.011 |
| statistiques du contexte, graine 0 | 0.6259 | +0.010 sur la base de même graine |
| statistiques du contexte, graine 1 | 0.6356 | −0.001 sur la base de même graine ; moyenne 0.6308 contre 0.6235 : rejeté |
| historique complet `p_full` 0.5, graine 0 | 0.6108 | sous les trois bases ; gains sur gfc, LOS_LOOP, fred_md, perte sur tourism_yearly (1.38) |
| historique complet `p_full` 0.5, graine 1 | 0.6195 | −0.017 sur la base de même graine ; moyenne 0.6152 contre 0.6261 (−1.7 %) : retenu |
| perte de contexte (10), poids 1, `p_full` 0.5, graine 0 | 0.5678 | avec l'attention causale et le repère cumulé qu'elle exige ; −0.043 sur `p_full` de même graine |
| perte de contexte (10), poids 1, `p_full` 0.5, graine 1 | 0.5731 | −0.046 ; moyenne 0.5704 contre 0.6152 (−7.3 %), écart dès 5k pas : retenu |
| attention causale et repère cumulé sans la perte, graine 0 | 0.6239 | +0.013 sur `p_full` de même graine : le masque seul n'apporte rien, le gain vient de la perte |
| attention causale et repère cumulé sans la perte, graine 1 | 0.6360 | +0.017 ; moyenne 0.6300 contre 0.6152 (+2.4 %) : le masque seul coûte un peu, il ne vaut que par la perte qu'il permet |
| SSM seul (4 blocs, 3.4 M paramètres contre 8.3 M), perte de contexte, graine 0 | 0.6331 | meilleur que le même modèle sans la perte à 5k pas (0.685 contre 0.708), moins bon à 20k |
| SSM seul (4 blocs, 3.4 M), sans la perte, graine 0 | 0.6186 | au niveau de la base à 8.3 M ; la perte de contexte coûte +0.015 sans attention (1σ, graines 1 en file) |
| corpus v6, `p_full` 0.5, graine 0 | 0.6222 | +0.011 sur v5 de même graine |
| corpus v6, `p_full` 0.5, graine 1 | 0.6353 | +0.016 ; moyenne 0.6288 contre 0.6152 (+2.2 %) : le filtrage des fuites coûte un peu sur le dev, v6 reste le seul corpus propre vis-à-vis de GIFT |
| v6, mélange de séries `p_mix` 0.3, graine 0 | 0.5980 | −0.024 (−3.9 %) sur v6 de même graine ; graine 1 en file |
| v6, mélange de séries `p_mix` 0.3, graine 1 | 0.6296 | −0.006 ; moyenne 0.6138 contre 0.6288 (−2.4 %), les deux graines dans le même sens : retenu, à confirmer sur la recette de la perte de contexte et v7 |
| tête à 21 quantiles (les 9 rendus en sont un sous-ensemble), recette de la perte de contexte, graine 0 | 0.5834 | +0.016 (+2.7 %) sur la même recette à 9 ; égal à 10k pas, l'écart se creuse à la fin |
| tête à 21 quantiles, graine 1 | 0.5722 | −0.001 sur la même recette à 9 ; moyenne 0.5778 contre 0.5704 (+1.3 %) : rejeté, la tête reste à 9 |
| patch 8 sur la recette de la perte de contexte, graine 0 | 0.5848 | +0.017 (+3.0 %) sur la recette au patch 16 ; égal à 5k et 10k pas, derrière ensuite ; graine 1 arrêtée à 7 200 pas : à égalité le patch 16 reste, deux fois moins cher ; rejeté |
| recette de la perte de contexte sur le corpus v7, graine 0 | 0.5574 | −0.010 (−1.8 %) sur v5 de même graine |
| recette de la perte de contexte sur le corpus v7, graine 1 | 0.5667 | −0.006 ; moyenne 0.5621 contre 0.5704 (−1.5 %), les deux graines dans le même sens : retenu, v7 devient la référence |
| mélangeur delta une couche sur deux (`[ssm, delta, ssm, delta]`), recette de la perte de contexte, v5, graine 0 | 0.5599 | −0.008 (−1.4 %) sur causal0 ; −2.6 % à 10k, l'avance se resserre ; ablation du rappel associatif, pas candidat (perd l'invariance au pas) ; graine 1 en file |
| v7, mélange de séries `p_mix` 0.3, graine 0 | 0.5651 | +0.008 (+1.4 %) sur v7 de même graine : avec la perte de contexte, le mélange ne paie plus ; graine 1 en file |
| v7, poids de la perte de contexte 2, graine 0 | 0.5608 | +0.003 (+0.6 %) sur le poids 1 de même graine, en avance à 5k ; graine 1 en file |
| deux passes sur la pile (`loops` 2, poids partagés), recette de la perte de contexte, v5, graine 0 | 0.5572 | −0.011 (−1.9 %) sur causal0 ; −0.9 % à 15k, l'écart se creuse en fin de cosinus ; pas 30 % plus lent ; graine 1 en file |
| v7, 40k pas (cosinus sur 40k), graine 0 | 0.5469 | −0.011 (−1.9 %) sur 20k de même graine ; 0.5630 / 0.5515 / 0.5529 / 0.5483 à 20k / 25k / 30k / 35k (palier à 30k dû à fred_md) ; doubler les pas paie encore |
| Mamba sélectif sur les quatre couches (`mixers` selective ×4), recette de la perte de contexte, v5, graine 0 | 0.5790 | +0.009 (+1.5 %) sur la moyenne causal0/1 (0.5705) ; +1.8 % à 15k, +0.5 % à 5k ; gagne sur les séries courtes (fred_md, tourism), perd sur les longues saisonnières (gfc, LOS_LOOP) ; rejeté, graine 1 retirée de la file |
| Mamba sélectif ×4 avec deux passes (`loops` 2), v5, graine 0 | 0.5726 à 15k | +1.0 % sur loop0 à 15k (0.5671), −0.4 % seulement sur causal0/1 sans boucles ; arrêté à 15k : le sélectif seul reste derrière S4D même avec les boucles, rejeté |
| hybride S4D / Mamba sélectif une couche sur deux (`[ssm, selective, ssm, selective]`), v5, graine 0 | 0.5649 | −0.006 (−1.0 %) sur causal0/1 ; −3.2 % à 10k, −0.9 % à 15k : apprend plus vite, finit au niveau des autres variantes (delta0 −1.9 %, loop0 −2.3 %) ; bat S4D et Mamba seuls sur gfc12/17 ; en attente de l'empilement avec les boucles (selhloop0) |
| v7, moitié des séries de chaque sous-ensemble (`series_frac` 0.5, tirage fixe, 50.1 % des points), poids des sous-ensembles inchangés, graine 0 | 0.5564 | −0.2 % sur c7_0, −1.0 % sur la moyenne c7_0/1 ; à égalité ou mieux à chaque point (5k +0.5 %, 10k −0.2 %, 15k −0.5 %) : à 20k pas le modèle n'est pas limité par le nombre de séries ; le corpus suivant vise la diversité, pas le volume |
| v7, planning WSD (warmup 1000, plat, puis linéaire vers 1e-3 sur les 20 derniers %), graine 0 | 0.5586 | −0.6 % sur la moyenne c7_0/1 (0.5621), entre les deux graines ; en retard pendant le plateau (15k 0.5793 contre 0.5683), rattrapé en 4000 pas de décroissance : WSD retenu pour le run final, ses checkpoints de plateau donnent la courbe d'échelle |
| v8 (v7 + 1.51 G points synthétiques : ARIMA-GARCH, régimes, calendrier, croissance), graine 0 | 0.5567 | −1.0 % sur la moyenne c7_0/1, −0.1 % sur c7_0 ; devant à chaque point (5k −1.2 %, 10k −0.5 %, 15k −0.7 %) ; retenu pour le run final, seconde graine différée |
| v5, 4 passes partagées (loop0 en a 2), graine 0 | 0.5646 à 15k | −0.4 % sur loop0 à 15k (5k −0.7 %, 10k −1.2 %) pour 2× le temps par pas : sous le seuil de 1 %, arrêté à 15k ; décision reportée à 20k (reprise du checkpoint de 15k) et à halt0, entraîné à 1..4 passes tirées au hasard avec arrêt par la perte en contexte |
| delta0 avec la graine 1 (S4D/delta en alternance + attention, v5) | 0.5594 | delta0/1 : 0.5599 / 0.5594, moyenne 0.5597, −1.9 % sur la moyenne causal0/1 (0.5704), devant à chaque point sur les deux graines : gain de delta confirmé ; reste dloop0 pour l'empilement avec la boucle |

Une seule fenêtre par série, en fin d'historique, donnait un écart de graine de 0.052 : un choc
commun à toutes les séries d'un sous-ensemble décidait de son score. À σ ≈ 0.015, un run unique ne
sépare que des effets d'environ 5 % ; les hypothèses suivantes tournent avec deux graines.

### Où se perd la perte : régularité des séries

`scripts/dev_sharpness.py` trie les coupes du dev par l'erreur du seasonal naive rapportée au
niveau de la cible, en trois tiers. Perte sur celle du naive (moyenne géométrique par coupe),
couverture de l'intervalle 10-90 %, largeur de l'intervalle sur l'erreur absolue de la médiane :

| Run | tiers régulier | tiers moyen | tiers bruité | couverture (régulier / moyen / bruité) |
|---|---|---|---|---|
| base 0 | 0.731 | 0.661 | 0.553 | 0.81 / 0.73 / 0.78 |
| `p_full` 0.5, graine 0 | 0.746 | 0.682 | 0.553 | 0.79 / 0.71 / 0.78 |
| patch 8 | 0.693 | 0.622 | 0.555 | 0.82 / 0.74 / 0.79 |

Sur les séries régulières, le modèle ne bat le naive que de 27 %, contre 45 % sur les bruitées.
La couverture y est nominale : l'intervalle n'est pas trop large, c'est la médiane qui manque de
précision. Le patch 8 gagne exactement là (−5 % sur les deux premiers tiers, rien sur le
bruité) : la résolution et le rappel exact d'un cycle passé sont le levier. D'où le mélangeur à
règle delta (mémoire adressée par le contenu, Yang et al. 2024 [10]), en file contre `p_full`
sur v6. Le tiers moyen est sous-couvert (0.71-0.74 pour 0.80).

### Perte de contexte : par tiers de l'horizon

`scripts/dev_horizon.py` découpe l'horizon de chaque sous-ensemble en trois tranches, même score que
le dev sur chacune. Moyenne des deux graines :

| Run | premier tiers | deuxième | troisième |
|---|---|---|---|
| `p_full` 0.5 | 0.537 | 0.625 | 0.661 |
| perte de contexte | 0.469 (−12.6 %) | 0.589 (−5.7 %) | 0.626 (−5.4 %) |

Le gain est plus fort au début de l'horizon, comme attendu d'une perte qui ne vise que le patch
suivant, mais il tient jusqu'au bout : pas de perte sur l'horizon long. Trois ablations en file,
deux graines chacune : attention causale et repère cumulé sans la perte (sépare le masque de la
densité du signal), SSM seul avec et sans la perte (la densité suffit-elle sans attention).

## Écartés pour l'instant

- **SAM** [16] : deux passes par pas, trop cher tant que le reste n'est pas exploré.
- **Neural CDE** [17] : la masse manquante est traitée par masque ; coût élevé pour un cas rare ici.
- **Quantile multi-horizon séparé** [18] : la tête par token couvre déjà tous les horizons.

## Combinaison

Les hypothèses 1, 2, 3 et 4 sont indépendantes et peu coûteuses : celles qui passent se cumulent pour la
base du deuxième tour. 6 se mesure sans entraînement sur la base retenue. 7 et 9 viennent après, à
budget de paramètres connu.

## Références

1. Izmailov et al., *Averaging Weights Leads to Wider Optima and Better Generalization*, https://arxiv.org/abs/1803.05407
2. Busbridge et al., *How to Scale Your EMA*, https://arxiv.org/abs/2307.13813
3. Nie et al., *A Time Series is Worth 64 Words*, https://arxiv.org/abs/2211.14730
4. Vincent, *RateIN: Time Series Foundation Models Have a Sampling Problem*, https://doi.org/10.5281/zenodo.23868904
5. Challu et al., *N-HiTS*, https://arxiv.org/abs/2201.12886
6. Wang et al., *TimeMixer*, https://arxiv.org/abs/2405.14616
7. Huang et al., *Snapshot Ensembles: Train 1, get M for free*, https://arxiv.org/abs/1704.00109
8. Sagawa et al., *Distributionally Robust Neural Networks*, https://arxiv.org/abs/1911.08731
9. Gu, Dao, *Mamba*, https://arxiv.org/abs/2312.00752
10. Yang, Kautz, Hatamizadeh, *Gated Delta Networks*, https://arxiv.org/abs/2412.06464
11. Hasani et al., *Liquid Structural State-Space Models*, https://arxiv.org/abs/2209.12951
12. Romano, Patterson, Candès, *Conformalized Quantile Regression*, https://arxiv.org/abs/1905.03222
13. Gibbs, Candès, *Adaptive Conformal Inference Under Distribution Shift*, https://arxiv.org/abs/2106.00170
14. Gasthaus et al., *Probabilistic Forecasting with Spline Quantile Function RNNs*, https://proceedings.mlr.press/v89/gasthaus19a.html
15. Dabney et al., *Implicit Quantile Networks*, https://arxiv.org/abs/1806.06923
16. Foret et al., *Sharpness-Aware Minimization*, https://arxiv.org/abs/2010.01412
17. Kidger et al., *Neural Controlled Differential Equations for Irregular Time Series*, https://arxiv.org/abs/2005.08926
18. Wen et al., *A Multi-Horizon Quantile Recurrent Forecaster*, https://arxiv.org/abs/1711.11053
