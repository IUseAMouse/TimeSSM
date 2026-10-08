# Registre expérimental TimeSSM

Newest-first. Même doctrine que `../TimeJEPA/docs/EXPERIMENTAL_LOG.md` : prédictions
gravées avant chaque run, une variable par bras, oracle = diagnostic jamais officiel.

## Journal des mises à jour

- **2026-10-08 (VERDICT P-SSM.12 : ENTRE LES DEUX SEUILS — dernier checkpoint 0.7544 / 0.5139 au
  stack, −0.3 pt de MASE et −0.16 pt de CRPS sur la ligne de base ; le dernier checkpoint
  `1.2814` devient le checkpoint à publier)** — `ssm_mini_v3_freq`, 5 checkpoints, flip + mix +
  pool + RateIN-up + `+freq_delta=true`, 97 configs sans échec, dans l'ordre du run : `1.2906`
  0.7547 / 0.5140 / 0.690 · `1.2842` 0.7576 / 0.5177 / 0.680 · `1.2820` 0.7565 / 0.5153 / 0.693 ·
  `1.2817` 0.7537 / 0.5139 / 0.696 · **`1.2814` (dernier) 0.7544 / 0.5139 / 0.698**. Ligne de
  base (`1.2841` sans entraînement, même flag) : 0.7572 / 0.5155. Seuils : succès ≤ 0.512 NON
  atteint, échec ≥ 0.5155 NON atteint : ni l'un ni l'autre. Lecture : la bande des cinq
  (0.5139-0.5177) est décalée d'environ 0.15 pt sous la ligne de base, soit la moitié du
  bruit entre checkpoints (0.3 pt) ; les deux derniers sont égaux au dixième de point et
  sont les meilleurs, ce qui est cohérent avec un petit apprentissage réel plutôt qu'avec
  du bruit (le bruit aurait placé le meilleur n'importe où). L'effet est donc petit et
  probablement réel, dilué par la moitié synthétique du batch (réserve écrite le 06/10).
  Le nu avec flag du dernier checkpoint (`logs/freq_nu.out`) reste à lire. **Décision de
  l'utilisateur** : publier `1.2814`, qui est le DERNIER checkpoint du run (aucune sélection
  sur le test ; il se trouve aussi à égalité de meilleur). Suite : γ recalibré sur `1.2814`
  à Δ lié, éval avec γ, config de reproduction `timessm_2.5m_gift` basculée sur ce
  checkpoint (hérite de `ssm_mini_v3_freq_eval`, plage Δ [1/64, 8]), contrôle de
  régression et reproduction à neuf. Le val loss de ce bras, plus régulier à l'annealing
  (remarque de l'utilisateur), n'est pas comparable aux bras précédents (validation à Δ lié).

- **2026-10-06 (BRAS P-SSM.12 LANCÉ, témoins W&B à 5 k pas : conformes, sauf la part
  étiquetée — 52 % et non ~70 % ; réserve inscrite AVANT le résultat)** — `ssm-mini-v3-freq`,
  SEED0 2026, départ `1.2841`, budget complet (~24 h). `freq/w_min` 0.016667 (séries à la
  minute), `freq/w_max` 6 (6H), `freq/n_unique` 9 à 14, décimation et horizon inactifs,
  contexte tiré entre 128 et 1024. `freq/labelled_frac` 0.51-0.53 : j'avais annoncé ~0.7
  en comptant les fichiers (30 synthétiques sur 106) ; l'échantillonneur pondère par
  famille et le synthétique pèse près de la moitié de chaque batch (remarque de
  l'utilisateur). `aug/w_neq1_frac` alterne entre 0.37 et 0.87 selon que le tirage hérité
  des items sans fréquence vaut 1 ou non : 37 % du batch a un Δ lié ≠ 1, ~15 % est horaire.
  **Conséquence pour la lecture du verdict** : la moitié du signal d'entraînement reste un
  Δ aléatoire ; le bras teste « Δ lié sur une moitié du batch ». Un succès n'en est que
  plus net ; un échec est moins concluant que les seuils ne le disent (dilution par le
  synthétique non exclue), et la variante suivante serait de donner au synthétique une
  fréquence cohérente avec son générateur. Les seuils ne changent pas. À surveiller :
  `train_loss_step` montre des pics à 15-20 pour une base ~1, à comparer aux bras
  précédents (wide, B5) avant d'y lire quoi que ce soit.

- **2026-10-06 (CHIFFRE DE LA CARTE DU 2.5M : 0.7572 / 0.5142, couverture 0.758 ; config de
  reproduction en une commande ; bras P-SSM.12 : option A, budget complet, ~24 h)** —
  γ recalibré à Δ lié (`calibrate_ssm.sh … --flip --frequency-table`, fenêtres de validation
  du corpus, jamais GIFT) : [1.149, 0.920, 1.026, 1.061, 1, 0.921, 0.993, 0.893, 1.164].
  Checkpoint `1.2841`, flip + mix + pool + RateIN-up + `+freq_delta=true` + γ, 97 configs :
  **0.7572 / 0.5142 / 0.758** (sans γ : 0.7572 / 0.5155 / 0.696 ; MASE inchangée au chiffre
  près, comme il se doit). Échelle complète du même checkpoint : nu 0.8534 / 0.5836 · nu +
  Δ lié 0.8126 / 0.5547 · stack + RateIN-up 0.7613 / 0.5194 · + Δ lié 0.7572 / 0.5155 ·
  + γ 0.7572 / 0.5142. **Reproduction** : `configs/timessm_2.5m_gift.yaml` porte TOUS les
  réglages d'inférence de ce chiffre (flip, RateIN, fenêtres, Δ lié, γ, batch) et le fichier
  γ est versionné (`configs/calibration/gamma_epoch00_valloss1.2841_flip_fdelta.json`,
  valable pour ce checkpoint et ces réglages seulement) :
  `EVAL_CONFIG=timessm_2.5m_gift scripts/eval_ssm.sh <ckpt>`. Les réglages ne sont PAS des
  défauts du harnais : toutes les autres configs en restent vierges (test), évaluer sans
  eux reste le comportement par défaut. Le harnais résout désormais un chemin de fichier
  relatif par rapport au dépôt de la config (`_resolve_data_file`) ; vérifié de bout en
  bout avec un checkpoint factice depuis un autre répertoire (réglages lus, γ trouvé, même
  nom de dossier que l'éval du pod). À faire après le bras : la reproduction complète sur
  le pod dans `evaluation/timessm_2.5m/` (2 h), qui doit rendre 0.7572 / 0.5142.
  **Vitesse du bras** (`profile_step.py --scales`, batch 128, une carte) : 1 cadence 3.02
  it/s / 14.9 Gio ; première version du chemin mêlé : +3.5 Gio dès 2 cadences puis +0.7 Gio
  par cadence, OOM à 5 ; après correctif (`495625f`, un noyau par cadence sous checkpoint,
  convolution groupée par cadence) : 4 cadences 2.20 it/s / 14.4 Gio, 12 : 1.51 / 14.7, 16 :
  1.31 / 14.9. Mémoire constante, vitesse divisée par deux à 12-16 cadences : le bras
  prendra ~24 h au lieu de 12. Décision de l'utilisateur : budget complet (option A).

- **2026-10-06 (Δ LIÉ À LA FRÉQUENCE, LIGNE DE BASE SANS ENTRAÎNEMENT : nu 0.8126 / 0.5547,
  stack + RateIN-up 0.7572 / 0.5155 — meilleur chiffre du 2.5M, obtenu à l'inférence seule ;
  P-SSM.12(a) dépassée du bon côté ; SEUILS DU BRAS RE-GRAVÉS avant le lancement)** —
  Rétrocompatibilité d'abord, sur le pod : `test_golden` (1e-5 entre machines, strict sur
  la machine d'enregistrement), `test_frequency_delta`, `test_ssm_harness` verts ;
  `check_regression_ssm.sh` : 10 comparaisons nu et stack sur poids réels, écart relatif
  maximal 2.3e-8 contre le cache : OK. Table des fréquences relue et poussée (TimeJEPA
  `f171b5f` : 76 fichiers à fréquence, 30 synthétiques sans ; journaliers à rythme hebdo :
  favorita ×2, m5, nn5 ; nn5 décidé sur la donnée et non sur son étiquette de domaine).
  **Mesure**, checkpoint `1.2841` (jamais entraîné avec un Δ lié), `+freq_delta=true`, plage
  [1/48, 4], 97 configs sans échec : **nu 0.8126 / 0.5547 / couv. 0.700** contre 0.8534 /
  0.5836 (−4.1 pt de MASE, −2.9 pt de CRPS) ; **flip + mix + pool + RateIN-up 0.7572 /
  0.5155 / 0.696** contre 0.7613 / 0.5194 (−0.4 pt et −0.4 pt ; comparaison appariée, même
  checkpoint, mêmes instances). Prédit : nu 0.570 ± 0.010 (réel 0.5547, mieux), stack dans
  ±0.3 pt (réel −0.39, mieux). La règle seule rend au modèle nu la moitié de ce que RateIN
  lui apporte (0.5836 → 0.5194), et les deux se composent. Exemples nus : solar/10T/long
  MASE 1.753 → 0.947 ; m4_hourly inchangé (w = 1) ; bizitobs 10S toujours mauvais (2.5 à
  4.7). Repères : Toto-2.0-4m 0.7565 / 0.5242 (égalité en MASE désormais), FlowState-9.1M
  0.7262 / 0.5019 (reste 1.4 pt de CRPS, 3.1 pt de MASE). Réserves : (1) la règle emprunte
  à FlowState deux connaissances propres à GIFT, le domaine de chaque jeu (journalier
  hebdo ou non) et l'exception bizitobs_l2c ; à publier comme telles, et à mesurer sans
  l'exception. (2) γ est à recalibrer à Δ lié (`calibrate_ssm.sh … --frequency-table`).
  (3) carte par fréquence à lire (`gift_gap_ssm.py` avec et sans le flag) avant de conclure
  sur M/Q/A où w est borné à 4. **Seuils du bras re-gravés** (ceux du 05/10 supposaient une
  ligne de base à 0.570 et sont déjà atteints sans entraînement) : dernier checkpoint de
  `ssm_mini_v3_freq`, `+freq_delta=true` : nu ≤ 0.545 de CRPS et ≤ 0.800 de MASE ; stack +
  RateIN-up ≤ 0.512 / ≤ 0.752. SUCCÈS si le stack passe sous 0.512. ÉCHEC si le stack reste
  ≥ 0.5155 : l'entraînement n'ajoute rien à la règle appliquée à l'inférence.

- **2026-10-05 (BRAS « Δ LIÉ À LA FRÉQUENCE DÉCLARÉE » : code livré dans les deux dépôts,
  rétrocompatibilité testée, PRÉDICTIONS P-SSM.12 gravées avant tout chiffre)** — Décision de
  l'utilisateur : méthode A (fréquence déclarée, comme FlowState), pas l'estimation de période
  (le détecteur FFT du 01/09 cassait D/W/M). **Convention** reprise pas à pas de l'implémentation
  de référence de FlowState (`get_fixed_factor`, granite-tsfm, lue le 05/10), dans
  `timejepa/data/frequency.py` : `w = 24 / saison` ; secondes → cycle d'une heure (10S : 360
  pas), minutes et heures → cycle d'un jour (15T : 96, H : 24, 6H : 4), journalier → 7 si le
  domaine suit un rythme humain (Transport, Healthcare, Sales) sinon 365, hebdo → 52.14,
  mensuel → 12, trimestriel et annuel → 4 ; exception GIFT de leur enveloppe : bizitobs_l2c
  sans cycle journalier (saison × 7). Leur commentaire sur 6H (« only CMIP6 in pretraining »)
  indique qu'ils appliquent la même fonction au pré-entraînement. **Où vit la fréquence** :
  pas dans les `.npy` ; une table versionnée `TimeJEPA/configs/corpus_v3_frequencies.yaml`
  (stem → fréquence, `null` pour le synthétique), construite sur le pod par
  `scripts/build_frequency_table.py` à partir du champ `freq` des sources Hugging Face à la
  révision épinglée (vérifié : taxi_30min → 30T, kdd2022 → 10T, favorita_sales → D) ; les
  fichiers journaliers sont écrits avec `weekly: REVIEW`, que le chargeur refuse tant qu'un
  humain n'a pas tranché. Un fichier du corpus absent de la table est une erreur. **Chemin** :
  table → `MultiDatasetMonashDataModule(frequency_table=)` → `TimeSeriesDataset(season_length=)`
  → clé d'item `season` (divisée par le facteur de résolution) → `SSMFinetuneModule
  (delta_from_frequency)` : `w = 24 / (saison / k)` par item, borné à `model.delta_range` ;
  items sans fréquence : tirage aléatoire hérité en entraînement, 1 en éval ; la validation
  tourne à Δ lié (val loss NON comparable aux bras précédents). **Zéro paramètre** : le
  `state_dict` est identique, `1.2841` se charge tel quel. Couche : sur le chemin par item,
  FFT des noyaux uniques PUIS indexation (le chemin à échelle unique n'est pas touché).
  **Harnais** : `+freq_delta=true`, tag `_fdelta`, `w` passé à chaque prévision et à chaque
  candidat du backtest (contexte décimé par k → `w · k`), exclusif avec `+ratein=delta` et
  `+ratein_w`, avertissement si un modèle `expects_frequency` est évalué sans.
  `calibrate_quantiles.py --frequency-table` (fichier γ suffixé `_fdelta`). **Rétro-
  compatibilité** : fichier de référence `tests/golden/ssm_golden.pt` enregistré AVANT le
  changement (prévisions à tous les types de `w`, pertes d'entraînement avec leurs tirages,
  perte d'éval, noms des paramètres) : égalité stricte après ; mode éteint : clé `season`
  ignorée, aucun tirage aléatoire en plus ; sans table : mêmes clés d'item, mêmes fenêtres ;
  sans flag : mêmes appels (`w is None`) et mêmes résultats du harnais sur stub et sur un
  vrai petit SSM. Suites : TimeMamba 71 verts (dont 11 nouveaux), TimeJEPA `test_frequency`
  35. Contrôle sur poids réels : `scripts/check_regression_ssm.sh` (réévalue `1.2841` nu et
  stack sur 5 configs dans un dossier neuf, exige l'égalité avec le cache). Non étendu :
  `scripts/evaluate.py` (Nixtla / Monash), dont le chemin n'est pas modifié. Config du bras :
  `ssm_mini_v3_freq` = `ssm_mini_v3_wide` + table + mode + `delta_range [1/64, 8]`
  (trimestriel, annuel et 6H demandent w = 6), même sampler, même budget, LR 1e-4.
  **P-SSM.12.** (a) Ligne de base sans entraînement, `1.2841` + `+freq_delta=true` (borné
  à [1/48, 4]) : nu 0.570 ± 0.010 de CRPS (contre 0.5836), gain ≥ 5 % sur le sub-horaire,
  perte possible sur le journalier sans cycle hebdo (w = 0.066) ; stack + RateIN-up dans
  ±0.3 pt de 0.5194. (b) Bras, dernier checkpoint, `+freq_delta=true` : nu ≤ 0.560 de CRPS
  et ≤ 0.830 de MASE (au moins 40 % du gain de RateIN obtenu nativement) ; stack +
  RateIN-up ≤ 0.516 / ≤ 0.757 ; `flat` sur solar/10T/long : amplitude du dernier bloc ≥ 0.5
  de la vérité (0.08 aujourd'hui). SUCCÈS si le stack passe sous 0.516. ÉCHEC si le nu reste
  ≥ 0.575 ou si le stack ne bat pas 0.5194 : lier Δ à la fréquence ne suffit pas en 12 h de
  continuation (un résultat nul en continuation prouve moins qu'un entraînement de zéro).

- **2026-10-05 (CHIFFRES DÉFINITIFS DU CHECKPOINT PUBLIÉ `1.2841` après la correction de la MASE
  du harnais : −0.08 pt de MASE sur chaque empilement, CRPS inchangé au chiffre près)** —
  Correction dans TimeJEPA (`2733c47`) : la MASE est moyennée sur les observations valides,
  comme gluonts (`axis=None`), et non par instance ; 15 configs à cibles partiellement NaN
  recalculées (`queue_recompute_nan.sh`), les 82 autres relues du cache. Avant → après,
  MASE / CRPS : nu 0.8542 → **0.8534** / 0.5836 · flip 0.8351 → **0.8343** / 0.5644 · stack
  0.7679 → **0.7671** / 0.5242 · RateIN-up 0.7621 → **0.7613** / 0.5194 · RateIN-up + γ 0.7621
  → **0.7613** / 0.5183. Les cinq CRPS ressortent identiques : la correction ne touche que
  l'agrégation de la MASE, comme prévu. L'hypothèse « le harnais sous-évalue TimeSSM » vaut
  donc 0.08 pt de MASE et 0 de CRPS. Les anciens JSON sont dans
  `per_config_before_2026-10-05/` de chaque dossier. Restent à l'ancienne MASE sur ces 15
  configs : les quatre autres checkpoints de la bande, tous les runs 10M, les runs de
  diagnostic (oracle, backtest, τ, lookbacks) ; écart attendu du même ordre (−0.08 pt), à
  recalculer seulement pour ce qui est publié. La ligne « * fewer than 97 configs » dans
  `queue_nan.out` est la légende du tableau attrapée par le grep de la file, pas un
  avertissement sur ces runs.

- **2026-10-05 (BANDE DU 2.5M AVEC RateIN-up : 0.761-0.765 / 0.519-0.522 ; R1 CLOS — la dernière
  couche était à son optimum et le checkpoint réajusté fait un peu MOINS bien sur GIFT ;
  P-SSM.10 : la longueur de contexte n'est pas un axe au niveau global)** — **Bande**, flip +
  mix + pool + `k_up=2x3x4` + `min_bt=4` + `bt_windows=4`, 5 checkpoints wide à 97 configs sans
  échec, dans l'ordre du run : `1.2836` 0.7607 / 0.5200 / 0.708 · `1.2845` 0.7613 / 0.5207 /
  0.701 · `1.2841` 0.7621 / 0.5194 / 0.699 · `1.2817` 0.7654 / 0.5220 / 0.701 · `1.2822`
  0.7644 / 0.5215 / 0.703. Étendue 0.47 pt de MASE, 0.26 pt de CRPS : c'est l'unité de bruit
  de toute comparaison sur ce modèle. Le `1.2841` est le meilleur des cinq en CRPS : le
  citer seul serait une sélection sur le test ; le chiffre à publier est la bande, et le
  dernier checkpoint (`1.2822`, 0.7644 / 0.5215) est le point sans sélection. Les deux
  files `queue_phase0` lancées en double ont calculé des configs deux fois (calcul
  déterministe, 0 échec) ; `logs/diag_flat.txt` et `diag_data.txt` ont pu être réécrits
  par le doublon, les chiffres de référence sont ceux des entrées du 04/10. **R1**
  (`refit_last_layer.py`, 24.36 M lignes d'entraînement, 0.75 M de validation, 1737
  paramètres, L-BFGS) : pinball d'entraînement 4.10683 → 4.10555 (−0.03 %), de validation
  4.00015 → 3.99809 (−0.05 %), MAE de la médiane −0.02 %, poids déplacés de 0.8 %. GIFT du
  checkpoint réajusté : stack RateIN-up 0.7653 / 0.5212 contre 0.7621 / 0.5194 (+0.32 pt de
  MASE, +0.18 pt de CRPS, donc PIRE, dans la bande des checkpoints) ; nu 0.8571 / 0.5834
  contre 0.8542 / 0.5836. P-SSM.11 : la partie validation est surestimée (prédit −0.1 à
  −0.5 %, mesuré −0.05 %), la partie GIFT est fausse du mauvais côté (prédit ±0.05 pt).
  Lecture : le gradient stochastique avait laissé la dernière couche à 0.05 % de son
  optimum ; il n'y a rien à gagner dans la tête à features gelées, l'écart est dans le
  corps. Et un déplacement de 0.8 % des poids de sortie bouge GIFT de 0.2-0.3 pt : même
  ordre que la bande. Checkpoint réajusté NON retenu. **P-SSM.10** (nu, référence 0.8542 /
  0.5836) : `+max_context=256` 0.9144 / 0.6334 (pire, prédit) ; moyenne des lookbacks
  256 + 1024 : 0.8693 / 0.5947 (pire de 1.5 pt de MASE ; prédiction ≤ 0.850 fausse). Le
  détail W/M/A/Q reste à lire sur la carte par fréquence avant de clore formellement.
  **Constat pour le bras « fréquence en entrée »** : le corpus `lotsa_v3` ne porte AUCUNE
  métadonnée de fréquence (`prepare_lotsa.py` ne la garde pas, le datamodule ne la
  transmet pas) : le bras demande une table famille → fréquence, sa propagation
  dataset → batch → modèle, et le passage de la fréquence GIFT dans le harnais (divisée
  par k sous RateIN). Correction d'une phrase du 04/10 : le corpus contient déjà des
  familles synthétiques (`generate_synthetic.py`, `build_corpus_v3.sh`).

- **2026-10-04 (DÉCISION DE PUBLICATION + BRAS R1 : réajustement de la dernière couche,
  PRÉDICTION P-SSM.11 gravée avant le run)** — Décision de l'utilisateur : publier le 2.5M
  comme modèle principal, en assumant la limite d'itération d'un indépendant (un bras de
  zéro = 5 jours des 3 cartes), RateIN étant ce qui a été optimisé autour. Bras R1 demandé
  par l'utilisateur : `scripts/refit_last_layer.py`. Le corps du réseau est gelé ; les
  entrées de la dernière projection de la tête (`unpatching.projection`, 192 → 9 sorties
  brutes ; PAS la couche cachée de 1536 annoncée le 03/10, qui est suivie d'un LayerNorm)
  sont stockées sur ~108 k fenêtres d'entraînement (1024 par famille, 16 positions par
  batch, contexte recadré comme à l'entraînement), cible dans le repère de la perte ; la
  pinball de la tête sur le fan monotone est minimisée en batch complet par L-BFGS depuis
  les poids entraînés (ligne médiane = régression en écart absolu, convexe ; lignes de
  largeur à travers softplus et somme cumulée). Checkpoint écrit seulement si la pinball
  de VALIDATION baisse ; c'est le checkpoint d'entrée avec deux tenseurs remplacés. 4 tests.
  File `scripts/queue_refit.sh` : réajustement, puis éval nue et stack RateIN-up.
  **P-SSM.11** : pinball de validation −0.1 à −0.5 % ; GIFT stack RateIN-up dans ±0.05 pt
  de 0.7621 / 0.5194 et nu dans ±0.1 pt de 0.8542 / 0.5836, donc dans le bruit de la bande.
  Raison : 1737 paramètres derrière un LayerNorm, sur un modèle dont le cosinus est allé
  au bout. SUCCÈS (prédiction fausse, bras à garder) si le stack gagne ≥ 0.15 pt de CRPS ou
  de MASE. Si rien ne bouge, la mesure vaut quand même : elle dit que le gradient
  stochastique a laissé la dernière couche à son optimum. Non testé de bout en bout hors
  du pod (pas de corpus en local) : la partie datamodule reprend `calibrate_quantiles.py`.

- **2026-10-04 (TEST « COUVERTURE DU CORPUS » : l'écart de MASE à FlowState est le même avec et
  sans famille cousine dans le corpus)** — 2.5M wide `1.2841`, stack + RateIN-up, 97 configs,
  `gift_gap_ssm.py --metric mase --by corpus_cousin,domain`. Sans cousin (64 configs) : MASE
  0.7694, ×1.051 contre FlowState-9.1M (19/64), ×1.027 contre Toto-2.0-4m, ×1.063 contre
  TTM-R3-PT. Avec cousin (33) : 0.7480, ×1.046 contre FlowState (8/33), ×0.970 contre Toto
  (20/33), ×1.033 contre TTM. Contre FlowState l'écart ne dépend pas de la couverture
  (0.5 pt de différence) ; contre Toto et TTM il en dépend de 3 à 6 pt, mais le rapport à un
  concurrent mélange sa couverture à la nôtre (Toto est chez lui sur le CloudOps, sans
  cousin chez nous). Par domaine contre FlowState : Energy ×1.044 (6/32), Web/CloudOps
  ×1.029, Nature ×1.020, Transport ×1.049 (3/15, alors que Toto y est battu 11/15),
  Healthcare ×1.027, Sales ×1.084, Econ/Fin ×1.237 (0/6). Lecture : hors Econ/Fin (les m4,
  ~1.3 pt de l'écart global à eux six), FlowState est meilleur de 2 à 5 % PARTOUT, cousin ou
  non : signature d'un modèle un peu meilleur en médiane sur tout, pas d'un trou de
  couverture. L'hypothèse « le corpus ne couvre pas GIFT » n'est pas soutenue ; la
  composition du corpus (tendances, synthétique) et le modèle restent ouverts, non séparés.

- **2026-10-04 (PHASE 0, `decomp` : l'erreur sur m4_hourly et m4_weekly est une erreur de
  NIVEAU ; l'hypothèse « statistiques périmées » est ÉCARTÉE sur m4_hourly et tient sur
  m4_weekly ; AUCUN bras d'architecture avant le 16)** — 2.5M wide `1.2841` nu, 256
  instances, moyennes par instance (sensibles aux extrêmes, non comparables au 1.165
  officiel), erreur en unités de l'erreur saisonnière : total | niveau | forme.
  **m4_hourly** : modèle 1.555 | 1.107 | 1.096 ; copie à 24 pas 1.197 | 0.845 | 0.897 ; copie
  à 168 pas 2.829 | 2.279 | 1.180 (les séries dérivent en une semaine : pas de saison
  hebdomadaire ignorée) ; décalage médian 0 (pas d'erreur de phase). **m4_weekly** : modèle
  2.695 | 2.173 | 1.591 ; dernière valeur 2.865 | 2.506 | 1.668 : séries à tendance, erreur
  de niveau. Hypothèse F formulée sur ces chiffres : médiane, MAD et RevIN sont pris sur
  tout le contexte (40 jours en horaire), donc périmés sur une série qui dérive. Seuils
  énoncés à l'utilisateur AVANT la mesure (pas inscrits ici à temps) : confirmé si le niveau
  de m4_hourly baisse de ≥ 20 % avec un contexte plus court, écarté si < 5 %. **Mesure par
  longueur de contexte**, m4_hourly : 1024 → 1.555 | 1.107 | 1.096 · 512 → 1.591 | 1.297 |
  0.945 · 256 → 1.556 | 1.240 | 0.974 · 96 → 2.727 | 2.321 | 2.353 : le niveau MONTE de 12 à
  17 % quand le contexte raccourcit (la forme gagne 11-14 %, le total ne bouge pas) : F
  ÉCARTÉE sur m4_hourly. m4_weekly : 1024 → 2.695 | 2.173 · 512 → 2.621 | 2.201 · 256 → 2.254 |
  1.741 · 96 → 2.040 | 1.636 : −24 % de total et −25 % de niveau à 96 points : sur l'hebdo à
  tendance, un contexte court est nettement meilleur. Deux configs, deux réponses
  opposées : pas un mécanisme unique, donc pas de bras de 12 h. Bilan de la phase 0 : A
  confirmé (payé par RateIN), B / D / E / F écartés, C non testable sans entraînement ;
  l'erreur de niveau de m4_hourly reste inexpliquée. Piste d'inférence ouverte : la
  longueur de contexte comme second axe de sélection (les flags `+max_context` et
  `+tta_lookbacks` existent dans le harnais). **Prédiction P-SSM.10** (nu, 97 configs,
  `+max_context=256` et moyenne `+tta_lookbacks=256,1024`) : contexte 256 seul, global PIRE
  que le nu (0.8542 / 0.5836) mais W/M/A/Q meilleurs de ≥ 3 % en MASE ; moyenne des deux
  lookbacks : MASE globale ≤ 0.850 et W/M/A/Q meilleurs de ≥ 2 %. ÉCHEC si W/M/A/Q ne gagnent
  pas 1 % dans les deux : l'effet vu sur m4_weekly ne se généralise pas, piste close.

- **2026-10-04 (PHASE 0, `data` : D ET E ÉCARTÉS)** — 2.5M wide `1.2841` au stack, 97 configs,
  MASE relative à FlowState-9.1M en géomoyenne. **D** : 7 configs majoritairement sous 128
  points ×1.056, 86 configs sans contexte court ×1.045 : écart de 1.1 %, sous le seuil
  d'écart de 2 % (et 7 configs < 8). Les 4 configs « en partie courtes » sont à ×1.138, tiré
  par m4_weekly (×1.42, 18 % de séries courtes seulement) : pas un effet de longueur de
  contexte. **E** : Spearman(masse de queue, MASE/FlowState) −0.18, (masse, découplage)
  +0.11 ; part de queue −0.15 / +0.14 ; poids moyen +0.01 / −0.14 : tous sous 0.2 en valeur
  absolue, et le signe sur la MASE est l'inverse de l'hypothèse (les configs à queues
  lourdes, bitbrains 5T ×0.82-1.00, solar ×0.96-1.03, sont celles où l'on tient FlowState).
  Le repère arcsinh n'est pas ce qui coûte la médiane. Bilan de la phase 0 : D et E écartés,
  A confirmé mais payé par RateIN, B écarté comme défaut d'amplitude ; il reste à qualifier
  l'erreur sur m4_hourly et m4_weekly (`decomp`), puis l'écart diffus de ~4.5 % sur les 86
  configs ordinaires, qu'aucune des cinq lectures du code n'explique.

- **2026-10-04 (PHASE 0, `flat` : B ÉCARTÉ sur m4_hourly comme défaut d'amplitude ; A CONFIRMÉ
  sur solar/10T et PAS sur electricity/H — l'aplatissement dépend de la PÉRIODE EN PAS, et
  c'est le mécanisme de RateIN)** — 2.5M wide `1.2841` nu, 256 instances par config,
  amplitude = écart-type par bloc d'une saison / celui des 4 derniers blocs du contexte,
  médiane sur les instances, modèle | vérité. **m4_hourly** (h 48, période 24) : 0.987 |
  1.007 puis 0.992 | 0.995 : rapport 0.98 et 1.00, seuil d'écart (≥ 0.9) atteint. La médiane
  a la bonne amplitude et reste pire que la saisonnalité naïve (MASE nue 1.294) : l'erreur
  est de niveau, de forme ou de phase, pas d'aplatissement. **electricity/H/long** (h 720,
  période 24) : rapport 1.03 au premier jour, 0.99 au pas 408, 0.90 au pas 696 ; variation
  des représentations par pas 0.56 → 0.14. Les représentations convergent bien, mais
  l'amplitude tient 30 cycles : critère A (dernier ≤ 0.6 × premier) NON atteint (0.88).
  **solar/10T/long** (h 720, période 144) : 0.489 | 1.032 au premier jour (rapport 0.47),
  puis 0.088 | 1.116 (0.08) et 0.08 jusqu'au bout : la médiane est PLATE dès le deuxième
  cycle ; critère A atteint (0.17). Lecture : le modèle tient une oscillation de période 24
  pas sur 720 pas et ne tient pas une période de 144 pas plus d'un cycle. Décimer par 6
  ramène 144 à 24 : c'est exactement ce que fait RateIN, et c'est une figure pour le papier
  (amplitude le long de l'horizon, natif contre décimé). Conséquence pour le plan : A est
  réel mais déjà payé par le stack ; B n'est pas un manque de copie d'amplitude. Suite :
  `decomp` (niveau / forme / phase de l'erreur contre les copies naïves aux retards 24 et
  168) sur m4_hourly et m4_weekly, les deux configs qui portent 0.8 pt d'écart.
  `diag_data` reçu tronqué (croisements D et E à relire).

- **2026-10-04 (BRAS S : CLOS SANS GAIN — ni 8 fenêtres ni la température ne bougent le
  stack ; le résidu à l'oracle est un désaccord backtest / test, pas du bruit de sélection.
  PHASE 0, `sn` : 20 configs au niveau de la saisonnalité naïve au stack, borne 0.43 pt)** —
  2.5M wide `1.2841`, flip + pool, 97 configs. Backtest dur 4 fenêtres 0.7668 / 0.5252 ; mix
  8 fenêtres 0.7622 / 0.5195 ; mix 4 fenêtres τ 0.03 : 0.7637 / 0.5207 ; τ 0.08 : 0.7633 /
  0.5203. Références : mix 4 fenêtres τ 0.05 = 0.7631 / 0.5201, RateIN-up = 0.7621 / 0.5194.
  8 fenêtres : −0.06 pt de CRPS pour un backtest deux fois plus long, dans le bruit ; τ : rien
  dans les deux sens. Aucun réglage adopté. Aucune prédiction n'avait été gravée pour ces
  trois évals (elles sont parties dans la même file que le rapport d'écart) : à noter comme
  un manquement à la règle, sans conséquence puisque rien n'est retenu. **Rapport d'écart**
  (backtest dur contre oracle-k, CRPS absolu 0.1324 contre 0.1290, soit 0.5252 contre
  0.5117) : missed 24 configs / 54 % du résidu, wrong_k 20 / 30 %, false_pos 5 / 16 %, match
  48. Dans les plus gros contributeurs le backtest ne manque pas de puissance, il dit
  l'INVERSE du test : loop_seattle/5T/medium k* = 12 gagne 26 % sur le test et PERD 9 % au
  backtest ; ett1/D k* = 3 : −19 % test, +6 % backtest ; bitbrains_fast_storage/H k* = 16 :
  −7 % test, +22 % backtest. Plus de fenêtres ou un mélange plus tranché ne peuvent pas
  corriger un signal de signe opposé : le mélange a déjà pris la moitié de l'écart
  (0.5252 → 0.5201), le reste n'est pas atteignable causalement par ce sélecteur. **`sn`**
  (seuils gravés le même jour : B retenu si ≥ 8 configs ET borne ≥ 0.5 pt, écarté si < 4
  configs ou < 0.2 pt) : au stack 20 configs à MASE ≥ 0.95 × SN dont 14 au-dessus de 1,
  borne oracle 0.7578 (0.43 pt) : ENTRE les deux seuils, ni retenu ni écarté. Au nu 26
  configs, borne 5.48 pt (bizitobs 10S ×2-2.6, solar/10T ×1.7-2.0 : ce que RateIN répare).
  Lecture : la plupart des configs au niveau de SN sont dures pour tout le monde (FlowState
  ×1.00-1.06 sur m4_daily, bitbrains_rnd, bizitobs, solar/10T). L'écart se concentre sur
  DEUX configs : m4_hourly (1.165, FlowState ×1.90) et m4_weekly (1.031, ×1.42), soit
  ln(1.90)/97 + ln(1.42)/97 = 1.0 % de MASE ≈ 0.8 pt sur les 3.6 pt d'écart à FlowState ;
  puis bitbrains_fast_storage/H ×1.19, bizitobs_service/medium ×1.17, solar/H/long ×1.13.
  En attente : `data --run` (D, E), `flat` (A, B sur m4_hourly), bande RateIN-up des
  checkpoints wide.

- **2026-10-04 (PLAN MASE DU 2.5M : cinq défauts lus dans le code, phase 0 de diagnostics —
  SEUILS GRAVÉS AVANT LES CHIFFRES)** — Le 2.5M wide `1.2841` devient le modèle principal
  (0.7621 / 0.5183 avec RateIN-up et γ). Leçon de B5 : corriger ce que RateIN compense ne
  bouge pas le stack ; on vise ce qu'il ne touche pas (H ×1.077 contre FlowState, W/M/A/Q à
  k = 1, m4_hourly pire que la saisonnalité naïve). Correction : bizitobs/10S n'est PAS la
  marge au stack (×1.08) ; le « 5 pt » annoncé le 04/10 portait sur le nu. Défauts lus dans
  `model.py`, `block.py`, `quantile_head.py`, `robust_scale.py`, `prepare_context` : **E** la
  pinball vit dans le repère arcsinh (erreur à z pondérée par 1/√(1+z²), MASE en brut) ;
  **D** contextes d'entraînement ≥ 128 points, éval sur séries de ~30 points (m4 annuel) ;
  **B** futur = token constant, une seule attention croisée par contenu, aucun chemin de
  copie saisonnière ; **A** rollout non piloté (`Re(a) < 0`, représentations futures vers
  un point fixe ; masqué par RateIN) ; **C** ni convolution ni sélectivité, sacrifiées à
  l'équivalence Δ ≡ décimation que l'inférence n'utilise plus (P-SSM.3 jamais lancée).
  Outil : `scripts/diagnose_median.py` (`sn`, `data`, `flat`), 5 tests. **Seuils** (champion
  nu ET stack pour `sn` et `data --run`) : **B** retenu si ≥ 8 configs à MASE ≥ 0.95 × SN et
  borne oracle ≥ 0.5 pt, écarté si < 4 configs ou < 0.2 pt ; **D** retenu si les configs
  majoritairement sous 128 points (≥ 8 configs) perdent ≥ 5 % de plus contre FlowState que
  celles sans contexte court, écarté si < 2 % ; **E** retenu si Spearman(masse de queue,
  MASE/FlowState) ou (masse, découplage) ≥ 0.4, écarté si |ρ| < 0.2 ; **A** confirmé si
  amplitude modèle/vérité du dernier bloc ≤ 0.6 × celle du premier sur electricity/H/long
  et solar/10T/long ; **B** (bis) si modèle/vérité ≤ 0.7 dès le premier bloc de m4_hourly,
  écarté si ≥ 0.9. Vu AVANT de graver, sur 5 configs d'essai sans résultat de modèle :
  m4_hourly a des queues légères (masse 0.15), donc E n'explique pas m4_hourly quoi qu'il
  arrive ; m4_yearly : contexte médian 29 points, 52 % des points cibles à |z| > 2 ;
  us_births/M/short ne compte que 2 instances. Phase 1 : un seul bras de 12 h en
  continuation du 2.5M, initialisé à l'identité, celui que la phase 0 désigne. Plan
  complet dans `../TimeJEPA/PLAN.md`.

- **2026-10-04 (VERDICT P-SSM.9 : ÉCHEC — B5 déplace la médiane nue dans le bon sens mais de 1 pt,
  pas de 5, et le stack y perd ; B5 CLOS)** — Nu (aucune couche), 97 configs, B5 dernier
  `3.0577` contre son départ B1 dernier `3.0559-v1` : global 0.8483 / 0.5765 contre 0.8506 /
  0.5803 (−0.2 pt MASE, −0.4 pt CRPS). **Par terme, MASE nue** B5 | B1 : short 0.7840 | 0.7811
  (+0.3) · medium 0.9227 | 0.9367 (−1.4) · long 0.9588 | 0.9658 (−0.7) ; CRPS nu medium 0.5804
  | 0.5913, long 0.5725 | 0.5783. Par fréquence : 10T 0.976 | 1.028 (−5 %), A −2.5 %, D −1 %,
  H −0.7 % ; W 0.866 | 0.836 (+3.6 %), 10S +2.4 %, M +1.4 %. Couverture nue 0.697 / 0.679 /
  0.668 contre 0.708 / 0.684 / 0.679 (−1 pt). Critère gravé : medium/long ≤ 0.90 non atteint
  (0.923 / 0.959, moyenne 0.941) ; seuil d'échec ≥ 0.94 : atteint. Le signe est celui prédit
  (medium et long gagnent, short non), l'amplitude est cinq fois trop faible, et le stack
  perd 0.3 pt (0.5248 contre 0.5219) : ce que le modèle apprend de la géométrie décimée,
  RateIN le fournissait déjà à l'inférence, et un tiers des batches retirés à la tâche
  native coûte au court terme et aux basses fréquences. Lecture : la médiane longue ne
  se corrige pas par 100 M fenêtres de continuation sur des fenêtres de 1280 pas natifs
  (cible décimée ≤ 768 pas natifs, contexte réduit à 170-384 pas) ; il faudrait des fenêtres
  plus longues (sampler homogène en k, hors budget) ou un décodeur continu. Fait annexe
  mesuré : nu du 10M (B1) 0.8506 / 0.5803 contre nu du 2.5M wide 0.8542 / 0.5836 : la
  capacité vaut 0.3 pt au modèle nu ; l'avance du 10M au stack (0.5219 contre 0.5242) est
  du même ordre. **Bilan des bras d'entraînement sur le 10M** : frac +0.3 pt, B1 0, B5 −0.3
  au stack. Champion inchangé (B1 dernier + stack + RateIN-up + γ, 0.7582 / 0.5160). Suite :
  bras S (en cours), fréquence en entrée, puis cartes.

- **2026-10-04 (B5, TABLE AU STACK SUR 5 CHECKPOINTS : dernier 0.7737 / 0.5248 / couv. 0.705 — le
  stack se DÉGRADE au fil du bras ; critère stack de P-SSM.9 échoué ; éval nue en attente)** —
  Stack flip + mix + pool, 97 configs sans échec : 20 % 0.7682 / 0.5228 / 0.699 · 40 % 0.7709 /
  0.5238 / 0.706 · 60 % 0.7704 / 0.5227 / 0.721 · 80 % 0.7730 / 0.5240 / 0.709 · 100 % 0.7737 /
  0.5248 / 0.705. Départ (B1 dernier) : 0.7668 / 0.5219. Le bras perd 0.3 pt de CRPS et 0.7 pt
  de MASE au stack, de façon monotone en MASE ; B1 et frac gagnaient 0.3 pt sur la même
  durée. Critère « stack ≤ 0.519 » échoué. Le critère principal (MASE nue medium/long) attend
  `logs/nu_dec.out`. Hypothèse à vérifier sur la carte, pas avant : un tiers des batches
  retirés à la tâche native (k = 1, 256 pas) pour une géométrie que le stack traite déjà par
  la décimation à l'inférence ; si le nu medium/long ne gagne pas, le bras est une perte
  sèche. Champion inchangé : B1 dernier checkpoint (`3.0559-v1`) + stack + RateIN-up + γ =
  0.7582 / 0.5160. B5 n'est pas rejoué sur le 2.5M.

- **2026-10-04 (APRÈS LA RELEASE : exploration « entraînement en une passe », R2 dimensionné et
  R3 noté ; prédictions gravées)** — Décision utilisateur : à explorer après les cartes de
  modèle et la publication, pas avant. **R2, réservoir S4D gelé à 2.5 M de paramètres appris** :
  p = 10 000 features gelées (banc de pôles multi-échelle `s = Kx`, puis non-linéarité
  aléatoire gelée), lecture ridge `W = (G + λI)⁻¹C` avec `G = Σφφᵀ` (800 Mo en float64,
  accumulation en double : G mal conditionnée), `C = Σφyᵀ` ; ~3·10¹⁶ opérations pour 300 M
  fenêtres, 1 à 3 h sur une 3090, résolution en secondes — l'équivalent en données de six
  jours × trois cartes du 2.5M appris. Lecture sur une base de fonctions du temps pour un
  horizon libre ; fan par moindres carrés repondérés ou quantiles des résidus ; adaptateur
  pour le harnais (contrat `ExternalForecaster`). **Prédiction** : CRPS nu 0.62 (fourchette
  0.60-0.68), 0.57 avec le stack flip + RateIN ; TimeSSM 2.5M nu 0.584, stack 0.524. Lecture
  du résultat : l'écart mesure ce que vaut l'apprentissage des features à données égales.
  **R3, Recursive Feature Machines** (idée utilisateur) : régression à noyau en formule
  fermée + métrique M mise à jour par la moyenne des produits extérieurs du gradient ; M
  (d × d) se moyenne en flux sur des batches bien composés, le prédicteur reste borné par le
  nombre de centres (~10⁵ sur 24 Go). Deux usages retenus : M comme diagnostic des retards
  utilisés (double saisonnalité horaire), et RFM après canonicalisation du rythme par RateIN.
  Estimation 0.58-0.65, très incertaine. Ordre : R2 puis R3, un week-end chacun.

- **2026-10-03 (BRAS R AJOUTÉ AU PLAN : résolution fermée — réajustement de la dernière couche de
  la tête, et réservoir S4D gelé comme ligne de comparaison ; calendrier jusqu'à la publication)**
  — Question de l'utilisateur : un réseau entraînable en une passe. Réponse consignée : possible
  dès que le modèle est linéaire en ses paramètres appris (features fixes + lecture ridge,
  `W = (ΦᵀΦ + λI)⁻¹ΦᵀY`, accumulable en flux par `G = Σφφᵀ`, `C = Σφyᵀ`) ; TimeSSM tel quel ne
  l'est pas (pôles, Δ, portes, tête à attention appris). Deux expériences en découlent.
  **R1, réajustement de la dernière couche** (inférence + une passe avant, aucun gradient sur
  le corps) : geler le champion, extraire les activations d'entrée de la dernière couche
  linéaire de la tête quantile (dimension 1536) sur ~20 M de fenêtres du corpus, résoudre la
  pinball de cette couche (convexe) par moindres carrés repondérés, ~10 itérations ; variante
  avec poids sur le niveau 0.5 (version sans entraînement du bras W). Attendu : CRPS 0 à
  −0.3 pt (un run annealé est déjà près de l'optimum de sa dernière couche), effet possible
  sur la calibration (γ a montré un fan mal formé en distribution) et sur la médiane.
  Prédiction à graver au lancement. « Couche par couche » n'a pas de sens : les couches
  intermédiaires n'ont pas de cible. **R2, réservoir pur** (ligne de comparaison du papier,
  « ce que vaut l'apprentissage des features ») : banc de p = 4096 pôles gelés sur plusieurs
  échelles (`s = Kx`), non-linéarité aléatoire gelée, lecture ridge sur une base de fonctions
  du temps pour un horizon libre ; estimation 0.60-0.68 de CRPS nu (TimeSSM nu 0.58, TinyCast
  0.545). Ni R1 ni R2 ne sont des conditions de publication. **Calendrier** : B5 finit
  dimanche 04/10 ~14 h (checkpoints toutes les 10 h, plus rapide que prévu : les batches
  décimés ont moins de tokens), éval en file ; puis S, B5 sur le 2.5M si P-SSM.9 tient, bras
  fréquence en entrée, tableaux complets, R1, cartes et soumission avant le 16/10.

- **2026-10-02 (PLAN « MÉDIANE » APPROUVÉ ; BRAS B5 LIVRÉ : fenêtre décimée par un k tiré par
  batch ; P-SSM.9 GRAVÉE)** — Plan (`~/.claude/plans/playful-pondering-dragonfly.md`) : B5
  (entraînement), puis S (sélecteur, inférence : rapport `ratein_selection_gap.py` sur
  backtest contre oracle, puis 8 fenêtres, MIX_TAU en flag, pooling), puis W (pinball pondérée
  sur la médiane, après B5 seulement). Deux faits d'exploration qui ont fixé la conception :
  (1) la multi-résolution du dataset TimeJEPA est un sous-échantillonnage STRIDED (un point
  sur f, ancré à gauche), pas la moyenne par blocs de l'inférence — le bras aug de TimeJEPA
  (2026-09-04, +0.1 pt) s'entraînait sur des entrées différentes de celles de RateIN, ce qui
  éclaire son faible gain ; (2) le collate par défaut exige une longueur de contexte par
  batch et le sampler fractionnaire mélange les familles : une décimation par item au niveau
  du dataset (contexte 1024/f) est impossible sans réécrire le sampler. D'où une décimation
  PAR BATCH dans le module, à l'intérieur de la fenêtre de 1280 : cible = les 256·k derniers
  pas natifs moyennés par blocs (256 pas décimés, le rollout natif), contexte = le reste,
  moyenné et aligné à droite (`_block_mean`, égal à `ratein.decimate`, test). Toutes les
  fenêtres sont éligibles ; k ∈ {1, 2, 3}, p 0.5, plancher de contexte décimé 128 (exclut
  k = 4). Horizon natif couvert 256 / 512 / 768 ; contexte décimé 384 / 170. Différence avec
  B1 (clos) : le rollout reste à 256, c'est l'échelle de l'entrée qui change, avec l'opérateur
  de l'inférence. `SSMFinetuneModule._maybe_decimate` (train seulement, masque propagé par
  ET de blocs), témoins `aug/decimation_k`, `aug/decimation_neq1_frac` (≈ 0.33 attendu),
  `geometry/horizon_native` ; 7 tests (égalité à `ratein.decimate`, identité à p = 0 / k = 1,
  formes et alignement, jamais en eval, plancher de contexte, masque, crop après décimation,
  les trois tirages Δ / horizon / k ensemble). Config `ssm_mid_v3_dec` (= frac + les trois
  clés ; horizon aléatoire à 0 ; Δ conservé), départ = dernier checkpoint de B1
  (`3.0559-v1`), SEED0 obligatoire. **P-SSM.9** : dernier checkpoint, MASE nu medium/long
  0.95 → ≤ 0.90 ; CRPS stack des H medium/long meilleur que B1 et le mélange pèse plus sur
  k > 1 sur les H ; stack ≤ 0.519 ; short pas pire que 0.5486. ÉCHEC si MASE nu medium/long
  ≥ 0.94. Hors budget noté : décimation par item (demande un sampler homogène en f).

- **2026-10-02 (LES k DU STACK SUR LES 42 CONFIGS LONGUES, et le nu point par point : le bras
  « rollout entraîné dans l'espace décimé » a sa cible)** — Poids du mélange RateIN par
  config medium/long (2.5M wide `1.2841`) : sub-horaire (5T, 10T, 10S, 15T) k dominant 3-12,
  jusqu'à 32-48 sur bizitobs_l2c/5T ; horaire k 1-3 (electricity/H medium : k = 1 à 97 %,
  solar/H, m_dense/H, loop_seattle/H : k 1-2) sauf ett2/H (12-32) et bizitobs_l2c/H (8-12).
  Horizon EFFECTIF après décimation : 480-900 pas natifs deviennent 40-240 pas décimés sur
  le sub-horaire (dans ou sous l'horizon d'entraînement de 256), mais restent 240-720 sur
  l'horaire à k 1-3. Donc les configs H medium/long sont celles où le rollout autonome
  extrapole encore 2-3× au-delà de l'entraînement, et c'est là que FlowState garde
  l'avantage (H ×1.077 en MASE). Nu point par point : bizitobs_service/10S/long MASE 3.57
  (SN locale 1.37) : CRPS 0.088 contre 0.056 — la médiane nue est 2.6× pire que la
  saisonnalité naïve ; solar/10T/long MASE 1.75 (SN 0.87), CRPS 0.62 contre 0.43. À 720-900
  pas sans décimation le modèle est hors de sa zone, confirmé. **Bras candidat B5
  (entraînement, à instruire en mode plan)** : multi-rythme sur la CIBLE — tirer k par
  batch, décimer contexte ET cible par k (moyenne par blocs, la même que `ratein.decimate`),
  prédire 256/k à 256 pas décimés ; le modèle voit alors en entraînement exactement ce que
  RateIN lui présente, et des cibles qui couvrent 256·k pas natifs (jusqu'à 1024-3072)
  sans allonger la fenêtre. Deux variables à ne pas confondre : (i) la décimation de la
  fenêtre (nouveau) et (ii) les horizons longs en pas décimés (déjà couvert par 256).
  Prédiction à graver : le nu medium/long bouge (MASE 0.95 → < 0.90) ET le stack H
  medium/long bouge (le sélecteur choisit alors k > 1 plus souvent sur l'horaire), sans
  dégrader le short. La fenêtre de 1280 pas décimée par k demande 1280·k pas natifs : ce
  bras exclut les fenêtres trop courtes pour k > 1 (le datamodule a déjà
  `multi_resolution_factors`, à relire avant d'écrire du code : TimeJEPA l'a eu en pretrain).

- **2026-10-02 (CARTE NU / FLIP / STACK EN MASE, 2.5M wide `1.2841` : le modèle nu perd 17.6 %
  de MASE contre FlowState, le stack en rend 10 ; la médiane nue décroche sur les HORIZONS
  LONGS et le sub-horaire, et le stack y fait presque tout)** — MASE par terme, nu | flip |
  stack : short 0.784 | 0.767 | 0.749 · medium **0.946** | 0.919 | 0.790 · long **0.966** | 0.949 |
  0.797. Horizon > 480 : nu 1.046 (PIRE que la saisonnalité naïve), stack 0.815. Le modèle
  nu est un modèle de court terme ; sur medium et long c'est RateIN (décimation → moins de
  pas à extrapoler) qui ramène la médiane de 0.95 à 0.79. CRPS nu par terme 0.578 / 0.597 /
  0.585 : le fan nu tient mieux que la médiane nue, et c'est l'inverse du stack (0.548 /
  0.505 / 0.485). Par fréquence, nu/FlowState en MASE : 10S ×2.35, 10T ×1.32, H ×1.15, W
  ×1.15, A ×1.25 ; 5T ×1.10, D ×1.05. Web/CloudOps nu ×1.48 (stack ×1.03 : RateIN y récupère
  presque tout), Econ/Fin ×1.28 (le stack n'y touche pas : W/M/A/Q, k = 1). Nu contre
  FlowState : 15/97 gagnées, ×1.176 ; stack : 27/97, ×1.049. Dix pires en nu : bizitobs ×2-3,
  m4_hourly ×2.11, solar/10T/long ×2.06 : rafales et sub-horaire à très long horizon (600-900
  pas, 2.5-3.5× l'horizon d'entraînement). Couverture nu 0.724 / 0.691 / 0.668, flip 0.738 /
  0.719 / 0.704 : le flip calibre plus que le mélange. **Lecture** : (1) RateIN n'est pas une
  rustine, c'est 60 % de la MASE du modèle sur les horizons longs ; (2) la médiane nue
  perd là où le rollout autonome est le plus long par rapport à l'entraînement, ce que
  l'oracle-k à 1.25 pt (décimation = horizon plus court en pas) et la carte « long = notre
  meilleur terme au stack » disaient déjà ; (3) ce que FlowState fait de mieux, c'est une
  médiane qui tient à 900 pas SANS décimation externe, par son décodeur en base de
  fonctions, continu en temps : la fréquence déclarée lui donne l'échelle, et son décodeur
  n'extrapole pas pas à pas. Pistes pour la médiane, à instruire en mode plan : (a) un
  rollout entraîné aux horizons de GIFT dans l'espace DÉCIMÉ (apprendre à prédire 48 pas
  d'une série décimée par k, en tirant k à l'entraînement : le multi-rythme sur la cible
  et pas seulement sur Δ) ; (b) le sélecteur, 1.25 pt d'oracle, 4 fenêtres déjà prises,
  reste 0.8 pt ; (c) un terme MAE sur la médiane dans la pinball, pour les horizons courts
  où le stack ne fait rien (W/M/A/Q, Econ/Fin ×1.28, Sales).

- **2026-10-02 (CARTE DE B1 : aucune redistribution, le short est IDENTIQUE au frac — B1 CLOS)**
  — Par terme, B1 | frac : short 0.5487 | 0.5486 · medium 0.4951 | 0.4945 · long 0.4825 |
  0.4809. Par fréquence, écarts de ±0.5 % sauf 10S (+2.7 %, bizitobs_application/10S/short
  ×2.73 contre Toto, 0.806 : la config la plus instable du benchmark d'un checkpoint à
  l'autre) et A (−1.7 %). Couverture 0.715 / 0.712 / 0.698 contre 0.717 / 0.705 / 0.688 : +0.1
  à +1 pt, dans le bruit. Verdict : les cibles courtes à l'entraînement n'ont déplacé ni le
  court terme (critère short ≤ 0.542 manqué : 0.5487), ni la calibration, ni rien. P-SSM.6b
  échouée sur ses trois critères. Lecture : la pinball sur 256 pas n'était pas ce qui
  limitait le court terme ; le modèle prédit déjà le futur proche aussi bien qu'il le
  peut avec ce qu'il a appris. L'horizon d'entraînement rejoint 1024/256 et l'univarié
  dans les non-leviers mesurés. Le dernier checkpoint de B1 reste le champion par la
  bande (même bande que le frac, 100 M fenêtres de plus), c'est lui qui porte les couches.

- **2026-10-02 (NUIT DE VERDICTS : B1 sur le 10M = 0.7668 / 0.5219 / couv. 0.711, P-SSM.6b
  MANQUÉE (bande identique au frac) ; ABLATION de RateIN-up : les 4 FENÊTRES font tout le
  gain ; nu / flip du 2.5M mesurés ; 10M + up + γ = 0.7582 / 0.5160, NOUVEAU CHAMPION)** —
  **B1 (`ssm_mid_v3_hrand`, 97 configs)** : 20 % 0.7696 / 0.5237 / 0.688 · 40 % 0.7696 /
  0.5226 / 0.726 · 60 % 0.7647 / 0.5195 / 0.720 · 80 % 0.7671 / 0.5218 / 0.711 · 100 %
  0.7668 / 0.5219 / 0.711. Trajectoire superposée à celle du frac (0.5245 / 0.5223 / 0.5194 /
  0.5207 / 0.5213) à 0.1 pt près : l'horizon aléatoire n'a rien changé au stack global ni
  à la couverture (0.711 contre 0.708 ; le 0.800 de h512 ne se reproduit pas sur le SSM).
  Critères gravés : stack ≤ 0.517 manqué, couverture ≥ 0.76 manquée ; carte par terme à
  faire pour le critère short (redistribution possible), mais le global dit déjà qu'un
  second passage + horizon aléatoire vaut un second passage. Troisième bras de continuation
  sur le 10M, troisième bande 0.519-0.522 : la recette plafonne, et le même seed a rejoué
  les mêmes 100 M fenêtres (confusion consignée le 01/10). **Ablation RateIN-up sur le 2.5M
  wide `1.2841`** (stack 0.5242 / 0.7679) : bt_windows=4 seul **0.5201 / 0.7631** ; k_up seul
  0.5239 / 0.7678 (rien) ; min_bt=4 seul 0.5251 / 0.7682 (légèrement pire) ; les trois
  ensemble 0.5194 / 0.7621. Le gain de RateIN-up est à 90 % celui des quatre fenêtres de
  backtest, une meilleure ESTIMATION de la sélection ; les candidats k < 1 n'apportent
  rien seuls et le backtest court nuit seul : mon mécanisme B3′ est RÉFUTÉ sur ses deux
  composantes. Garder `ratein_bt_windows=4` comme couche officielle candidate (à
  confirmer sur un second checkpoint) ; k_up et min_bt restent des options documentées,
  non recommandées. **Nu / flip du 2.5M wide** : nu 0.8542 / 0.5836, flip 0.8351 / 0.5644,
  stack 0.7679 / 0.5242 : le flip vaut 1.9 pt de CRPS et 1.9 pt de MASE, RateIN mix+pool 4.0
  et 6.7 pt. H1 (le mélange lisse la médiane) est RÉFUTÉE : le stack gagne PLUS en MASE
  qu'en CRPS. La médiane qui décroche contre FlowState est dans le modèle nu (0.854 contre
  0.726), pas dans la couche. **Combinaisons** : 2.5M up + γ 0.7621 / 0.5183 (γ ajoute 0.1 pt
  sous up, additif) ; 10M B1 dernier + up 0.7582 / 0.5170 ; + son γ recalibré **0.7582 /
  0.5160**. Champion du projet : TimeSSM-10M (B1 dernier checkpoint, stack + RateIN-up + γ)
  0.7582 / 0.5160, à 1.4 pt de FlowState-9.1M (0.5019 / 0.7262) en CRPS et 3.2 pt en MASE ;
  devant TTM-R3-PT (0.5195) et Toto-2.0-4m (0.5242). Nu, flip et stack du 10M à publier à
  côté (nu et flip du 10M encore à mesurer). **Lecture** : trois bras de continuation du 10M
  ont acheté 0.3 pt (0.5245 → 0.5219) ; les couches d'inférence 0.6 pt (0.5219 → 0.5160).
  L'entraînement sur cette recette est au plateau ; ce qui reste de marge mesurée est le
  sélecteur (oracle 1.25 pt) et la médiane du modèle nu.

- **2026-10-01 (B1 REJOUE LES BATCHES DU BRAS FRAC : même seed de données ; facteur de confusion
  consigné, run non relancé)** — L'utilisateur a remarqué que la val_wql de `ssm-mid-v3-hrand`
  suit presque le tracé de `ssm-mid-v3-frac`. Cause : la boucle de relance ne change le seed
  qu'après un plantage ; en première tentative les deux bras ont pris le seed de la config
  (420) avec le même sampler fractionnaire et la même fraction, donc le même générateur
  (seed + époque·1000 + rang) et LA MÊME SUITE D'INDICES : B1 revoit les 100 M fenêtres du
  bras frac dans le même ordre, avec l'horizon re-tiré une fois sur deux. Le checkpoint de
  départ (`3.0571`, dernier du frac) et le sampler sont corrects. Impact : B1 = horizon
  aléatoire + second passage sur le même sous-ensemble ; les fenêtres se chevauchant à pas
  8, un second passage vaut presque des données fraîches (régime « corpus répété »), mais
  c'est une confusion de plus à lire avec P-SSM.6b : un second passage n'a aucune raison de
  déplacer le court terme plus que le moyen, la carte par terme reste le juge. Décision : pas
  de relance (run à mi-parcours). Correctif : `SEED0=<seed>` dans `train_ssm_loop.sh` fixe le
  seed de la première tentative (les reprises prennent SEED0 + tentative) ; en-tête des
  configs de continuation : toujours un seed jamais utilisé par le run repris.

- **2026-09-30 (CARTE MASE : où la MÉDIANE perd — l'horaire, les doubles saisonnalités, et les
  m4 ; découplage médiane/fan mesuré)** — `gift_gap_ssm.py --metric mase`, 2.5M wide + RateIN-up
  (0.7621) et 10M frac (0.7674) contre FlowState-9.1M (0.7262), Toto-2.0-4m (0.7565),
  Kairos_10m (0.7527). **Global** : nous/FlowState ×1.049 (27/97 gagnées), nous/Toto ×1.007
  (40/97), nous/Kairos ×1.013 (32/97). Contre Toto le MASE suit le CRPS (short ×1.040 perdu,
  medium ×0.983 et long ×0.950 gagnés) ; contre FlowState la médiane perd sur TOUS les
  termes (×1.059 / 1.043 / 1.031). **Par fréquence contre FlowState** : H ×1.077 avec 5/31
  gagnées — le plus gros groupe du benchmark, perdu presque en bloc sur la médiane alors
  que le CRPS y est à ×1.048 ; W ×1.098, A ×1.143, 10S ×1.082, Q ×1.065, M ×1.053, D ×1.041 ;
  seule 5T gagne (×0.979). Domaine Econ/Fin ×1.237 (les m4), Sales ×1.084. **Dix pires en
  MASE** : m4_hourly ×1.90 (nous 1.165, PIRE que la saisonnalité naïve ; eux 0.613), m4_weekly
  ×1.42, electricity/W ×1.26, bizitobs_application/10S/short ×1.26, car_parts/M ×1.22,
  bitbrains_fast_storage/H ×1.19, m_dense/H medium et long ×1.17, bizitobs_service/10S/medium
  ×1.17, loop_seattle/5T/medium ×1.17. **Découplage (MASE rel / CRPS rel)** : m4_hourly 1.52,
  covid_deaths/D 1.43 (fan bien meilleur que FlowState ×0.69, médiane égale ×0.99), m4_weekly
  1.18, bitbrains_fast_storage/H 1.18, jena_weather/10T 1.15, bitbrains_rnd/5T, bizitobs
  long, jena/H/long 1.11-1.13. Trait commun des configs où la médiane décroche : horaire ou
  sub-horaire à DOUBLE saisonnalité (jour + semaine : m4_hourly, bitbrains/H, jena/H,
  m_dense/H, loop_seattle/5T), plus les séries compétition m4 sans cousin au corpus.
  **Hypothèses à tester, dans l'ordre du coût** : (H1) le mélange de rythmes (Vincentization
  de fans à k différents) LISSE la médiane : un fan couvert et une médiane amortie sur les
  séries à pics — la table nu / flip / stack en MASE par config le dit sans entraînement
  (les évals nu et flip du 2.5M wide `1.2841` sont à faire : ~1 h chacune) ; (H2) la médiane
  ne porte pas le cycle hebdomadaire dans 1024 pas horaires (6 cycles) : erreur par pas
  (jours 1 et 2 de l'horizon 48) sur les configs H, à instrumenter dans le harnais ; (H3)
  la pinball donne 1/9 du gradient à la médiane : terme MAE auxiliaire ou pondération du
  niveau 0.5 (bras d'entraînement, après B1). Rappel : les seuls leviers qui aient bougé le
  MASE sur ce projet sont le mélange de données (−0.5 pt) et RateIN-up (−0.6 pt).

- **2026-09-30 (CORRECTION DE CIBLE : FlowState sous 10M = 0.5019, pas 0.4866 ; premiers
  checkpoints de B1 sur le 10M)** — Relevé par l'utilisateur, vérifié : FlowState-r1.1
  (0.4866) et Granite-FlowState-r1.1 (0.4901) font ~18.5M de paramètres ; le seul FlowState
  sous 10M est FlowState-9.1M (0.5019 / 0.7262). Toutes les mentions « FlowState 9M à
  0.4866 / 0.487 » de ce registre (dont « hors de portée du 10M », 2026-09-17) portaient le
  chiffre de r1.1 ; les cartes `gift_gap_ssm.py` comparaient déjà au bon fichier
  (`FlowState-9.1M.csv`). Position réelle dans la classe ≤ 10M zero-shot : FlowState-9.1M
  0.5019 · TimeSSM-2.5M + RateIN-up 0.5194 · [TTM-R3-PT 0.5195, a vu GIFT] · TimeSSM-10M
  frac 0.5213 (10.1M, juste au-dessus du seuil) · Toto-2.0-4m 0.5242 · TimeSSM-2.5M stack
  0.5242. Écart à la barre : 1.75 à 1.9 pt, pas 3.5. L'objectif « 0.495-0.50 avec le 10M »
  était calé sur la mauvaise barre ; 0.51 mettrait le 10M à 0.8 pt de FlowState-9.1M, et
  le second tour (RateIN-up + γ sur le 10M, +0.5 pt mesuré sur le 2.5M) vise ~0.515.
  Détail au registre TimeJEPA du même jour (`86e1f34`). **B1 (`ssm-mid-v3-hrand`)**,
  validation à h = 256 : val_wql 0.334 puis 0.333 aux deux premiers checkpoints, contre
  0.328 à la fin du bras frac dont il part. La remontée de 0.006 au départ est celle d'un
  cosinus qui repart à 1e-4 depuis des poids annealés (le bras frac avait commencé à 0.335
  de la même façon) ; et la validation ne mesure que l'horizon 256, auquel le bras ne
  consacre plus qu'un batch sur deux : les cibles courtes ne peuvent pas la faire baisser,
  elles ne s'y voient pas. Le témoin de B1 est la carte par terme sur GIFT, pas val_wql.

- **2026-09-29 (CARTE DE RateIN-up SUR LE 2.5M WIDE : P-SSM.7 tenue sur le global, MANQUÉE sur
  ses deux critères de mécanisme — le gain est DIFFUS, pas celui des basses fréquences)** —
  `gift_gap_ssm.py`, `gift_flip_ratein-mix-pool-up234-bt4-w4` contre
  `gift_flip_ratein-mix-pool`, 97 configs. **Par terme** (up | référence) : short 0.5416 |
  0.5476 (−0.60 pt) · medium 0.5033 | 0.5053 (−0.20) · long 0.4805 | 0.4851 (−0.46). **Par
  fréquence**, variation du CRPS ratio : 10S −3.7 %, A −1.8 %, 10T −1.8 %, 5T −1.3 %, W
  −1.2 %, D −1.2 %, Q −0.7 %, H −0.4 %, M −0.3 % (MASE M 0.816 → 0.836, pire), 15T +0.3 %.
  Le groupe W/M/A/Q gagne ~1 %, pas les ≥ 5 % gravés ; contre FlowState W reste à ×1.112,
  A à ×1.116. Les configs à h ≥ 16 ne sont PAS inchangées : part d'instances à k ≠ 1 medium
  0.62 → 0.86, long 0.90 → 0.81 (les 4 fenêtres de backtest changent la sélection partout).
  **Régressions** : electricity/W/short 0.642 → 0.754 (+17 %, devient la 2e pire config
  contre Toto, ×1.36), car_parts/M et saugeen/D entrent dans les dix pires, domaine Sales
  0.4354 → 0.4518 (+3.8 %, MASE 0.712 → 0.752) : sur des backtests de 8 à 12 pas, le
  sélecteur choisit parfois mal (malédiction du vainqueur, déjà vue en v2.1). Couverture
  short 0.717 → 0.705. **Verdict** : critère global tenu (0.5194 ≤ 0.521) ; critère de
  groupe manqué ; critère « le reste inchangé » manqué. Mon mécanisme (« le sélecteur est
  éteint sur les basses fréquences, c'est là qu'on perd 13 % ») n'explique qu'une petite
  part du gain : RateIN-up gagne surtout par une sélection mieux estimée PARTOUT. Les trois
  flags sont confondus dans cette mesure ; ablation à faire, une éval chacun : (a)
  `ratein_bt_windows=4` seul, (b) `ratein_k_up=2x3x4` seul, (c) `ratein_min_bt=4` seul.
  GARDE-FOU DE MÉTHODE : ne dessiner AUCUNE garde (marge plus haute sous h_bt < 16,
  exclusion de familles) à partir de ces résultats par config, ce serait régler sur le
  test ; une garde se justifie par le diagnostic du backtest (table `ratios`, nombre de
  fenêtres), pas par le score GIFT. L'écart aux basses fréquences contre FlowState reste
  ouvert : ni la capacité, ni le mélange, ni le sélecteur ne le ferment ; B1 (cibles
  courtes) est le dernier bras qui le vise.

- **2026-09-29 (B2 SUR LE 2.5M WIDE : 0.7679 / 0.5232 / couv. 0.765 — P-SSM.8 TENUE au bord
  bas : couverture 0.717 → 0.765, CRPS −0.10 pt, MASE identique)** — Stack +
  `quantile_gamma` (γ calibré sur la validation du corpus, ×flip), 97 configs sans échec.
  q10 0.143 → 0.117, q90 0.860 → 0.882 : l'intervalle 80 % passe à 0.765 (critère ≥ 0.76
  tenu), le CRPS gagne 0.10 pt (critère −0.1 à −0.3 : bord bas ; l'écriture « ≤ 0.523 »
  du même critère est manquée de 0.02 pt, arrondi), MASE bit-identique comme prévu.
  Lecture : la moitié du déficit de couverture était bien un fan mal formé en
  distribution, réparable hors test ; le reste (0.765 contre 0.80) est du décalage.
  Contrairement à TimeJEPA (G4.2 neutre), γ PAIE sur le SSM, peu en CRPS, beaucoup en
  calibration. Tableau des couches sur `1.2841` : stack 0.7679 / 0.5242 / 0.717 · + γ 0.7679
  / 0.5232 / 0.765 · + RateIN-up 0.7621 / 0.5194 / 0.699 · oracle-k 0.7564 / 0.5117 /
  0.713. Les deux couches sont complémentaires par construction (RateIN-up gagne le CRPS
  et resserre le fan, γ l'élargit sans toucher la médiane) : l'éval combinée
  (stack + up + γ) reste à faire, γ étant à RECALIBRER sous RateIN-up (le fan calibré
  n'est pas le même). Outil : `gift_gap_ssm.py` fusionnait deux variantes d'un même
  checkpoint en une colonne (clé run/checkpoint) ; clé = checkpoint/tag, test ajouté ; la
  carte avant/après de RateIN-up est à relancer.

- **2026-09-29 (2.5M WIDE `1.2841` : ORACLE-k 0.7564 / 0.5117 ; RateIN-up 0.7621 / 0.5194 /
  couv. 0.699 — P-SSM.7 : critère global TENU (≤ 0.521), critère de groupe en attente de la
  carte)** — 97 configs sans échec pour les deux. **Oracle-k** (`+tta_flip +ratein=oracle`,
  décimation seule, k ≥ 1, diagnostic) : 0.5117, 33/97 configs gagnent > 5 % contre k = 1 ;
  écart au stack de référence (0.7679 / 0.5242) : 1.25 pt de CRPS, 1.15 pt de MASE. La règle
  gravée (« oracle − stack ≥ 1 pt → le sélecteur est le levier principal ») est remplie :
  sur le SSM comme sur TimeJEPA (1.5 pt sur head8), la plus grande marge mesurée du projet
  est dans le choix du rythme, pas dans le modèle. **RateIN-up** (stack + `ratein_k_up=2x3x4
  ratein_min_bt=4 ratein_bt_windows=4`) : 0.5194, soit −0.48 pt de CRPS et −0.58 pt de MASE
  sur le stack, 38 % de l'écart à l'oracle capturé ; part d'instances à k ≠ 1 : 50 % →
  56.7 % ; couverture 0.717 → 0.699 (le fan se resserre encore : −1.8 pt, à surveiller avec
  B2). NB : l'oracle ne contient pas les candidats k < 1, RateIN-up peut donc le dépasser sur
  les configs courtes ; le plafond réel est au-dessus de 0.5117. Le 2.5M avec RateIN-up
  égale le meilleur checkpoint du 10M frac au stack de référence (0.5194) : la couche
  d'inférence vaut autant que le passage à 10M + bras frac. Reste à lire sur la carte
  (`gift_gap_ssm.py` référence contre up) : le gain vient-il des 17 configs basse fréquence
  (critère ≥ 5 %) ou des 4 fenêtres de backtest sur l'ensemble ? B2 (γ) : résultat attendu
  dans `logs/b2.out`.

- **2026-09-29 (B1 PASSE DIRECTEMENT SUR LE 10M — décision utilisateur ; P-SSM.6b GRAVÉE ;
  deux pannes d'outillage des évals B2 et B3′ corrigées)** — Pannes : (1) B2, le tag du
  fichier de log était construit depuis la chaîne STACK et contenait le chemin absolu du
  JSON de γ : chaque `tee` échouait (« No such file or directory »), table à nan / 0
  config ; `eval_checkpoints_ssm.sh` réduit maintenant toute valeur-chemin à son basename.
  (2) B3′, sortie en 10 s : `+ratein_k_up=2,3,4` non quoté est un SWEEP pour Hydra, le run
  meurt avant main ; le flag accepte `x` et `-` comme séparateurs (`+ratein_k_up=2x3x4`,
  TimeJEPA `6b454d0`). Aucune des trois évals d'inférence n'a donc encore tourné.
  **Décision** (utilisateur, assumée comme non propre) : avec 3 GPU un bras from scratch
  n'est pas abordable, et le 10M a vu moins de pas que le 2.5M (259 k + 90 k contre 505 k) ;
  B1 se fait donc en continuation du 10M frac, pas sur le 2.5M. Config `ssm_mid_v3_hrand` =
  `ssm_mid_v3_frac` + `horizon_lengths [32..512]` p 0.5, 100 M fenêtres (fraction 0.04316),
  LR 1e-4. Départ = DERNIER checkpoint du bras frac (`3.0571`, 0.5213), pas le meilleur au
  stack (0.5194 à 60 %) : choisir le départ sur le score GIFT serait sélectionner sur le
  test, et l'écart est dans le bruit. Identification : le bras frac (même recette SANS
  horizon aléatoire) n'a bougé le short que de −0.26 pt (0.5512 → 0.5486) ; ce que ce bras
  y ajoute est attribuable à l'horizon. **P-SSM.6b** : dernier checkpoint, stack ≤ 0.517
  (−0.4 pt), short 0.5486 → ≤ 0.542, couverture 80 % ≥ 0.76, medium et long pas pires que
  0.4945 / 0.4809. ÉCHEC si stack ≥ 0.521 ou short ≥ 0.5486. P-SSM.6 (version 2.5M) n'est
  pas courue. File : `scripts/queue_b2_b3_oracle_then_hrand.sh` lance B2, B3′ et oracle-k
  sur le 2.5M wide (une carte chacun), attend leurs PID par `wait`, puis lance B1 sur les
  trois cartes ; `EVALS_ONLY=1` pour s'arrêter après les évals.

- **2026-09-29 (CARTE AVANT/APRÈS DU BRAS FRAC : la capacité et le bon mélange paient sur
  MEDIUM et LONG, pas sur le COURT terme ; les basses fréquences ne reviennent qu'à moitié)**
  — `gift_gap_ssm.py`, 10M frac `3.0571` contre 10M entier `3.0672-v1` (et 2.5M wide `1.2841`
  de la carte du 26/09). **Par terme**, CRPS ratio 2.5M wide | 10M entier | 10M frac : short
  0.5476 | 0.5512 | 0.5486 · medium 0.5053 | 0.4975 | 0.4945 · long 0.4851 | 0.4855 | 0.4809.
  Le 10M frac gagne 1.1 pt sur medium et 0.4 pt sur long contre le 2.5M, et RIEN sur short
  (0.5486 contre 0.5476). Contre Toto : short ×1.040 (20/55), medium ×0.949 (14/21), long
  ×0.927 (15/21), total ×0.995, 49/97 gagnées ; contre FlowState ×1.057 / ×1.021 / ×1.008,
  total ×1.039, 34/97 ; contre TTM-R3-PT ×1.003. **Basses fréquences**, 2.5M | entier | frac :
  W 0.644 | 0.666 | 0.658 · M 0.764 | 0.809 | 0.783 · A 0.886 | 0.950 | 0.945 · 10S 0.735 |
  0.803 | 0.771. Le sampler fractionnaire récupère un tiers à la moitié de ce que le sampler
  entier avait perdu, en 100 M fenêtres ; le 10M reste derrière le 2.5M sur W/M/A/10S.
  **Couverture** inchangée (0.717 / 0.705 / 0.688). **Queue** : bizitobs_application/10S/short
  ×2.31 contre Toto (0.681 ; le 2.5M y fait 0.529 : config instable d'un modèle à l'autre),
  puis bizitobs_service, bitbrains_fast_storage ×4, covid_deaths, electricity/W, solar/10T.
  Cousin au corpus : ×0.961 avec, ×1.012 sans. **Lecture** : l'écart restant à FlowState
  (3.9 %) est aux deux tiers dans le court terme (55 configs à ×1.057) ; la capacité n'y
  touche pas. Les deux bras qui visent le court terme sont exactement ceux qui restent :
  B3′ (sélecteur éteint sous h = 16, inférence seule) et B1 (cibles courtes). Ordre
  confirmé : B2 / B3′ / oracle-k en cours sur le 2.5M wide, puis B1 ; le bras gagnant se
  reporte sur le 10M frac.

- **2026-09-29 (BRAS FRAC DU 10M, TABLE À 97 SUR 5 CHECKPOINTS : dernier 0.7674 / 0.5213 / couv.
  0.708, meilleur 0.7645 / 0.5194 (60 %) — P-SSM.5 MANQUÉE DE 0.13 PT À LA LETTRE (≤ 0.520 au
  dernier checkpoint), loin de l'échec (≥ 0.524) ; NOUVEAU CHAMPION par le chiffre, bande
  0.519-0.521)** — Stack flip + mix + pool : 20 % 0.7760 / 0.5245 / 0.697 · 40 % 0.7696 /
  0.5223 / 0.689 · 60 % **0.7645 / 0.5194** / 0.713 · 80 % 0.7666 / 0.5207 / 0.708 · 100 %
  0.7674 / 0.5213 / 0.708. Départ (poids du 10M entier) : 0.7722 / 0.5245. Gain du bras :
  −0.3 à −0.5 pt de CRPS, −0.5 à −0.8 pt de MASE ; le 10M bat maintenant le 2.5M wide sur les
  deux axes (0.5242 / 0.7679), de 0.3-0.5 pt et 0.1-0.3 pt. Le mélange ET les pas
  supplémentaires comptaient (les deux changent dans ce bras : non séparés). Bande des trois
  derniers 0.19 pt : on publie la bande, pas le 0.5194. Règle du second bras (gravée le
  25/09 : le stack descend-il encore entre 80 et 100 % ?) : NON (0.5207 → 0.5213), malgré une
  val_wql interne qui descend jusqu'au bout (0.335 → 0.328 ; validation proportionnelle,
  non comparable à celle du run entier). Pas de second bras frac. Suite : carte avant/après
  (short et basses fréquences revenus ?), puis B2 / B3′ / oracle-k sur le 2.5M wide, puis
  B1. Incident d'outillage : la file de nuit (`bash -c` avec `while pgrep -f
  eval_all_gpus_ssm.sh`) ne s'est jamais déclenchée — `pgrep -f` trouvait la ligne de
  commande du `bash -c` lui-même, qui contient le motif ; une attente sur un PID ou un
  fichier témoin est la forme correcte. Nuit de GPU perdue sur les trois évals d'inférence.

- **2026-09-26 (B2, CALIBRATION CQR DU 2.5M WIDE `1.2841` SUR LA VALIDATION DU CORPUS, ×flip :
  le fan est un peu étroit EN DISTRIBUTION et surtout mal FORMÉ ; P-SSM.8 gravée)** —
  `calibrate_ssm.sh`, 192 fenêtres par jeu, h = 256, 100 jeux. Couverture avant, médiane par
  jeu : q10 ≈ 0.13, q90 ≈ 0.88, intervalle 80 % ≈ 0.75 en distribution, contre 0.717 sur GIFT
  short et 0.685 sur GIFT long : la moitié du déficit de couverture est un fan étroit
  (0.75 contre 0.80), l'autre moitié du décalage (0.75 → 0.70). Différence avec TimeJEPA
  (G4.2 : γ 1.01-1.12, neutre) : ici γ est NON MONOTONE — q10 1.136, q90 1.154 (les niveaux
  extrêmes à élargir de 14-15 %), mais q20 0.908, q60 0.914, q80 0.885 (les niveaux
  intérieurs à RESSERRER). La tête pinball produit un fan à épaules lourdes et queues
  courtes. Par jeu : `synthetic_ops_bursty` γ q90 1.4-2.85 (les rafales vers le haut ne
  sont pas couvertes — le domaine CloudOps de bizitobs/bitbrains, encore), climat
  era5/cmip6 1.3-1.6 des deux côtés, bitcoin q90 4.00 (borné). JSON :
  `TimeJEPA/evaluation/calibration/gamma_epoch00_valloss1.2841_flip.json`. **P-SSM.8**
  (inférence seule, stack + `+quantile_gamma=<json absolu>`, 97 configs) : couverture 80 %
  ≥ 0.76 (contre 0.717), CRPS −0.1 à −0.3 pt (stack ≤ 0.523), MASE bit-identique par
  construction. ÉCHEC si le CRPS monte : le fan est miscalibré différemment sur GIFT et en
  distribution, et γ ne se règle pas hors test. Commande : `STACK="+tta_flip=true
  +ratein=mix +ratein_pool=true
  +quantile_gamma=/workspace/TimeJEPA/evaluation/calibration/gamma_epoch00_valloss1.2841_flip.json"
  ONLY=epoch00_valloss1.2841 scripts/eval_checkpoints_ssm.sh
  checkpoints/timessm_mini_v3_wide_zs/pretrain_False +gift_batch_size=32`.

- **2026-09-26 (B3′ = RateIN-up, LIVRÉ dans le harnais TimeJEPA ; P-SSM.7 GRAVÉE)** — Les trois
  per_config du 2.5M wide (electricity/W, m4_yearly/A, us_births/M) confirment le mécanisme :
  `backtest.n_base = 0`, `ratios = {}`, `k_hist = {1: n}` — le sélecteur n'a jamais tourné.
  Nuance : us_births/M est UNE série sur deux fenêtres (ratio local 0.95, officiel 1.22 :
  bruit) ; les témoins sont electricity/W (370 séries) et m4_yearly (22 974). Harnais :
  `+ratein_k_up=2,3,4 +ratein_min_bt=4 +ratein_bt_windows=4` (détail au registre TimeJEPA du
  même jour), inerte sans les flags, tag `-up234-bt4-w4`. **P-SSM.7** (2.5M wide `1.2841`,
  stack flip + mix + pool + ces trois flags, 97 configs) : les 17 configs W/M/A/Q + m4 gagnent
  ≥ 5 % de CRPS ratio en géomoyenne (nous/FlowState ×1.132 → ≤ 1.08) ; stack global ≤ 0.521
  (−0.3 pt) ; les configs à h ≥ 16 inchangées (mêmes k, un candidat k < 1 ne s'impose que s'il
  bat k = 1 de 5 %). ÉCHEC si stack ≥ 0.5242 ou si le groupe basse fréquence ne bouge pas :
  le sélecteur n'est pas ce qui manque aux séries courtes, c'est le modèle (→ B1 avec cibles
  courtes). Coût : le backtest s'allume sur m4_yearly (23 k séries), m4_monthly (48 k) et
  m4_daily avec 14 candidats × 4 fenêtres : compter 2 à 3× le temps d'une éval, à lancer sur
  une carte quand P-SSM.5 en libère une, ou sur le 4090 loué. Commande :
  `STACK="+tta_flip=true +ratein=mix +ratein_pool=true +ratein_k_up=2,3,4 +ratein_min_bt=4
  +ratein_bt_windows=4" ONLY=epoch00_valloss1.2841 scripts/eval_checkpoints_ssm.sh
  checkpoints/timessm_mini_v3_wide_zs/pretrain_False +gift_batch_size=32` (TimeJEPA à jour sur
  le pod), puis `gift_gap_ssm.py` avant/après.

- **2026-09-26 (B3′ PRÉCISÉ : sur les basses fréquences à horizon court, RateIN est ÉTEINT
  par construction)** — Lecture de `all_results.csv` du 10M (`3.0672-v1`, copié localement)
  contre les CSV du leaderboard, 17 configs W/M/A/Q + m4 : géomoyenne nous/FlowState
  ×1.132, nous/Toto ×1.037. Les pires : electricity/W 0.658 contre 0.439 (FlowState),
  solar/W 0.868 contre 0.583, us_births/M **1.219** contre 0.887 (nous sommes PIRES que la
  saisonnalité naïve, MASE 1.261 : la médiane est fausse, pas seulement le fan), m4_hourly
  0.667 contre 0.545, m4_yearly 0.951 contre 0.780. Cause candidate vérifiée dans le code
  du harnais : `_backtest_series_k` force k = 1 quand `h_bt = min(h, avail) < 16`
  (`evaluate_gift.py:311-312, 399-400, 472-473`), et `K_CANDIDATES` ne contient aucun k < 1
  (`ratein.py:15` : « never k<1 »). Donc sur A (h 6), Q (8), W (8, 13), m4_daily (14) et les
  M à h = 12 (us_births, hospital, car_parts, saugeen), le sélecteur de rythme n'a jamais le
  droit d'agir, et les périodes courtes en pas (12 mensuel, 52 hebdo à contexte ~150) ne
  peuvent pas être ramenées dans la bande [16, 48] que le modèle préfère, faute de candidat
  de SUR-échantillonnage. C'est cohérent avec la part d'instances décimées au short : 0.31.
  Hypothèse B3′ (inférence seule, à graver après lecture des `per_config` W/M/A/Q du pod,
  `k_hist` attendu = {1: n}) : (a) autoriser le backtest à h_bt ≥ 4 avec plus de fenêtres,
  (b) ajouter des candidats k ∈ {1/2, 1/3, 1/4} par interpolation linéaire du contexte (le
  fan est ensuite décimé, l'inverse exact de `decimate`/`reinterp_fan`), pour les périodes
  < 16 pas ; le bouton Δ (w > 1) est l'autre chemin. Prédiction à écrire avec les JSON.
  NB : `all_results.csv` laisse domaine et variables vides ; lire ces colonnes dans
  `raw/seasonal_naive.csv`.

- **2026-09-26 (A1, CARTE PAR CONFIG DU SSM : LES DEUX HYPOTHÈSES DE DÉPART TOMBENT — le long
  terme est notre MEILLEUR terme, le court notre pire ; l'écart multivarié est le domaine
  CloudOps de Toto ; sous-couverture partout)** — `gift_gap_ssm.py`, 2.5M wide `1.2841` et
  10M `3.0672-v1`, 97 configs, contre Toto-2.0-4m / FlowState-9.1M / TTM-R3-PT (résultats des
  runs wide sous `evaluation/timessm_mini_v3_zs/`, nom de l'eval config, pas du run).
  **Par terme** (CRPS ratio, 2.5M | 10M ; nous/Toto, victoires) : short 55 : 0.5476 | 0.5512 ;
  ×1.038, 17/55 · medium 21 : 0.5053 | 0.4975 ; ×0.969, 13/21 · long 21 : **0.4851** | 0.4855 ;
  ×0.935, 14/21. Contre FlowState : ×1.056 / ×1.043 / ×1.017. Le rollout autonome à 720-900
  pas depuis un état entraîné à 256 est ce que le modèle fait de MIEUX ; la géométrie 1024/256
  n'est PAS le facteur limitant. La règle « medium/long ≥ 1.05 × short → B1 d'abord » n'est
  pas remplie, c'est l'inverse. **Par fréquence** : pertes contre FlowState sur les basses
  fréquences à horizon court (W ×1.126, M ×1.054, A ×1.136, Q ×1.029 : 15 configs, contextes
  de 13 à 240 pas) et sur 10S (×1.121 ; contre Toto ×1.152) ; victoires sur 10T (×0.986) et 5T
  (×0.986). **Par variables** : 1 var ×0.983 vs Toto, 2-10 var ×1.024 — mais les 10 pires
  configs sont bizitobs (×1.79, ×1.46, ×1.15) et bitbrains_fast_storage (×1.42, ×1.32, ×1.19,
  ×1.14), domaine Web/CloudOps (×1.028 vs Toto), les métriques d'observabilité sur lesquelles
  Toto (Datadog) est entraîné et dont aucun corpus public ne dispose ; l'écart « multivarié »
  est un écart de DOMAINE. **Cousin au corpus** : avec cousin ×0.961 vs Toto (20/33), sans
  ×1.021 (24/64) : les données pèsent plus que l'architecture. **Couverture 80 %** : short
  0.717, medium 0.704, long 0.685 (10M : 0.714 / 0.701 / 0.684) : sous-couvert partout, pire
  aux longs horizons ; part d'instances décimées 0.31 / 0.62 / 0.90 par terme : RateIN
  travaille surtout au long terme. **10M contre 2.5M** : meilleur sur medium, 5T, D, H ;
  moins bon sur short, W, M, A, 10S : le mélange du sampler entier (1 fenêtre par grosse
  famille) a coûté aux petites familles basse fréquence, cohérent avec le diagnostic du 20/09.
  **Verdicts** : (1) 1024/256 n'est pas le levier ; (2) l'univarié non plus, le multivarié
  est confondu avec CloudOps ; (3) le levier est le COURT terme à basse fréquence et la
  calibration. **Ordre révisé** : B2 (γ, zéro entraînement) d'abord ; B1 gardé mais
  REDÉFINI : `horizon_lengths [32, 64, 128, 256, 384, 512]` — les cibles courtes repondèrent
  la pinball vers le futur proche (à 256 pas, 3/4 du poids est au-delà du pas 64) et la
  randomisation calibre (h512 : 0.800). P-SSM.6 réécrite : couverture ≥ 0.78, stack ≤ 0.521,
  le gain venant du short (0.5476 → ≤ 0.540) sans dégrader medium/long ; échec si ≥ 0.5245.
  Nouveau bras candidat B3′ (basses fréquences à contexte court, 15 configs) : à lire dans
  les per_config W/M/A/Q avant de le graver (S4-a′ du PLAN, contextes variables courts). B4
  (multivarié) reste hors budget : le domaine manque, pas l'architecture. Oracle-k (A2)
  reste à mesurer.

- **2026-09-26 (DIAGNOSTIC DE LA SOUS-PERFORMANCE : relecture des trois registres, plan
  approuvé ; outillage livré pour la phase A et le bras B1 ; P-SSM.6 GRAVÉE)** — Question de
  l'utilisateur : les facteurs limitants restants sont-ils la géométrie 1024/256 et
  l'univarié ? Ce que les registres disent déjà : (1) le « gap de distribution » de h512
  (TimeJEPA, 2026-08-31) était une AMPUTATION DE CORPUS (fenêtre 1536 → lotsa_short 1280 et
  bloc décimé 1024/682 hors finetune) : configs saines 0.995, amputées 1.111, levier horizon
  « réel mais petit » ; et h512 a donné une couverture 0.800 EXACTE (randomisation d'horizon
  [64..512]) ; recommandation gravée alors, jamais courue : « horizon randomisé large SANS
  étendre la fenêtre ». (2) 42/97 configs ont un horizon > 256 (medium 480-600, long
  720-900) ; le SSM n'a jamais vu de cible au-delà de 256. (3) Sur TimeJEPA l'écart à Toto
  est PLAT par terme (E17 ×1.28/1.32/1.29 ; E19 0.611/0.619/0.616) et DIFFUS (corps de 81
  configs ~0.54 contre ~0.47 chez eux) ; pertes systématiques contre FlowState sur W et M à
  horizon court, et bizitobs/10S (domaine sans corpus public) ; AUCUNE carte par config
  n'existe pour le SSM (résultats sur le pod). (4) L'univarié n'est pas un handicap de
  protocole (le harnais explose les jeux multivariés variable par variable comme
  l'officiel ; FlowState est univarié) ; l'écart plus grand sur les jeux multivariés (E17
  ×1.38 contre ×1.22) est confondu avec l'absence de leur domaine des corpus. (5) Fan trop
  étroit (couv. 0.70) ; γ neutre sur TimeJEPA en distribution, jamais essayé sur le SSM.
  (6) Plus grande marge connue : oracle-k de RateIN (0.5190 contre 0.5340 sur head8, 1.5 pt),
  jamais mesuré sur le SSM. Lecture des objectifs : 0.52 pour le 2.5M plausible (B1 + B2) ;
  0.50 pour le 10M demande 2.5 pt que rien ne soutient, 0.51 meilleur scénario réaliste.
  **Plan** (`~/.claude/plans/playful-pondering-dragonfly.md`) : phase A = carte gift_gap du
  SSM par terme / horizon / fréquence / variables / cousin au corpus
  (`scripts/gift_gap_ssm.py`, testé sur fixture) + oracle-k sur le 2.5M wide (`ORACLE-k`
  ajouté au résumé d'éval) ; règles de décision gravées ; phase B = B1 horizon aléatoire
  DANS la fenêtre fixe, B2 température de quantiles (calibrate_quantiles.py accepte
  `--config-dir/--horizon/--set`, enrobage `scripts/calibrate_ssm.sh`), B3/B4 conditionnés
  à la carte. **B1 livré** : `SSMFinetuneModule._maybe_resplit_horizon` (re-découpage de la
  même fenêtre 1280 en [1280−h | h], h ∈ {64..512} p 0.5, masque propagé, refus si un pas
  rembourré entrerait dans le contexte), `_forward_and_loss` réimplémentée avec `n` (le
  parent appelait forecast sans n : mismatch de forme à h ≠ 256), témoins
  `geometry/horizon_len` et `aug/horizon_neq_native_frac`, validation à 256 ; 6 tests
  (conservation de la fenêtre, identité au parent à p = 0, backward jusqu'au token futur,
  masque, crop après re-découpage, deux tirages actifs) ; config `ssm_mini_v3_hrand`
  (continuation du champion wide `1.2841`, LR 1e-4, 89 M fenêtres, sampler fractionnaire
  batch 128 plafonné). **P-SSM.6** : dernier checkpoint, couverture 80 % ≥ 0.78 et stack
  ≤ 0.521 ; medium/long baissent au moins autant que short ; ÉCHEC si stack ≥ 0.5245.
  Ordre : A1 (une soirée, CPU, après rsync des `per_config` du pod) → A2 → B2 → B1 (12 h
  de GPU) ; report sur le 10M après P-SSM.5.

- **2026-09-26 (SAMPLER FRACTIONNAIRE MESURÉ SUR LE VRAI CORPUS ; bras `ssm_mid_v3_frac` lancé)**
  — `audit_batch_sizes.py`, 250 k batches, batch 48, plafond 48 : réalisé médian 48, moyenne
  46.5, p1 32 (début d'époque, parts fractionnaires des petites familles pas encore à 1),
  max 48 partout, rien au 111 111e batch. Époque à cette composition 2.32 B fenêtres ;
  100 M fenêtres = fraction 0.04316 = 717 k micro-batches par carte = 89.6 k pas
  d'optimiseur ; warmup 0.004316. Lancement : poids de `3.0672-v1` (100 % du 10M), LR 1e-4,
  seed 420, ~2 j 18 h. P-SSM.5 gravée le 25/09 (≤ 0.520 ; échec ≥ 0.524).

- **2026-09-26 (10M, TABLE COMPLÈTE À 97 SUR 20 CHECKPOINTS : dernier checkpoint 0.7722 /
  0.5245 / couv. 0.705, bande des cinq derniers 0.5235-0.5259 — P-SSM.4 NON ATTEINTE (≤ 0.515),
  hors zone d'échec (≥ 0.525) de justesse : ÉGALITÉ AVEC LE 2.5M WIDE)** — Stack flip + mix +
  pool, ordre de création, `*` = lancement en warmup hors courbe :
  5 %* 0.8139 / 0.5544 / 0.779 · 10 % 0.7716 / 0.5312 / 0.738 · 15 % 0.7850 / 0.5329 / 0.742 ·
  20 % 0.7700 / **0.5235** / 0.707 · 25 % 0.7701 / 0.5239 / 0.730 · 30 % 0.7891 / 0.5312 / 0.705 ·
  35 % 0.7776 / 0.5252 / 0.735 · 40 % 0.7790 / 0.5239 / 0.713 · [reprise r2] 45 % 0.7783 /
  0.5278 / 0.693 · 50 % 0.7798 / 0.5275 / 0.708 · 55 % 0.7775 / 0.5286 / 0.719 · 60 % 0.7723 /
  0.5276 / 0.719 · 65 % 0.7744 / 0.5282 / 0.706 · 70 % 0.7728 / 0.5248 / 0.703 · 75 % 0.7715 /
  0.5246 / 0.697 · 80 % 0.7732 / 0.5259 / 0.697 · 85 % 0.7708 / **0.5235** / 0.709 · 90 % 0.7731 /
  0.5244 / 0.704 · 95 % 0.7717 / 0.5243 / 0.703 · 100 % 0.7722 / 0.5245 / 0.705.
  **Verdict** : le dernier checkpoint annealé fait 0.5245, dans la bande du champion 2.5M wide
  (0.524-0.526), avec un MASE moins bon de 0.4 pt (0.772 contre 0.767-0.769) et la même
  couverture. Quadrupler la capacité, sur ce run, rapporte 0 pt de CRPS annealé contre
  annealé, et 0.6 pt plateau contre plateau à LR haut (0.5237 contre 0.530). L'anneal final
  a valu 0.3-0.4 pt (65 % 0.5282 → 85-100 % 0.5235-0.5245), comme sur le 2.5M. Bande des cinq
  derniers checkpoints 0.24 pt, plus large que celle du 2.5M (0.18) ; le point à publier
  pour ce run, si on le publie, est la bande, et le champion par le chiffre est le 85 %
  (0.7708 / 0.5235, meilleur MASE du run). Ce que le run NE dit PAS : « la capacité n'est
  pas le levier ». Il porte deux confusions mesurées (mélange du sampler entier ; reprise
  à 40 % qui a coûté 0.4 pt pendant ~20 % du run) et moitié moins de pas d'optimiseur que
  le 2.5M (259 k contre 505 k, bras wide compris). À pas égal (259 k), le 10M annealé
  (0.5245) est 0.5 pt devant le 2.5M au même pas (0.530, LR haut). Lecture utilisateur,
  retenue : ou bien la capacité est sous-exploitée (pas assez de pas), ou bien le protocole
  (mélange) l'empêche de servir — le bras fractionnaire `ssm_mid_v3_frac` (P-SSM.5, ≤ 0.520)
  teste les deux à la fois, depuis le checkpoint 100 % (`3.0672-v1`), et un second bras
  fractionnaire ne se lance que si le stack descend encore entre ses checkpoints 80 et 100 %.
  Position pour la publication : si P-SSM.5 échoue, le modèle publié est le 2.5M à config
  unique (égalité Toto-2.0-4m, 3 GPU) et le 10M devient une ligne d'ablation ; RateIN sur
  les modèles externes reste le résultat principal.

- **2026-09-25 (10M, TABLE PARTIELLE 10 À 70 % : PLATEAU DANS LA BANDE DU 2.5M WIDE, LA
  REPRISE À 40 % A COÛTÉ 0.4 PT ; P-SSM.4 hors d'atteinte ; BRAS DE CONTINUATION SUR LE
  SAMPLER FRACTIONNAIRE, P-SSM.5 gravée)** — Stack flip + mix + pool, 97 configs, par pas
  d'optimiseur (validation tous les 12 933 pas ; le checkpoint 3.1195 du lancement en warmup
  est hors courbe : 0.8139 / 0.5544) : 10 % 0.7716 / 0.5312 · 15 % 0.7850 / 0.5329 · 20 %
  **0.7700 / 0.5235** · 25 % 0.7701 / 0.5239 · 30 % 0.7891 / 0.5312 · 35 % 0.7776 / 0.5252 ·
  40 % 0.7790 / 0.5239 · [reprise r2, seed 430] · 45 % 0.7783 / 0.5278 · 50 % 0.7798 / 0.5275 ·
  55 % 0.7775 / 0.5286 · 60 % 0.7723 / 0.5276 · 70 % 0.7728 / 0.5248 ; 65 %, 75 % et 80-100 %
  en cours. Lecture : plateau à LR haut dès 20 % (60 M fenêtres, contre 220 M pour le 2.5M),
  0.6 pt sous le plateau du 2.5M classique (0.530) mais DANS la bande du 2.5M wide annealé
  (0.524-0.526), MASE moins bon (0.770-0.779 contre 0.767-0.769). Les quatre checkpoints qui
  suivent la reprise à 40 % sont à 0.5275-0.5286, +0.4 pt, puis le 70 % revient à 0.5248 :
  la reprise déforme le mélange en début de segment (sampler non reprenable, petites
  familles absentes tant que leur allocation fractionnaire n'a pas atteint 1) et le modèle
  met ~20 % du run à s'en remettre. Mesure, pas excuse : les reprises ne sont pas gratuites,
  un sampler reprenable est à écrire avant le prochain long run. Le 30 % (0.5312, MASE
  0.789) dépasse aussi le bruit checkpoint à checkpoint du 2.5M (±0.2 pt) : la bande du
  10M est plus large, à publier telle quelle. Verdict anticipé : P-SSM.4 (≤ 0.515) ne sera
  pas atteinte ; le seuil d'échec 0.525 se joue sur les cinq derniers checkpoints (anneal
  final). Ce run porte deux confusions (mélange du sampler entier, reprise) : il ne tranche
  pas « la capacité n'est pas le levier ». **Décision utilisateur** : ce 10M est un modèle
  d'ingénierie (leaderboard, model card), pas un papier ; on ne relance pas de zéro, on
  reprend ses poids sur le sampler fractionnaire avec un cosinus court, la recette qui a
  fait le champion 2.5M (bras wide). Config `ssm_mid_v3_frac` : poids du dernier checkpoint
  (`+training.pretrained_encoder_path`), `fractional_batch` + `max_batch_size` 48, LR 1e-4,
  warmup 10 % du run, 100 M fenêtres réalisées (~2.5 j), validation tous les 20 % ; fraction
  et warmup passés en ligne de commande depuis l'audit (`--windows 100e6`). **P-SSM.5** :
  stack au dernier checkpoint ≤ 0.520 ; ÉCHEC si ≥ 0.524 (le mélange n'était pas non plus la
  pièce manquante). Deux variables changent (mélange, second cosinus) : consigné, ce bras
  ne rentre pas dans une courbe de scaling.

- **2026-09-20 (10M CHECKPOINT 8, 40 % DU RUN, MI-COSINUS, MÉLANGE DÉGRADÉ : stack 0.7790 /
  0.5239 / couv. 0.713 sur 97 — prédiction 0.530 ± 0.005 BATTUE, sous le seuil 0.525 : ON
  LAISSE FINIR)** — Stack flip + mix + pool, `ssm_mid_v3_eval`, gift_batch_size 48 : MASE
  0.7790, CRPS 0.5239, q10 0.141 / q90 0.854, 51/97 configs majoritairement décimées
  (52.6 % d'instances à k > 1, en hausse régulière : 46 → 50 → 53 % du 2.5M au 10M).
  Lecture : à pas égal (103 k), 1.1 pt de CRPS sous le 2.5M classique (~0.535 interpolé) ;
  déjà dans la bande du champion 2.5M wide (0.524-0.526) et au niveau de Toto-2.0-4m
  (0.5242), avec 60 % du run et tout l'anneal devant lui, et sur le mélange « 1 par grosse
  famille » du sampler entier. La capacité paie donc malgré le mélange. Réserve : le MASE
  (0.7790) est MOINS bon que le 2.5M wide (0.767-0.769) alors que le CRPS est meilleur — le
  10M gagne sur le fan, pas sur la médiane, à surveiller au dernier checkpoint (si l'écart
  MASE persiste, c'est un fait à publier, pas à lisser). Décision : pas de relance sur le
  sampler fractionnaire pour ce run ; reprise depuis le checkpoint 8, seed 430, même
  fraction. Estimation révisée pour la fin du run (anneal complet à 1e-6) : stack
  0.512-0.518 ; P-SSM.4 (≤ 0.515) reste ouverte, plausible. Le sampler fractionnaire
  devient le défaut du PROCHAIN run (100M ou reprise 10M), pas de celui-ci.

- **2026-09-20 (10M à 40 % du run, sampler entier conservé — l'utilisateur garde le run en
  cours ; ÉVAL INTERMÉDIAIRE GIFT du checkpoint 8 `3.0745` (pas 103 471), prédiction gravée)**
  — Checkpoints val_loss 3.1030 → 3.0745 de 15 à 40 % du run, monotone. Décision utilisateur :
  ne pas relancer sur le sampler fractionnaire pour l'instant ; le run continue sur le
  mélange « 1 par grosse famille, le reste par quotas » décrit le 2026-09-20, donc l'écart au
  2.5M mêle capacité, schedule et mélange. Éval : stack flip + mix + pool, 97 configs,
  `ssm_mid_v3_eval`, entraînement coupé le temps de l'éval (~45 min sur un GPU) puis repris
  du checkpoint 8. Repère 2.5M classique au même pas (103 k) : entre 10 % (77.6 k, 0.5419)
  et 15 % (116 k, 0.5305) de son run, ~0.535 interpolé, LR encore en montée chez lui, à 0.6
  du pic (cosinus) chez le 10M. **Prédiction** : stack 0.530 ± 0.005. Sous 0.525 : la
  capacité paie déjà malgré le mélange dégradé, on laisse finir. Au-dessus de 0.540 : le
  mélange coûte plus que la capacité ne rapporte, relance sur le sampler fractionnaire
  sans attendre la fin. `ONLY=<stem>` ajouté à eval_checkpoints_ssm.sh pour évaluer un
  seul checkpoint.

- **2026-09-20 (LES MÉTRIQUES DE VALIDATION DU 10M NE SONT PAS COMPARABLES À CELLES DU 2.5M :
  jeu de validation différent par construction ; et le mélange d'entraînement du 10M n'était
  pas celui du 2.5M — sampler fractionnaire livré, RELANCE RECOMMANDÉE)** — Courbes W&B à pas
  égal : val_loss 3.05 contre 1.3, val_mse 9000 contre 3600, val_mae 3.2 contre 1.4, val_wql
  0.348 contre 0.29-0.325. Un facteur 2 sur la MSE dénormalisée ne vient pas du modèle (le
  val_wql descend normalement : 0.399 → 0.377 → 0.348). Cause, lue dans le sampler de
  validation (T = 1, sans suréchantillonnage, non rationné, 300 batches) : l'allocation
  entière floor(p_i × batch) puis plancher 1 donne, à batch 48 < 106 familles, exactement 1
  fenêtre par famille et par batch — validation UNIFORME sur les familles (dominée par les
  petites familles courtes, synthétiques, à forte échelle) — là où le 2.5M à batch 128
  validait sur un mélange à peu près proportionnel (dominé par electricity, traffic…).
  Deux jeux différents, aucune inquiétude à tirer de ces courbes ; seul GIFT compare. Même
  mécanisme côté train, plus grave : à batch nominal 106 toutes les familles ont n_i = 1,
  la température 0.5 disparaît, les grosses familles sont plafonnées à 1 par batch et le
  reste suit les quotas (proportionnel) — le 10M ne s'entraîne PAS sur le mélange du 2.5M.
  Troisième variable entre les deux runs (capacité, schedule, mélange). **Correctif**
  (TimeJEPA `96d0bf2`, opt-in, bit-identique sans l'option) : `fractional_batch` — part
  p_i × batch conservée en flottant et réalisée par accumulation (arriéré borné à une part
  + 1), plan d'époque sur la part flottante. Tests : mélange conforme à la température à
  batch 48 comme à 128 (train, T 0.5, rationné), validation proportionnelle et
  indépendante du batch (T 1), plafond à batch_size respecté. Config `ssm_mid_v3` :
  `fractional_batch: true`, `max_batch_size: 48` (le pic mémoire est celui de 48 fenêtres),
  `schedule_fraction` laissé MANQUANT exprès (train_ssm refuse de lancer) : la longueur
  d'époque dépend du batch réalisé, `scripts/audit_batch_sizes.py` l'imprime pour 298 M
  fenêtres. **Décision proposée** : relancer de zéro avec le sampler corrigé (le run
  courant, ~15 % fait, s'entraîne sur un autre mélange que le 2.5M et plante toutes les
  10 h) ; coût ~1 jour, gain : un point de scaling lisible et un run qui va au bout.
  Chiffres du registre à corriger quand l'audit aura tourné : « batch effectif 1152 »,
  « 298 M fenêtres = 0.06667 », et les budgets en fenêtres du 2.5M (batch nominal 128,
  réalisé inférieur ; sur le corpus factice 59).

- **2026-09-18 (CAUSE DES PLANTAGES TROUVÉE : le batch réalisé n'est pas `data.batch_size` ;
  batch 48 et batch 64 étaient LE MÊME RUN ; pics déterministes du sampler rationné)** —
  Quatrième mort du 10M, et le motif : les trois derniers processus meurent à leur 111 111e
  batch exactement (111111/1552000 à batch 64 seed 420 ; 111111/2069436 à batch 48 seed 420 ;
  214582 − 103471 = 111111 à batch 48 seed 421), à 10 h 04 chaque fois, OOM simultané sur
  les trois GPU à 22.83 Gio. Ni fuite (la marge supposée différait), ni contenu (seed
  changé). Mécanisme, lu dans `TemperatureSampler` : 106 familles, `samples_per_dataset`
  plancher à 1 par famille, donc dès que `batch_size` < 106 le batch nominal vaut 106 quel
  que soit `batch_size`, et la boucle de retrait ne peut pas descendre sous 1. La taille
  RÉALISÉE vient alors des seuls quotas fractionnaires du rationnement (max_samples /
  num_batches, accumulés), qui tirent ensemble à des indices de batch déterministes,
  indépendants du seed et de `batch_size`. Vérifié sur corpus factice (106 familles) :
  batch_size 48 et 64 donnent des itérations identiques batch pour batch, moyenne réalisée
  22, pic à 49 au même indice ; à batch_size 128 la moyenne réalisée est 59, pas 128.
  **Erreurs de ma part à corriger** : « batch 48 laisse 6 Gio de marge » était faux (même
  sampler, même mémoire) ; le « batch effectif 1152 » et les budgets en fenêtres du
  registre (298 M, et ceux du 2.5M à batch 128) sont calculés sur le batch NOMINAL et sont
  donc faux ; les vrais chiffres attendent l'audit sur le corpus réel
  (`scripts/audit_batch_sizes.py`, rejoue l'arithmétique du sampler sans lire de données,
  validé batch pour batch contre le vrai itérateur). Seule différence réelle entre les
  lancements : l'accumulation (6 puis 8), donc le batch effectif a changé de 33 % entre
  eux. **Correctif** (TimeJEPA, opt-in, itération bit-identique sans l'option) :
  `max_batch_size` dans le sampler — au-delà du plafond les familles au plus petit arriéré
  sont REPORTÉES au batch suivant, allocation conservée : même exposition, mémoire bornée ;
  tests (identité, borne, aucun affamement par famille, arriéré borné). `data.max_batch_size`
  branché dans train_ssm.py ; la VALEUR n'est pas fixée à l'aveugle : au-dessus de la
  moyenne réalisée (sinon l'arriéré diverge), sous ce que la carte encaisse (64 fenêtres à
  contexte 1024 = 19.6 Gio au profil). En attendant, la boucle de relance + autosave
  horaire borne la perte à 1 h sur 10.

- **2026-09-17 (troisième plantage du 10M à 07:36, 41 min après le checkpoint 5 % `3.1195`
  (val_wql 0.399) ; cause non vue (trace tronquée) ; reprise r1 validée par la continuité du
  LR ; LE CHAMPION 2.5M A ÉTÉ PRIS PENDANT SON WARMUP)** — Reprise depuis 3.1195 avec
  `data.seed=421` : dans W&B, `lr-AdamW` du run r1 repart au pas 12 933 exactement au niveau
  de la rampe interrompue et continue jusqu'au pic 3e-4 au pas 25 900. Le point d'inflexion
  de `train_smape` vers 25 k est le pic du LR, pas la reprise. Lecture des courbes à pas
  égal : le 10M descend plus vite et plus bas parce que son warmup dure 26 k pas contre
  258 k pour le 2.5M (au pas 30 k : LR 3e-4 contre 3.5e-5), le schedule domine, pas la
  capacité ; sa bande de sMAPE plus étroite (moins de batches catastrophiques) est le seul
  indice qui puisse relever de la capacité. **Constat rétroactif** : `warmup_epochs` 0.1 sur
  un run de 0.3 époque = 33 % du run en warmup ; le champion 2.5M classique (25 % du run,
  pas 194 k) a été pris LR en montée à 2.3e-4, et le « plateau » 0.528-0.533 de 25 à 55 %
  couvre le pic et le début du cosinus, jamais un régime annealé. Le seul anneal du 2.5M
  est le bras wide (1e-4 → 1e-6 sur 77.6 k pas), là où il a gagné ses 0.25 pt de stack
  (0.5282 → 0.5257). Le 10M aura un cosinus complet jusqu'à 1e-6 : deuxième variable entre
  2.5M et 10M (avec la recette wide dès le départ), à tenir dans la lecture de P-SSM.4.
  Désaccord de métriques sur le checkpoint 5 % : val_loss 3.12 (lancement en warmup au même
  pas : 2.46) mais val_wql 0.399 (contre 0.428) ; le WQL est la grandeur proche du CRPS de
  GIFT, le harnais tranche. Outillage livré après le troisième plantage : autosave horaire
  `last-autosave.ckpt`, `scripts/train_ssm_loop.sh` (reprise automatique, seed incrémentée,
  journal des tentatives), étape 5 du preflight qui exerce une vraie reprise. Estimation
  gravée avant la fin du run : stack au dernier checkpoint 0.512, bande 0.505-0.520, nu
  0.52-0.53 ; FlowState 9M (0.4866 nu, fréquence en entrée) hors de portée du 10M.

- **2026-09-16 (LE PREMIER LANCEMENT 10M ÉTAIT ENTIÈREMENT EN WARMUP : `warmup_epochs` 0.1
  hérité du 2.5M sur un run de 0.06667 époque — relance DE ZÉRO, pas de reprise)** — En
  cherchant le repère commun aux deux runs, relecture du scheduler : `warmup_steps =
  warmup_epochs × steps_per_epoch` et `total_steps = schedule_fraction × steps_per_epoch`,
  les deux en unités d'époque. À 0.1 sur 0.06667, le warmup (388 k pas) dépasse le run
  (258.7 k) : LR à 4 % de son pic au checkpoint 5 % (1.3e-5), cosinus jamais commencé, T_max
  négatif. C'est ce que montrait la courbe de train « qui descend plus lentement ». Le
  checkpoint 2.4551 n'est donc pas repris (son état de scheduler restaurerait le mauvais
  warmup, et le corriger à la main dans le state_dict est plus risqué que 7 h de GPU).
  Au passage, fait consigné : le 2.5M classique a eu 33 % de son run en warmup (0.1 époque
  sur 0.3), le wide 10 % ; le 10M prend 10 % (0.006667, 25.9 k pas), convention du wide.
  Garde : `check_schedule` dans train_ssm.py refuse un warmup ≥ 50 % du run, test sur les
  trois configs livrées. Deux trous du chemin de reprise bouchés au passage (weights_only,
  buffers RevIN [B, 1, 1]) ; la reprise elle-même reste non exercée sur un run réel.
  **Repère commun entre runs** : même batch effectif 1152 → l'axe est le pas d'optimiseur
  (`trainer/global_step` dans W&B), 1 pas = 1152 fenêtres, identique dans les deux runs ;
  la courbe comparable est `val_wql` (pas de tirage de Δ en éval), PAS le train sMAPE (le
  10M tire 13 facteurs à p 0.7, le 2.5M classique 5 à p 0.5 : mélange plus dur). Ancrages
  (pas d'optimiseur → fenêtres) : 2.5M classique checkpoint 5 % = 38.8 k pas = 45 M
  (stack 0.5761), 10 % = 77.6 k (0.5419), 15 % = 116 k (0.5305), 20 % = 155 k (0.5334),
  25 % = 194 k = 223 M (champion 0.5282), 30 % = 233 k (0.5301) ; 10M checkpoints tous les
  12.9 k pas, donc le checkpoint 3k du 10M tombe sur le checkpoint k du 2.5M, et sa fin
  (258.7 k pas, 298 M) sur le 2.5M à 33 % de son run, en plein plateau 0.528-0.533. À LR
  égal ce n'est pas garanti (cosinus de 776 k pas contre 258.7 k), donc le verdict reste
  P-SSM.4 au dernier checkpoint, les ancrages servent à lire la courbe en cours.

- **2026-09-15 (RUN 10M MORT UNE DEUXIÈME FOIS À 10 h, pas 111k / 1 552 000 : vrai OOM cette
  fois, 22.83 Gio alloués, 133 Mio réservés inutilisés — reprise du checkpoint 5 % à batch 48 ×
  acc 8)** — Les segments extensibles ont tenu leur rôle (plus de fragmentation), le pic
  d'activations dépasse simplement la carte : 22.8 Gio en entraînement réel contre 19.6 au
  profil à contexte 1024 fixe. Les 3 Gio d'écart ne sont pas attribués avec certitude
  (espaces de travail cuFFT des transformées batchées de tailles variables, buckets DDP,
  résidus de la boucle de validation) ; le profil mesure le modèle nu hors module et hors
  DDP, il sous-estime le pic réel, à noter pour le 100M. Décision : batch 48 × accumulation
  8 × 3 GPU = 1152 (inchangé), débit identique (129 contre 134 fenêtres/s, régime
  mémoire-bound), ~6 Gio de marge au lieu de 4. Fraction recalculée depuis les fenêtres :
  l'époque du sampler reste 31.04 M batches (la plus grande famille garde 1 fenêtre par
  batch), donc 298 M fenêtres = 2.07 M batches = 0.06667 ; l'optimiseur voit les mêmes
  258.7 k pas (2.07 M / 8 = 1.55 M / 6), le cosinus est le même. Reprise depuis
  `epoch00_valloss2.4551.ckpt` (pas 12 933 = 5 % du run, val_loss 2.4551, val_wql 0.428,
  écrit à 7 h de run) : Lightning restaure la boucle, l'optimiseur et le scheduler ; les
  5 premiers % ont vu des batches de 64 × 6, la suite 48 × 8, même batch effectif et même
  nombre de pas — équivalent au niveau de l'optimiseur, le mélange de familles diffère
  marginalement par l'arrondi de floor(p × batch). Consigné, pas une variable. Coût des
  deux plantages : ~15 h de GPU. Premier signal du 10M : à 5 % du run, val_wql 0.428 ; le
  2.5M classique à 5 % de SON run (1.5 % d'époque, 45 M fenêtres) était à 0.5761 en stack —
  pas comparable directement (val interne vs GIFT), le premier checkpoint s'évalue au
  harnais quand un GPU se libère, pas pendant le run.

- **2026-09-15 (RUN 10M MORT À 4 h 45 : OOM de fragmentation au pas 52.7k ; et la fraction
  0.1 valait 596 M fenêtres, pas 298 M — relance à 0.05 avec allocateur extensible)** —
  Trace : `torch.fft.irfft` demande 466 Mio, 18.16 Gio alloués + **4.57 Gio réservés mais
  inutilisables** : fragmentation du caching allocator par les formes variables (contextes
  aléatoires 128..1024, tailles de FFT par item), pas un manque de mémoire (le profil à
  1024 fixe pointe à 19.6 Gio). Correctif : `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  posé dans `train_ssm.py` avant l'import de torch (hérité par les rangs DDP). Aucun
  checkpoint écrit (val tous les 5 % du run, mort à 1.7 %) : relance de zéro, 4 h 45
  perdues. Second constat, sur la barre de progression (52704 / 3 104 000 batches) : l'époque
  du sampler DÉPEND DU BATCH — la plus grande famille reçoit floor(p × batch) ≥ 1 fenêtres
  par batch, 2 à batch 128, 1 à batch 64, donc l'« époque » à batch 64 fait 5.96 B fenêtres
  (31.04 M batches × 64 × 3) contre 2.98 B pour le 2.5M ; `schedule_fraction` 0.1 aurait
  coûté 596 M fenêtres et 11.4 jours. Le budget se compte en FENÊTRES : 0.05 à batch 64 =
  1.55 M batches × 64 × 3 = 298 M, ce qui était prévu ; P-SSM.4 inchangée. Vitesse mesurée
  sous DDP 3.14 it/s par GPU = 603 fenêtres/s (contextes aléatoires plus courts que le profil
  à 1024, 134/s) → ~5.7 jours, ~95 € au tarif du pod. Note pour les overrides 8 GPU de
  l'en-tête : la fraction est à recalculer depuis les fenêtres, jamais recopiée. Commande :
  `python scripts/train_ssm.py --config-name ssm_mid_v3 wandb.run_name=ssm-mid-v3`.

- **2026-09-14 (TABLE WIDE COMPLÈTE, 5 checkpoints à 97 : bande 0.524-0.526, égalité avec
  Toto-2.0-4m ; run 10M LANCÉ sur les 3× 3090)** — Stack flip + mix + pool, bras wide (20 %
  à 100 % de son budget de 3 % d'époque) : 1.2836 0.7670 / 0.5257 / couv. 0.717 · 1.2845
  0.7677 / 0.5260 / 0.705 · 1.2841 0.7679 / **0.5242** / 0.707 · 1.2817 0.7694 / 0.5255 /
  0.709 · 1.2822 0.7686 / 0.5252 / 0.710. Écart max 0.18 pt de CRPS et 0.24 pt de MASE :
  le corps est convergé pour ce bras, et la val loss ne discrimine plus (la plus basse,
  1.2817, n'est pas le meilleur stack). Choisir "le bon checkpoint" dans cette bande
  serait sélectionner sur le test : on PUBLIE LA BANDE (0.525 ± 0.002), le 1.2841 n'est
  champion que par le chiffre. Toto-2.0-4m est à 0.5242 sur le leaderboard : égalité à la
  quatrième décimale, avec un modèle de 2.5M et le stack d'inférence. Position tenue sur
  la comparaison : le stack fait partie du modèle au même titre qu'un embedding de
  fréquence fait partie de FlowState ; zero-shot, mêmes entrées (on n'utilise même pas la
  fréquence), le calcul est déplacé de l'entraînement vers l'inférence. Que Toto-2.0
  encode la fréquence n'est PAS vérifié : à sourcer avant de l'écrire. L'argument devient
  imparable seulement si les modèles externes gagnent aussi avec le stack (table centrale
  du papier RateIN, en attente de GPU). Le MASE ne bouge pas d'un checkpoint à l'autre :
  le gain CRPS du wide vient de la calibration (q10 0.143-0.153, intervalle 80 % à
  0.705-0.717 contre 0.805 pour le run classique), le fan s'est resserré et le modèle est
  sous-couvert. Levier séparé, non exploré : température des quantiles à l'inférence, à
  graver comme bras. **Vitesse** : le train est bien parallèle (convolution FFT, jamais la
  récurrence) ; le ×3 par fenêtre contre TimeJEPA mini tient aux 1280 tokens par fenêtre
  (10× le patching), à la largeur double du bloc gated et à la FFT float32 ; régime
  limité par la bande passante mémoire, pas par le calcul. Marge restante : `torch.compile`
  sur le bloc (1.3-1.5× attendu, à profiler avec `scripts/profile_step.py` sur un GPU
  libre après le run), noyau FFT fusionné (FlashFFTConv, ce qui donne le 2× de S4 /
  FlowState). Aucun des deux dans le run 10M : une variable. **Run 10M lancé** (`ssm_mid_v3`,
  wide dès le départ, 13 facteurs p 0.7, pas de reprise de poids ; batch 64 × acc 6 × 3
  GPU = 1152, num_workers 8, `schedule_fraction` 0.1). Preflight passé (35 tests, audit corpus OK, smoke DDP 3 GPU). Profil mesuré : batch 48 129 fenêtres/s et 15.2 GiB, batch 64 134/s et 19.6 GiB, batch 80 et 96 OOM, batch 128 avec activation checkpointing 107/s et 9.6 GiB (le recalcul coûte plus que le batch ne rapporte). Le débit par fenêtre est plat en batch : régime mémoire-bound confirmé. 402 fenêtres/s sur 3 GPU → ~9 jours pour 298M fenêtres, pas 6 : le 10M coûte 5× le 2.5M par fenêtre, pas 3. ~150 € au tarif du pod
  contre ~190 $ + migration sur 8× 5090 qui coûte ~2× par fenêtre). Décision : pas de
  migration avant un 100M, qui ne se fait que si le 10M tient P-SSM.4 (courbe de scaling
  à montrer, sinon point isolé), et sur TimeSSM, pas sur TimeJEPA. Réserve de lecture :
  2.5M contre 10M n'est pas à une variable (le 2.5M a vu 5 facteurs sur 7.5 % puis 13 sur
  3 % ; le 10M voit 13 facteurs sur 10 %) ; en cas d'échec, un bras 2.5M wide-from-scratch
  départagerait capacité et recette, non planifié. Borne 0.1 et pas 0.3 : budget (6 j
  contre 18), cosinus annealé sur la fraction (un run complet, lisible au dernier
  checkpoint, comparable au 2.5M annealé ~10.5 % d'époque au total), et porte de sortie
  par second anneal court ou `schedule_fraction=0.2` si le stack descend encore entre les
  deux derniers checkpoints. Évals externes (Chronos-Bolt, TTM-R3, t0-alpha) à faire sur
  un 4090 community pendant le run, elles ne dépendent que des données GIFT.

- **2026-09-14 (P-SSM.2c ÉCHOUE ; le run wide donne pourtant un NOUVEAU CHAMPION par le stack :
  `wide 1.2836` 0.7670 / 0.5257 / couv. 0.717 contre 1.2942 0.7717 / 0.5282 / 0.805)** — Sur
  les quatre premiers checkpoints wide : delta sans garde 0.822-0.827 / 0.558-0.560 (bouton
  activé sur 37-39 configs), backtest dur 0.7821 / 0.5361 (1.2836) et 0.7872 / 0.5420
  (1.2845). Delta perd 2.2 pt contre la décimation sur les mêmes poids : entraîner le bouton
  sur toute la plage [1/48, 4] n'a rien changé au bouton. Mécanisme retenu : la décimation ne
  change pas seulement l'échelle de temps, elle raccourcit contexte et horizon EN PAS pour la
  tête (cross-attention sur moins de tokens, fan plus court), ce que le bouton ne fait pas par
  construction — l'invariance exacte au rythme n'est pas ce que RateIN exploite. Résultat
  négatif clos, publiable. En revanche le multi-rythme large a amélioré le CORPS : backtest
  dur 0.5445 → 0.5361 (−0.8 pt) et stack 0.5282 → 0.5257, MASE 0.7717 → 0.7670 ; couverture
  0.805 → 0.717 (le fan se resserre, à déclarer). Champion leaderboard : `wide 1.2836`
  (`timessm_mini_v3_wide_zs`, checkpoint 1 du bras wide, 20 % de son budget) ; le 25 % du run
  classique reste le meilleur en couverture. Cinquième checkpoint wide en cours d'éval.
  **Scaling** : config `ssm_mid_v3` (d_model 384, 8 blocs, 10.1M), DDP (chemin validé ; FSDP
  jamais couru en multi-GPU, smoke seulement), batch 48 × acc 3 × 8 GPU, `schedule_fraction`
  0.1 (298M fenêtres, ~20 h à 8× 5090, ~160 $), multi-rythme large conservé (il a aidé le
  corps). **P-SSM.4** : stack au dernier checkpoint ≤ 0.515 ; ÉCHEC si ≥ 0.525 (la capacité
  n'est pas le levier à ce budget de données). `scripts/preflight.sh` : tests, audit corpus,
  profil d'un pas, smoke DDP 30 pas + rechargement, à passer sur le pod loué avant le run.

- **2026-09-13 (P-SSM.2b NE SE RÉPLIQUE PAS sur 1.2850 : hybride k ≤ 4 0.7920 / 0.5471 / couv.
  0.725 contre backtest dur 0.7901 / 0.5475 / 0.720 — 0.04 pt, bruit)** — Bilan sur deux
  checkpoints : +0.47 pt (1.2942), +0.04 pt (1.2850). Le bouton sur sa plage entraînée n'est
  pas un gain établi : match nul avec la décimation. Même dessin par config (gains à k ≤ 4
  par le bouton : electricity/15T/short 0.103 vs 0.112, ett1/15T/long, m_dense/H/medium ;
  pertes quand l'hybride glisse vers une grosse décimation : bizitobs_service/short k32,
  saugeen/D k24, solar/10T/medium k8 0.369 vs 0.326, ett1/H/long k6). Le checkpoint plus
  entraîné change de cadence moins souvent (35/97 contre 41/97) : le gain du rythme
  décroît avec l'entraînement, bouton comme décimation. À conditions identiques, 1.2942
  reste devant 1.2850 (0.5445 contre 0.5475 en backtest dur) : les 0.7689 de MASE du stack
  sur 1.2850 viennent des couches, pas du checkpoint ; champion inchangé. Wide est le
  dernier test du bouton (P-SSM.2c gravée) ; en cas d'échec, la ligne du papier est un
  résultat négatif propre : le corps LTI tient, le bouton ne bat pas la décimation.

- **2026-09-13 (P-SSM.2b TIENT sur le champion 1.2942 : hybride bouton k ≤ 4 + décimation
  au-delà 0.7906 / 0.5398 / couv. 0.765 contre backtest dur 0.7892 / 0.5445 / 0.768, dur, sans
  flip, 97 configs : CRPS −0.47 pt (seuil 0.3), MASE +0.14 dans le bruit)** — Sur sa plage
  entraînée [1/4, 4], le bouton bat la décimation. Par config : gains là où le sélecteur
  prend le bouton à k ≤ 4 (m_dense/H/medium k4 vs 2 : 0.129 vs 0.147 ; us_births/D k3 vs 1 ;
  jena/H/long k3 vs 2 ; electricity/15T/short k4 vs 4 : 0.099 vs 0.109 ; solar/H/short k3
  vs 2), pertes là où l'hybride glisse vers une grosse décimation que le backtest pur ne
  prenait pas (solar/10T/medium k12 vs 3 : 0.376 vs 0.329 ; kdd/H/long k8 vs 3 ; bizitobs_l2c
  k6-8 vs 2-4) — la table mixte bouton / décimation déplace le winner's curse. Le stack
  flip + mix vaut 1.6 pt sur ce checkpoint (0.5445 → 0.5282), même ordre que sur TimeJEPA.
  Décision : pas de mode supplémentaire dans le harnais ; wide (bouton sur toute la plage,
  plus de candidats décimés) est le test propre. Second checkpoint (1.2850) en cours.

- **2026-09-13 (TABLE COMPLÈTE À 97 SUR 11 CHECKPOINTS, 5 % à 55 % du run : PLATEAU à
  0.528-0.533 depuis 25 % ; champion inchangé `1.2942` ; décision de couper le run)** —
  Stack flip + mix + pool : 5 % 0.5761 · 10 % 0.5419 · 15 % 0.5305 · 20 % 0.5334 · 25 %
  **0.5282** (0.7717, couv. 0.805) · 30 % 0.5301 · 35 % 0.5311 · 40 % 0.5292 · 45 % 0.5296 ·
  50 % 0.5278 (0.7762, couv. 0.719) · 55 % 0.5289 (0.7689, couv. 0.729). Huit checkpoints
  consécutifs sous ou au niveau de head8 (0.5340). Le 50 % gagne 0.04 pt de CRPS sur le
  25 % et lui perd 0.45 pt de MASE et 8.6 pt de couverture : même bande, le 25 % reste le
  champion (seul à gagner sur les trois axes). Couverture en dérive descendante depuis 25 %
  (0.805 → 0.66-0.73, la pinball resserre le fan), à traiter au papier comme limite
  partagée avec le transformer. Les 45 % restants (2.3 j) sont la fin du cosinus, sans gain
  attendu (anneal-30) : run coupé, GPU au bras wide-Δ depuis le dernier checkpoint (1.2850,
  le plus entraîné), P-SSM.2c gravée plus haut.

- **2026-09-13 (PRÉPARATION DU SCALING : checkpointing d'activations et option FSDP, code
  livré, non couru sur GPU)** — Décision utilisateur : après le verdict wide et un second
  seed, scaler TimeSSM sur un pod loué (8× RTX 5090, ~1 $/h/GPU). À un token par pas, la
  mémoire est dans les activations (batch × 1280 × 2·d_model × blocs), pas dans les poids :
  `model.ssm.activation_checkpointing` recompute chaque bloc en backward (train seulement,
  inerte en eval ; test : mêmes sorties et mêmes gradients en float64). `trainer.strategy:
  fsdp` construit une `FSDPStrategy` (FULL_SHARD, un unit par `GatedSSMBlock`, checkpoints
  en un fichier pour le harnais, policy de checkpointing si demandée) ; test de
  construction seulement, la validation multi-GPU se fait sur le pod avant toute mention
  au CV. Budget chiffré (registre TimeJEPA du jour) : un 10M à exposition du champion SSM
  ≈ 15 h ≈ 120 $ sur 8× 5090, un 20M le double ; un 100M à un token par pas n'entre pas
  dans 1 000 $ (≈ 5 j de 8× H100) sans patcher, ce qui casserait le bouton Δ. Corpus v3
  reconstructible par `../TimeJEPA/scripts/build_corpus_v3.sh` (révisions HF épinglées,
  audit 106 fichiers / 15.85 Md).

- **2026-09-13 (CORRECTION D'ÉCHELLE : les « % » des checkpoints SSM sont des % du RUN borné à
  30 %, pas de l'époque ; le champion « 25 % » est à 7.5 % de l'époque, 223M fenêtres, contre
  746M pour head8 à 25 % de l'époque)** — L'époque du sampler vaut 2.98 Md de fenêtres quel
  que soit le batch (7.76M × 128 ou 2.59M × 384 par GPU) ; le run SSM est borné à 30 % de
  l'époque (894M fenêtres, = anneal-30) et ses checkpoints tombent tous les 5 % du run (1.5 %
  de l'époque). Table compute : head8 25 % époque 746M fenêtres / 1.4 j GPU (0.5340) ; head8
  5 % époque 149M / 0.3 j (0.5585) ; anneal-30 30 % époque 894M / 1.7 j (0.5375) ; SSM 6 %
  époque 179M / 1.0 j (0.5334) ; SSM 7.5 % époque 223M / 1.3 j (0.5282, champion) ; SSM 9 %
  époque 268M / 1.6 j (0.5301, 2e sous head8) ; SSM fin de run 30 % époque 894M / 5.2 j. Le
  SSM coûte 3× par fenêtre (2000 contre 6100 fenêtres/s sur 3 GPU) : à temps GPU égal il bat
  le champion, et avec 3.3× moins de données. Conséquence pour la coupe : le « pic à 25 % puis
  dérive » de TimeJEPA (G7.3c) est à 25 % de l'ÉPOQUE, où le SSM n'est pas encore (LR à 80 %
  du pic) — laisser courir au moins jusqu'à 15-20 % de l'époque avant de décider. Le stack
  30 %-du-run (1.2936-v1) : 0.7760 / 0.5301 / couv. 0.693 sur 97, deuxième checkpoint sous
  head8 ; couverture en baisse (0.805 → 0.693), à suivre.

- **2026-09-13 (NOUVEAU CHAMPION : `epoch00_valloss1.2942` (25 % du run), stack flip + mix +
  pool, 0.7717 / 0.5282 / couverture 0.805 sur 97 configs, contre head8 0.7842 / 0.5340 /
  0.756)** — Table à 97 : 5 % 0.5761, 10 % 0.5419, 15 % 0.5305, 20 % 0.5334, 25 % 0.5282.
  Trois checkpoints consécutifs à ou sous le champion ; le 25 % gagne sur les trois axes,
  MASE −1.25 pt, CRPS −0.58 pt (au bord de la bande de bruit, 0.6 pt), couverture +4.9 pt —
  c'est la MASE et la couverture qui sortent le verdict du doute, pas le CRPS seul. Un
  scratch LTI de 2.5M, sans pretrain, sans attention dans le corps, sans conv, à un quart de
  son budget, à inférence strictement identique (RateIN par décimation, sans le bouton).
  Lecture : sur ce corpus le prior « état continu + horloge » vaut plus que l'attention ; le
  pretrain JEPA (≈ 1 pt) n'était pas ce qui manquait. Réserves : 30-45 % encore sur des
  sous-ensembles (71-74 configs, OOM d'une éval parallèle), à compléter ; couverture à
  suivre (0.679-0.720 sur les sous-ensembles suivants). Suite : passe complète 30-45 %,
  série delta-k4 par checkpoint, bras wide depuis le meilleur.

- **2026-09-12 (TABLE À 97 SUR LES PREMIERS CHECKPOINTS : P-SSM.1 TIENT, égalité avec le
  champion à 15-20 % du budget ; bras à plage large prêt, non lancé)** — Stack officiel
  (flip + mix + pool, décimation), GPU libres, batch 64, 97 configs : 5 % 0.8560 / 0.5761 /
  couv. 0.763 ; 10 % 0.7948 / 0.5419 / 0.760 ; 15 % 0.7757 / 0.5305 / 0.730 ; 20 % 0.7839 /
  0.5334 / 0.732 ; champion head8 0.7842 / 0.5340 / 0.756. P-SSM.1 (bande 0.545-0.575 à 5 %) :
  0.5761 à 5 % est au bord, avec le 5 % au milieu du warmup (LR 1.5e-4) ; à 10 % le modèle est
  sous la bande. Le 15 % est le meilleur chiffre jamais mesuré sur ce projet, MASE comprise,
  mais à 0.35 pt du champion il est dans la bande de bruit checkpoint à checkpoint mesurée
  sur anneal-30 (0.5352-0.5412) : pas un nouveau champion par notre propre règle ; couverture
  plus basse (0.730). Pente aplatie entre 15 et 20 % ; val_loss en plateau 1.289-1.295 depuis
  20 %. LEÇON D'INSTRUMENT : les tables de la veille mélangeaient des sous-ensembles de 55 à
  70 configs (OOM pendant le run, les lourdes manquantes) et donnaient 0.508-0.524 — jamais
  comparables ; `eval_checkpoints_ssm.sh` affiche désormais n_cfg et cached/computed/failed.
  Budget réel : le run borné à 30 % de l'époque = 2.33M batchs par GPU à 5.2 it/s = 5.2 jours
  (pas 2.7, erreur de ma part au lancement), checkpoint tous les 6 h ; à 42 % du run le 12/09.
  **Décision à prendre** : couper au meilleur checkpoint et lancer le bras à plage large, ou
  laisser finir (3 jours). **Bras à plage large** (`configs/ssm_mini_v3_wide.yaml`) : reprise
  des POIDS du meilleur checkpoint (`pretrained_encoder_path`, optimiseur et cosinus neufs),
  `delta_scales` = les 13 valeurs de K_CANDIDATES de 1/48 à 4, p 0.7, LR 1e-4, budget 3 % de
  l'époque (10 % du spike, ~12 h), 5 checkpoints ; smoke CPU passé (52 clés chargées).
  **P-SSM.2c** (gravée) : sur le checkpoint wide, `+ratein=delta` sans max_k ≥ `+ratein=backtest`
  de 0.3 pt et disparition des pertes à k ≥ 16 (bizitobs_service, bizitobs_l2c) ; ÉCHEC si
  delta ≤ backtest : le bouton ne transfère pas aux grands k même entraîné, l'hybride reste
  la couche officielle.

- **2026-09-11 (P-SSM.2 ÉCHOUE sur le checkpoint 1.3022 : delta 0.8276 / 0.5558 contre
  backtest 0.7953 / 0.5409, dur, sans flip, 97 configs ; DIAGNOSTIC MESURÉ : le bouton gagne
  sur sa plage entraînée (k ≤ 8) et perd au-delà (k ≥ 16))** — Run corrigé (schedule),
  checkpoint ~10 % du budget (val 1.3022). Sélecteur identique des deux côtés, pooling CRPS,
  marge 5 %. Delta active le bouton sur 28 configs, la décimation sur 32. Par config
  (`compare_subset.py`) : à k égal 16 (bizitobs_service ×3, bizitobs_application/long) delta
  perd 11-23 % — à w = 1/16 le bouton est moins bon que la décimation par 16 ; là où delta
  choisit un k énorme (sz_taxi 48, kdd/D 24, bizitobs_l2c/medium 32) le backtest accepte un w
  que le test punit (winner's curse hors distribution) ; à k ≤ 8 des deux côtés delta gagne
  (bitbrains_fast_storage/5T/long 0.606 contre 0.898, /medium 0.642 contre 0.691, bitbrains_rnd
  /5T/long, solar/H/short, ett2/H où delta refuse à raison un k = 6 que la décimation accepte à
  tort). Cause : `delta_scales` [0.25..4] n'entraîne le bouton que jusqu'à k = 4, le harnais
  demande w jusqu'à 1/48 ; à Δ/48 les constantes de temps sont 48× hors de tout ce que le
  modèle a vu, alors qu'un contexte décimé par 48 reste un contexte court à Δ, en
  distribution. Deux suites : (1) garde `+ratein_delta_max_k=N` dans le harnais (bouton
  jusqu'à N, décimation au-delà, même sélecteur ; livrée, test stub), à mesurer sur 1.3022 à
  N = 4 et 8 ; (2) second bras d'entraînement, `delta_scales` sur toute la plage demandée
  (1/48 à 4). Aussi : P-SSM.1 encore à lire sur 97 (stack complet sur 55 : 0.5045 contre
  0.5041 pour le champion sur les mêmes configs — égalité, pas domination ; les 42 manquantes
  sont les longues, tombées en OOM pendant le run). Vitesse : m4_monthly et
  temperature_rain dominent l'éval (48k et 96k instances), batch 64 puis 8.

  **P-SSM.2b** (hybride, même checkpoint) : `+ratein=delta +ratein_delta_max_k=4` bat
  `+ratein=backtest` d'au moins 0.3 pt de CRPS ; N = 8 dans le bruit de N = 4. ÉCHEC si
  l'hybride ≤ backtest : le bouton n'apporte rien même sur sa plage, et la décimation
  reste la couche officielle.

- **2026-09-10 (PREMIER RUN INVALIDE : schedule étiré ×3 par l'accumulation ; correctifs
  bf16 et vitesse ; run à relancer)** — Trois problèmes rencontrés au lancement sur le pod,
  tous corrigés et poussés : (1) `bf16-mixed` plante dans la FFT (cuFFT sans noyau
  bfloat16, autocast ne convertit pas les FFT) → convolution en float32 sous autocast
  (`4ed7a73`) ; (2) 2.7 it/s : un `w` uniforme par batch était traité comme « un Δ par item »,
  noyau dupliqué 128× et 128 FFT du noyau par bloc → chemin à noyau unique (`e568a5d`),
  5.3 it/s ensuite ; (3) le scheduler de TimeJEPA comptait des batchs alors qu'il avance par
  pas d'optimiseur : avec `accumulate_grad_batches: 3`, warmup et cosinus ×3, TOUT le run
  borné à 30 % était du warmup (LR 1.5e-5 au checkpoint 5 %, val 1.33, courbe `lr-AdamW` à
  pente 1/3 du scratch head8) → corrigé côté TimeJEPA (`// accumulate`). Le checkpoint 5 %
  de ce run ne mesure pas l'architecture ; son éval (stack, 24 configs vues : bitbrains
  0.47-0.88, electricity/15T 0.086) n'est pas retenue. Vitesse : le SSM à un token par pas
  coûte ~6× TimeJEPA par échantillon (1280 tokens contre 127 patchs, FFT bornée par la
  mémoire) ; le verdict P-SSM.2 se lit au checkpoint 5 % (~6 h à 5.3 it/s), le run à 30 %
  (~5 j) n'est engagé que si P-SSM.1 tient. Prédictions P-SSM.0-3 inchangées.

- **2026-09-09 (TimeSSM SPIKE — CODE LIVRÉ, NON COURU ; P-SSM.0 TENUE : l'équivariance
  au rythme passe à 1e-8)** — Branche `timessm` de TimeMamba, dépendance éditable sur
  TimeJEPA (aucune copie de code hors les 8 lignes d'`apply_schedule_fraction` et le bloc
  datamodule de `train.py`, non importables). Livré : `S4DLayer` (diagonal, ZOH exact,
  noyau Vandermonde + convolution FFT, récurrence `step`, `delta_scale` par batch ou par
  item, `Re(a) = −exp(·) < 0` par construction), `GatedSSMBlock` (sans conv depthwise par
  défaut : une conv est un filtre en pas, pas en temps physique — le test montre qu'elle
  casse l'équivalence décimation ≡ Δ/k ; `selective_readout` en ablation, porte sur la
  LECTURE seulement), `SSMForecaster` duck-typé JEPATST (RobustScale + RevIN + un token par
  pas + 6 blocs + tête quantile 1536 cross-attentive ; rollout autonome par un token futur
  appris, horizon libre 8-900 sans boucle ; `forecast(x, n, w)` avec `w` = facteur de Δ ;
  `rate_knob = 'delta'`, `core_prefixes` pour le chargeur), `SSMFinetuneModule` (module
  finetune de TimeJEPA + tirage d'un facteur Δ ∈ {0.25, 0.5, 1, 2, 4} avec p 0.5 par batch
  d'entraînement, cible inchangée, témoins `aug/delta_scale`), `train_ssm.py`, config
  `ssm_mini_v3` (recette du champion résolue et copiée ; déviations déclarées : batch 128 ×
  acc. 3, LR unique, multi-rythme, pas de conv), `eval_ssm.sh`. Côté TimeJEPA :
  `+ratein=delta` (même sélecteur backtest, contexte natif, `w = 1/k`, fan à l'horizon
  natif, diag `knob`), gardes `check_model_flags`, `model.builder` dans
  `create_model_from_config`, `core_prefixes` dans `load_checkpoint`. **Tests** (29 verts
  ici, 4 nouveaux dans TimeJEPA, anciens tests Mamba 20 verts sous l'extra `legacy`) :
  équivariance sur entrée constante par blocs (état et sortie, couche et bloc, 1e-8 en
  float64), écart croissant en k sur une sinusoïde, FFT == récurrence à L = 1024, |Ā| < 1
  et gradient fini à 1024 (l'échec du scan de 2025), `w` par item == boucle, contrat
  `forecast` à n ∈ {8, 128, 256, 900}, FinetuneModule (loss finie, gradients jusqu'aux
  blocs, groupes encoder/decoder), taille 2-4M (mesuré : 3.0M dont 0.77M de tête), harnais
  GIFT en off / flip / mix-pool / backtest / delta, aller-retour checkpoint par le chargeur
  TimeJEPA. **Smoke** : train 20 pas CPU sur corpus factice (loss 0.83 → val 0.734) ; éval
  du checkpoint smoke (46k paramètres, non entraîné) sur m_dense/D/short : off = delta à
  K = 1 bit-identiques (1.1638 / 0.1417), backtest choisit k = 6 par décimation, delta
  refuse tous les k (ratios 1.01-1.19 : le bouton d'un modèle non entraîné au multi-rythme
  ne transfère pas — c'est précisément ce que le run doit trancher). Coût du backtest à
  11 candidats : ~2× celui de TimeJEPA (contexte natif à chaque k au lieu de L/k).

  **Prédictions gravées** (checkpoint 5 % = 1/6 du run borné à 30 %) :
  - **P-SSM.0** (tenue le jour 1) : le test d'équivariance passe ; sinon rien ne se lance.
  - **P-SSM.1** (stack flip + mix + pool, décimation, modèle-agnostique) : à 5 %, CRPS
    entre 0.545 et 0.575 (head8 5 % standard 0.5585, S4-c 0.5506) ; > 0.60 ⇒ architecture
    ou recette à revoir avant toute question de rythme.
  - **P-SSM.2** (le verdict) : sur le MÊME checkpoint, `+ratein=delta` ≥ `+ratein=backtest`
    d'au moins 0.3 pt de CRPS, et delta sur les 32 configs > 5 % de l'oracle-k capture
    ≥ 50 % du gain oracle (RateIN classique : 37 %). ÉCHEC-DIAGNOSTIC si delta ≤
    décimation : le bouton est exact mais la lecture/tête n'est pas invariante ⇒ vérifier
    `aug/delta_scale`, second bras `training.p_delta_scale=1.0`.
  - **P-SSM.3** (ablation, seulement si P-SSM.1 tient) : `model.ssm.selective_readout=true`
    ± 0.3 pt à 5 %.

  Commandes : runbook. À lancer par l'utilisateur après le verdict anneal-30 et xres.
