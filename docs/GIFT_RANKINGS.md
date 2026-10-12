# Classements GIFT-Eval — fondations seules (snapshot 2026-09-06)

Source : `docs/assets/gift_leaderboard/2026-09-06/` (leaderboard.csv + models_meta.csv, 127 entrées, agrégats recalculés avec la formule officielle : moyenne géométrique des ratios par config contre la Seasonal Naive officielle).

Filtre « fondation » : `model_type` ∈ {zero-shot, pretrained} et `testdata_leakage` = No. Exclus : `agentic` (routeurs et ensembles de plusieurs modèles), `fine-tuned` (adaptation par jeu de données), `deep-learning` (entraînés par jeu), `statistical`, et toute entrée avec fuite déclarée. Les lignes en gras sont nos checkpoints, insérés à leur rang (nos agrégats : même formule, mêmes CSV officiels de la Seasonal Naive).

Sur 127 entrées, 63 sont des fondations au sens ci-dessus. Exclus en plus par leur nom (enveloppes déclarées `pretrained`) : STRIDE (+Chronos-2), STRIDE (+Timer-S1).

### Fondations, top 40 (sur 63)

| # | modèle | params (M) | CRPS | MASE | type |
|---|---|---|---|---|---|
| 1 | TimesFM-3 | 330.7 | 0.4557 | 0.6668 | zero-shot |
| 2 | EXAONE-Forecast | n/a | 0.4600 | 0.6733 | zero-shot |
| 3 | DeOSAlphaTimeGPTPredictor-2025 | n/a | 0.4663 | 0.6815 | zero-shot |
| 4 | TiRex-2-Pretrained | n/a | 0.4669 | 0.6777 | pretrained |
| 5 | Granite-PatchTST-FM-r2 | 384.6 | 0.4672 | 0.6846 | zero-shot |
| 6 | Toto-2.0-2.5B | 2454.3 | 0.4759 | 0.6956 | pretrained |
| 7 | CHARM | n/a | 0.4776 | 0.7582 | zero-shot |
| 8 | TiRex-2-Zeroshot | n/a | 0.4781 | 0.6973 | zero-shot |
| 9 | Toto-2.0-1B | 1041 | 0.4784 | 0.6992 | pretrained |
| 10 | Zeus | 102.1 | 0.4803 | 0.6931 | pretrained |
| 11 | tafsut | 105.3 | 0.4809 | 0.6926 | pretrained |
| 12 | Toto-2.0-313m | 312.7 | 0.4814 | 0.7028 | pretrained |
| 13 | EFG-base | n/a | 0.4816 | 0.7006 | zero-shot |
| 14 | PatchTST-FM-r1 | 257.9 | 0.4829 | 0.7069 | zero-shot |
| 15 | Timer-s1 | 8303.7 | 0.4853 | 0.6934 | pretrained |
| 16 | chronos-2 | 119.5 | 0.4854 | 0.6978 | pretrained |
| 17 | Falcon-X | n/a | 0.4857 | 0.6873 | pretrained |
| 18 | Falcon-2.0 | n/a | 0.4863 | 0.6660 | pretrained |
| 19 | FlowState-r1.1 | 18.5 | 0.4866 | 0.7015 | zero-shot |
| 20 | Granite-PatchTST-FM-r1 | 257.9 | 0.4877 | 0.7171 | zero-shot |
| 21 | Xihe-ultra | n/a | 0.4880 | 0.7011 | zero-shot |
| 22 | TiRex | n/a | 0.4885 | 0.7158 | zero-shot |
| 23 | Granite-FlowState-r1.1 | 18.5 | 0.4901 | 0.7014 | zero-shot |
| 24 | TimesFM-2.5 | 231.3 | 0.4903 | 0.7050 | zero-shot |
| 25 | Xihe-max | n/a | 0.4905 | 0.7109 | zero-shot |
| 26 | LongSeer-v1.0 | n/a | 0.4907 | 0.7101 | zero-shot |
| 27 | t0-alpha | 101.6 | 0.4941 | 0.7240 | pretrained |
| 28 | chronos-2-synth | 119 | 0.4958 | 0.7203 | zero-shot |
| 29 | Toto-2.0-22m | 21.9 | 0.4963 | 0.7188 | pretrained |
| 30 | VISIT-2.0 | n/a | 0.4996 | 0.7138 | pretrained |
| 31 | FlowState-9.1M | 9.1 | 0.5019 | 0.7262 | zero-shot |
| 32 | **TimeSSM-2.5M+resampling** | 2.5 | 0.5128 | 0.7544 | zero-shot |
| 33 | Moirai2 | 11.4 | 0.5164 | 0.7281 | pretrained |
| 34 | Toto_Open_Base_1.0 | 151.3 | 0.5173 | 0.7501 | zero-shot |
| 35 | TTM-R3-PT | 1.4 | 0.5195 | 0.7240 | pretrained |
| 36 | Toto-2.0-4m | 4.1 | 0.5242 | 0.7565 | pretrained |
| 37 | TempoPFN | n/a | 0.5327 | 0.7875 | zero-shot |
| 38 | recursive-moirai-2 | n/a | 0.5345 | 0.7709 | pretrained |
| 39 | CleanTS-65M | 65.4 | 0.5433 | 0.7981 | zero-shot |
| 40 | tabpfn_ts | n/a | 0.5441 | 0.7709 | zero-shot |

### Fondations de moins de 10M de paramètres

| # | modèle | params (M) | CRPS | MASE | type |
|---|---|---|---|---|---|
| 1 | FlowState-9.1M | 9.1 | 0.5019 | 0.7262 | zero-shot |
| 2 | **TimeSSM-2.5M+resampling** | 2.5 | 0.5128 | 0.7544 | zero-shot |
| 3 | TTM-R3-PT | 1.4 | 0.5195 | 0.7240 | pretrained |
| 4 | Toto-2.0-4m | 4.1 | 0.5242 | 0.7565 | pretrained |
| 5 | **TimeSSM-2.5M** | 2.5 | 0.5451 | 0.7985 | zero-shot |
| 6 | TinyCast | 0.1 | 0.5454 | 0.7738 | zero-shot |
| 7 | goia-forecast-nano-v0 | 4.7 | 0.5527 | 0.8152 | zero-shot |
| 8 | Kairos_10m | 9.9 | 0.5541 | 0.7527 | zero-shot |
| 9 | Metamorph1.0-4.5M | 4.5 | 0.5549 | 0.7761 | pretrained |
| 10 | YingLong_6m | 7.3 | 0.6090 | 0.8802 | zero-shot |

Notes : TempoPFN n'a pas de compte de paramètres dans les métadonnées ; son article (arXiv 2510.25502) indique 34.69M, il n'est donc pas dans la table sub-10M. FlowState apparaît trois fois (FlowState-9.1M, FlowState-r1.1 et Granite-FlowState-r1.1 : une lignée, deux versions) — un seul modèle au sens du classement. TTM-R3-FT est exclu (type fine-tuned), TTM-R3-PT est la référence pretrained. Le type est auto-déclaré par les auteurs : le filtre est le meilleur possible depuis les métadonnées, pas une vérité absolue.
