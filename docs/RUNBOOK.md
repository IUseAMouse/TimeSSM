# Runbook TimeSSM

Commandes pour installer, entraîner, évaluer et publier. Depuis la racine du dépôt,
branche `timessm`. L'entraînement et le harnais d'évaluation lisent les données dans
`../TimeJEPA/data/` (corpus `processed/lotsa_v3`, GIFT-Eval `gift_eval`), construites
avec le runbook de TimeJEPA.

## Environnement

```bash
export UV_PROJECT_ENVIRONMENT=~/venvs/timessm   # un venv hors de tout dossier synchronisé
uv sync --extra train --extra dev                # timejepa : checkout voisin ../TimeJEPA, sinon l'URL git
uv run pytest -q                                 # 73 tests, moins d'une minute sur CPU
```

L'inférence seule n'a besoin que du paquet : `pip install git+https://github.com/IUseAMouse/TimeSSM.git`.

## Entraînement

Trois étapes, chacune reprend les poids de la précédente avec un optimiseur neuf.
`train_ssm_loop.sh` relance après un plantage (sauvegarde horaire, reprise automatique)
et exige `SEED0`, une graine jamais utilisée par le run repris.

```bash
python scripts/train_ssm.py --config-name ssm_mini_v3 wandb.run_name=ssm-mini-v3          # de zéro, 30 % d'une époque
SEED0=<graine> scripts/train_ssm_loop.sh ssm_mini_v3_wide ssm-mini-v3-wide \
    '+training.pretrained_encoder_path=checkpoints/timessm_mini_v3_zs/pretrain_False/<ckpt>'   # plage de taux large
SEED0=<graine> scripts/train_ssm_loop.sh ssm_mini_v3_freq ssm-mini-v3-freq \
    '+training.pretrained_encoder_path=checkpoints/timessm_mini_v3_wide_zs/pretrain_False/<ckpt>'   # pas lié à la fréquence
```

Avant un run : `DEVICES=3 CONFIG=<config> scripts/preflight.sh` (tests, audit du corpus,
profil d'un pas, fumée DDP et rechargement). Pendant les dix premières minutes, les
témoins W&B : `aug/delta_scale` et `aug/delta_neq1_frac` pour les deux premières étapes,
`freq/labelled_frac`, `freq/n_unique`, `freq/w_min`, `freq/w_max` pour la troisième.
Mémoire : batch 128 par carte avec accumulation 3 ; en cas de dépassement,
`data.batch_size=64 trainer.accumulate_grad_batches=6`. DDP est le chemin validé.

## Évaluation GIFT-Eval

```bash
CK=checkpoints/timessm_mini_v3_freq_zs/pretrain_False/epoch00_valloss1.2814.ckpt
EVAL_CONFIG=timessm_2.5m_gift scripts/eval_ssm.sh $CK                                   # le chiffre publié
EVAL_CONFIG=ssm_mini_v3_freq_eval scripts/eval_ssm.sh $CK +freq_delta=true               # nu, fréquence donnée
EVAL_CONFIG=ssm_mini_v3_freq_eval scripts/eval_ssm.sh $CK +freq_delta=true +tta_flip=true +ratein=oracle   # plafond, diagnostic
```

`configs/timessm_2.5m_gift.yaml` porte tous les réglages du chiffre publié : retournement
de signe, resampling par backtest (`+ratein=mix +ratein_pool=true +ratein_k_up=2x3x4
+ratein_min_bt=4 +ratein_bt_windows=4`), pas lié à la fréquence et facteurs γ du fan.
Résultats dans `evaluation/<model.name>/<ckpt>/gift<tag>/`, en cache par configuration.
Tous les checkpoints d'un run en série : `scripts/eval_checkpoints_ssm.sh <dossier>`
(`STACK=` fixe les drapeaux, `ONLY=<stem>` un seul checkpoint). Une liste Hydra
s'écrit `2x3x4`, les virgules déclenchent un balayage.

Les facteurs γ se calibrent sur les fenêtres de validation du corpus, jamais sur GIFT :

```bash
scripts/calibrate_ssm.sh $CK --flip --config-name ssm_mini_v3_freq --frequency-table
```

Avant toute modification du modèle, du harnais ou des données,
`scripts/check_regression_ssm.sh` réévalue le checkpoint publié sur cinq configurations
et exige l'égalité avec les résultats en cache.

## Diagnostics

```bash
python scripts/gift_gap_ssm.py evaluation/<run>/<ckpt>/<tag> --competitors Toto-2.0-4m,FlowState-9.1M,TTM-R3-PT \
    --corpus-dir ../TimeJEPA/data/processed/lotsa_v3        # carte par configuration contre les concurrents
python scripts/profile_step.py --config ssm_mini_v3_freq --no-profile --scales 12   # vitesse d'un pas à cadences mêlées
```

## Publication des poids

```bash
python scripts/export_hf.py --checkpoint $CK --out export/timessm-2.5m
hf auth login
hf upload IUseAMouse/timessm-2.5m export/timessm-2.5m . --repo-type model
```

L'export écrit `config.json` et `model.safetensors` (lus par `timessm.from_pretrained`),
les facteurs γ, la table des fréquences, le checkpoint d'origine et la carte avec son
en-tête Hugging Face, et vérifie l'aller-retour au bit près.

## Règles de travail

Une variable par expérience. Les prédictions sont écrites dans
un journal d'expériences tenu hors du dépôt, avant le lancement, avec leurs seuils, et le verdict après.
Rien n'est réglé sur le test GIFT-Eval ; l'oracle est un diagnostic. Aucun fichier n'est
supprimé, il est déplacé. Code et commentaires en anglais, documents en français, carte
de modèle en anglais.
