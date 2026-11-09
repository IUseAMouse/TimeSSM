# Sources externes, troisième lot

Ce lot est construit par `scripts/fetch_sources_c.py` dans `v9_ext/`, avec les conventions de `fetch_sources_b.py`.

## Règles communes

- Chaque variable devient une série univariée.
- Chaque série est coupée entre son premier et son dernier point observé.
- Les séries constantes et celles de moins de 16 observations sont écartées.
- Les trous restent en NaN et sont lus par le masque.

## Critère du lot

Le critère est le nombre de séries distinctes, pas une faiblesse vue sur GIFT. Dans un sous-ensemble, on ne garde qu'une seule série par classe de séries égales à une transformation affine près. C'est le cas des indices rebasés et des changements d'unité.

Astuce signalée : l'empreinte d'une série est un hash de la série centrée réduite, arrondie à 3 décimales, les NaN à une valeur fixe. Deux séries qui ne diffèrent que par l'échelle et le décalage ont la même empreinte.

## Exclusion de GIFT

Aucune de ces sources ne fait partie des 28 jeux de GIFT-Eval, mais M4 a puisé dans Eurostat.

### Contrôle par valeur

Le contrôle par valeur passe chaque série de `v9_ext` contre chaque jeu de GIFT :

    python scripts/audit_overlap.py --gift /workspace/data/gift_eval --corpus /workspace/data/v9_ext \
        --out overlap_v9ext.json --verify --all --min-aligned 5

### Pourquoi l'option `--all`

La première passe de `audit_overlap.py` refait un `np.isin` contre chaque jeu de GIFT pour chaque série. À 5.4 millions de séries, cela prendrait des jours. L'option `--all` passe donc directement au contrôle par alignement, sur toutes les séries, avec une recherche dans une table triée. Ce contrôle est plus strict que la première passe : aucun filtre préalable, et aucune limite d'échantillon.

### Résultats

| Sous-ensemble | Séries | Signalées à 5 fenêtres | à 20 fenêtres | Jeu de GIFT |
|---|---|---|---|---|
| `eurostat_bulk_m` | 691 150 | 6 326 | 4 500 | M4 mensuel surtout |
| `eurostat_bulk_q` | 4 639 883 | 656 | 344 | M4 trimestriel surtout |
| `usgs_water_temp_15T` | 606 | 102 | 0 | M4 journalier et mensuel |
| `usgs_oxygen_15T`, `usgs_conductance_15T`, `usgs_discharge_15T` | 2 881 | 16 | 0 | M4, `jena_weather` (2) |
| `isd_temp_H` | 2 935 | 1 | 0 | M4 mensuel |
| Les 16 autres | 99 744 | 0 | 0 | |

### Lecture

- **Eurostat :** ce sont de vraies fuites. Les exemples s'alignent au décalage 0 sur 55 à 126 fenêtres.
- **Autres sources :** les correspondances ne tiennent qu'à 5 à 8 fenêtres. Ce sont des coïncidences, du même ordre que celles de CMIP6 ou PVDAQ contre M4 (DATA.md).

### Fusion

Comme pour v6, v9 écarte toutes les séries signalées à 5 fenêtres, soit 7 101 séries :

    python scripts/merge_corpus.py --out /workspace/data/v9 \
        --drop overlap_v9ext_verified_5.json /workspace/data/v8 /workspace/data/v9_ext

v9 compte 302 sous-ensembles et 14.9 milliards de points, soit environ 7.1 millions de séries réelles, contre 3.2 millions dans v8.

## Second lot : v10

Le second lot est écrit dans `v9_ext_b/` :

    python scripts/fetch_sources_c.py ndbc isd_more usgs_more \
        --out /workspace/data/v9_ext_b --raw /workspace/data/raw_d

| Sous-ensemble | Séries | Signalées à 5 fenêtres | à 20 fenêtres |
|---|---|---|---|
| `isd_temp_H_s2` | 3 899 | 2 | 0 |
| `isd_dew_H_s2`, `isd_pressure_H_s2`, `isd_wind_H_s2` | 10 621 | 0 | 0 |
| `usgs_discharge_15T_s2` | 2 328 | 2 | 0 |
| `ndbc_*_H` (9 canaux) | 4 424 | 0 | 0 |

Les 4 séries signalées s'alignent sur 5 fenêtres seulement, dont 2 sur la série 7703 de `m4_monthly`, déjà la cible des coïncidences du premier lot. Elles sont écartées par la même règle :

    python scripts/merge_corpus.py --out /workspace/data/v10 \
        --drop overlap_v9extb_verified_5.json /workspace/data/v9 /workspace/data/v9_ext_b

v10 compte 316 sous-ensembles et 16.0 milliards de points. Les échantillons `_s2` rejoignent la famille de leur premier échantillon : ils ajoutent des stations sans changer le poids de tirage de la famille.

## Sous-ensembles

| Sous-ensemble | Cadence | Détenteur | Licence | Origine | Transformations |
|---|---|---|---|---|---|
| `eurostat_bulk_m`, `eurostat_bulk_q` | M, Q | Union européenne, Eurostat | Politique de réutilisation d'Eurostat, source citée | https://ec.europa.eu/eurostat/web/main/data/database | Toutes les tables mensuelles et trimestrielles du catalogue, sauf celles des lots précédents. Une série par clé de dimensions : unités de taux de croissance écartées, série non désaisonnalisée préférée. Doublons affines retirés. |
| `eurostat_deaths_nuts3_w` | W | Union européenne, Eurostat | Politique de réutilisation d'Eurostat, source citée | https://ec.europa.eu/eurostat/databrowser/view/demo_r_mweek3 | Décès hebdomadaires par région NUTS 3, sexe et âge. Doublons affines retirés. |
| `usgs_discharge_15T`, `usgs_water_temp_15T`, `usgs_conductance_15T`, `usgs_oxygen_15T`, `usgs_ph_15T`, `usgs_turbidity_15T` | 15T | U.S. Geological Survey | Domaine public | https://waterservices.usgs.gov/ | Valeurs instantanées de 2021 à 2024, en moyenne sur une grille de 15 minutes. Échantillon aléatoire des stations actives de chaque paramètre, dans les 50 États et à Porto Rico. Premier capteur du site. Les stations de `usgs_streamflow` sont écartées. |
| `isd_temp_H`, `isd_dew_H`, `isd_pressure_H`, `isd_wind_H` | H | NOAA NCEI, Integrated Surface Database (ISD-Lite) | Domaine public | https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database | Échantillon aléatoire de 3 000 stations actives de 2020 à 2024. Les stations australiennes sont écartées, à cause de la proximité avec temperature_rain de GIFT, ainsi que toute station nommée Jena. |
| `isd_temp_H_s2`, `isd_dew_H_s2`, `isd_pressure_H_s2`, `isd_wind_H_s2` | H | NOAA NCEI, ISD-Lite | Domaine public | https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database | Second échantillon de 4 000 stations, mêmes règles, sans les stations du premier. |
| `usgs_discharge_15T_s2` | 15T | U.S. Geological Survey | Domaine public | https://waterservices.usgs.gov/ | Second échantillon de 3 000 sites de débit, mêmes règles, sans les sites déjà pris. |
| `ndbc_wspd_H`, `ndbc_gst_H`, `ndbc_wvht_H`, `ndbc_dpd_H`, `ndbc_apd_H`, `ndbc_pres_H`, `ndbc_atmp_H`, `ndbc_wtmp_H`, `ndbc_dewp_H` | H | NOAA National Data Buoy Center | Domaine public | https://www.ndbc.noaa.gov/data/historical/stdmet/ | Données météo-océaniques standard des bouées et stations côtières, 2020 à 2024, moyenne horaire. Les valeurs manquantes (99, 999, 9999) deviennent des trous. |
| `epa_co`, `epa_so2`, `epa_pm10`, `epa_wind`, `epa_temp`, `epa_pressure`, `epa_humidity` | H | US EPA, Air Quality System | Domaine public | https://aqs.epa.gov/aqsweb/airdata/download_files.html | Une série par moniteur et par paramètre, de 2021 à 2024. La direction du vent (un angle) est écartée. |
