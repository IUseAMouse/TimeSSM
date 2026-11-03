# External training data

Sources added to corpus v4 by `scripts/fetch_sources.py`, on top of
`Salesforce/GiftEvalPretrain`. None of them is one of the 28 datasets of
GIFT-Eval. Every variable is stored as a separate univariate series; leading
and trailing missing values are removed, missing values inside a series stay
missing. Constant series and series with fewer than 16 observations are dropped.

| Subset | Rate | Holder | License | Origin | Transformations |
|---|---|---|---|---|---|
| `azure_functions_2019` | 1T | Microsoft | CC-BY-4.0 | [AzurePublicDataset](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md) | about 4,000 of 72,000 functions, by hash; invocations per minute over 14 days; a missing day stays missing |
| `exathlon_spark` | 1S | Exathlon authors | Apache-2.0 | [exathlon](https://github.com/exathlonbenchmark/exathlon) | up to 24 varying metrics per trace |
| `rcaeval_microservices` | 1S | RCAEval authors | CC-BY-4.0 | [doi:10.5281/zenodo.14590730](https://doi.org/10.5281/zenodo.14590730) | RE1 and RE2 testbeds; up to 150 cases per archive, up to 24 varying metrics per case |
| `smd_servers` | 1T | OmniAnomaly authors | MIT | [OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) | train and test parts joined for each machine |
| `uci_hydraulic` | 1S | N. Helwig, E. Pignanelli, A. Schütze | CC-BY-4.0 | [UCI 447](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) | 60 s cycles concatenated; 100 Hz and 10 Hz channels averaged to 1 Hz |
| `uci_household_power` | 1T | G. Hebrail, A. Berard | CC-BY-4.0 | [UCI 235](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) | none |
| `uci_steel` | 15T | V E Sathishkumar, C. Shin, Y. Cho | CC-BY-4.0 | [UCI 851](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption) | seconds-since-midnight column removed |
| `borealis_6s` | 6S | O. Ardakanian, S. Huo, S. Keshav | CC0-1.0 | [doi:10.5683/SP2/R4SVBF](https://doi.org/10.5683/SP2/R4SVBF) | last two channels of each home |
| `eurostat_sts_m` | M | European Union | Eurostat re-use policy, source acknowledged | [sts_inpr_m, sts_trtu_m, sts_copr_m](https://ec.europa.eu/eurostat/web/short-term-business-statistics) | unadjusted series only |
| `eurostat_gdp_q`, `eurostat_gdp_a` | Q, A | European Union | Eurostat re-use policy, source acknowledged | [namq_10_gdp, nama_10_gdp](https://ec.europa.eu/eurostat/web/national-accounts) | current prices, million euro |
| `eurostat_deaths_w` | W | European Union | Eurostat re-use policy, source acknowledged | [demo_r_mwk_ts](https://ec.europa.eu/eurostat/databrowser/view/demo_r_mwk_ts) | none |
| `statcan_m` | M | Statistics Canada | Statistics Canada Open Licence | [tables 16-10-0047, 20-10-0008, 25-10-0015](https://www150.statcan.gc.ca/n1/tbl/csv/) | one series per vector |
| `worldbank_wdi_a` | A | World Bank | CC-BY-4.0 | [WDI](https://databank.worldbank.org/data/download/WDI_CSV.zip) | CO2, life expectancy and tourism indicators removed (fev-bench tasks) |
| `noaa_water_levels` | H | NOAA CO-OPS | Public domain | [Tides and Currents](https://tidesandcurrents.noaa.gov/) | 120 stations sampled, verified hourly heights 2015-2024 |
| `usgs_streamflow` | 15T | U.S. Geological Survey | Public domain | [NWIS](https://waterservices.usgs.gov/) | discharge 2023-2024, 25 stations in each of 12 states |

Adapted from Statistics Canada, tables 16-10-0047-01, 20-10-0008-01 and
25-10-0015-01. This does not constitute an endorsement by Statistics Canada of
this product. Source: Eurostat; modified data, Eurostat is not responsible for
this re-use.

## Overlap with GIFT-Eval, checked by value

The M4 data behind GIFT-Eval are anonymized, so names cannot rule out a shared
series. `scripts/audit_overlap.py` hashes every window of 8 consecutive growth
ratios, rounded to 1e-3, which a change of units leaves intact. A first pass
flags corpus series sharing 3 windows with a GIFT-Eval dataset; the `--verify`
pass keeps those whose shared windows sit at one offset between the two series,
which a shared stretch of history produces and chance does not.

At 20 aligned windows every hit is plausible: the EIA series (2 870 annual,
1 030 monthly, 235 quarterly, 4 weekly) found in M4, FRED-MD, Eurostat, the WDI,
M1, M3 and tourism series found in M4, and the `weather` stations found in
`temperature_rain_with_missing`. At 5 a few coincidences appear (CMIP6 or PVDAQ
against M4). Corpus v6 leaves out every series flagged at 5, about 5 000
series, the whole `weather` subset, and `wdi_A`, which duplicates
`worldbank_wdi_a`:

    python scripts/merge_corpus.py --out /workspace/data/v6 \
        --drop overlap_verified_5.json <roots> --drop-subset weather wdi_A

The first pass scans at most 20 000 series per subset, so a subset larger than
that is cleared on a sample.
