# External sources, second batch

Built by `scripts/fetch_sources_b.py` into `v4_ext_b/`. Every variable is one
univariate series. Series are cut to their first and last observed point,
constant series and series with fewer than 16 observations are dropped, holes stay
NaN and are read through the mask. None of these sources is one of the 28 GIFT-Eval
datasets.

| Subset | Rate | Holder | Licence | Origin | Transformations |
|---|---|---|---|---|---|
| `epa_pm25`, `epa_ozone`, `epa_no2` | H | US EPA, Air Quality System | Public domain | https://aqs.epa.gov/aqsweb/airdata/download_files.html | One series per monitor, 2019 to 2024 |
| `webtris_15T` | 15T | National Highways | OGL-UK-3.0 | https://webtris.nationalhighways.co.uk/api/swagger/ui/index | Total volume of a random sample of active sites |
| `nsw_traffic_H` | H | Transport for NSW | CC-BY-4.0 | https://opendata.transport.nsw.gov.au/dataset/nsw-roads-traffic-volume-counts-api | Permanent counters, one series per station, direction and class; data before 2025-10 |
| `kelmarsh_*`, `penmanshiel_*` | 10T | Cubico Sustainable Investments; C. Plumley, R. Takeuchi | CC-BY-4.0 | https://zenodo.org/records/16807551, https://zenodo.org/records/16807304 | Wind speed, power, ambient temperature, generator speed; 2018 to 2020, Penmanshiel WT11-15 |
| `pvdaq_5T`, `pvdaq_15T` | 5T, 15T | NREL; C. Deline et al. | CC-BY-4.0 | https://data.openei.org/submissions/4568 | Systems 2105, 2107, 9068; numeric channels on a regular grid |
| `n2o_wastewater` | 1T | L. D. Hansen, A. Rani, D. Ortiz Arroyo, P. Durdevic | CC-BY-4.0 | https://doi.org/10.17632/xmbxhscgpr.4 | Values with a non-zero quality flag removed; regular grid |
| `bosch_cnc_1S` | 1S | M.-A. Tnani, M. Feil, K. Diepold | BSD-3-Clause (repository) | https://github.com/boschresearch/CNC_Machining | 2 kHz vibration reduced to a 1-second RMS; runs of one operation joined |
| `darck_10S` | 10S | DARCK authors | CC-BY-4.0 | https://zenodo.org/records/17159850 | 1 Hz power reduced to 10-second means |
| `rae_10S` | 10S | S. Makonin, Z. J. Wang, C. Tumpach | CC-BY-4.0 | https://doi.org/10.7910/DVN/ZJW4LC | Mains and sub-meters, 1 Hz reduced to 10-second means |
| `cesnet_10T`, `cesnet_H` | 10T, H | CESNET | CC-BY-4.0 | https://zenodo.org/records/13382427 | Institution aggregates: flows, packets, bytes, destinations, ports, duration |
| `eia_W`, `eia_M`, `eia_Q`, `eia_A` | W, M, Q, A | US Energy Information Administration | Public domain | https://www.eia.gov/opendata/ | Bulk files PET, NG, TOTAL, SEDS, COAL |
| `wdi_A` | A | World Bank | CC-BY-4.0 | https://datacatalog.worldbank.org/search/dataset/0037712 | Country and indicator pairs with at least 20 years; random sample of 60,000 |
| `cdc_deaths_W` | W | US CDC, NCHS | Public domain | https://data.cdc.gov/d/muzy-jte6 | Deaths by jurisdiction and cause; datasets 3yf8-kanr and muzy-jte6 kept apart |
| `cdc_nndss_W` | W | US CDC, NNDSS | Public domain | https://data.cdc.gov/d/x9gk-5huc | Cases by area and disease |
| `wikipedia_projects_H` | H | Wikimedia Foundation | CC0-1.0 | https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews | Project totals, 2016 to 2025-09 |
| `wikipedia_articles_W` | W | Wikimedia Foundation | CC0-1.0 | https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews | Daily article views summed by week; articles drawn from monthly top lists |
| `eia930_H` | H | US Energy Information Administration | Public domain | https://www.eia.gov/electricity/gridmonitor/ | EIA-930 bulk file EBA: demand, forecast, generation by fuel and interchange of every balancing authority, UTC series |
| `google_power_5T` | 5T | Google | CC-BY-4.0 | https://github.com/google/cluster-data/blob/master/PowerData2019.md | Measured power utilization of the 57 power domains; points flagged as bad measurements removed |
| `azure_functions_2021` | 1T | Microsoft Azure | CC-BY-4.0 | https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsInvocationTrace2021.md | Invocations per minute of each function, counted by end minute; functions with fewer than 200 invocations left out |
