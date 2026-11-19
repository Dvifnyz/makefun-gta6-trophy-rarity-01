# makefun-gta6-trophy-rarity-01
[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Dvifnyz/makefun-gta6-trophy-rarity-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Dvifnyz/makefun-gta6-trophy-rarity-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
A statistical forecaster for Grand Theft Auto VI's PlayStation 5 trophy earn rates on Exophase at 2026-11-26T00:00:00Z, a week after launch. It reads one case file (the house's reads of the game's trophy page up to the checkpoint) and writes five quantiles per trophy: a recency-weighted log-time trend fit on the logit scale with tier-shrunk slopes, a spread from process noise, fit uncertainty and page staleness (refreshes dilute rates, so the lower tail is wider), and a tight band for trophies already at 100%.
## Run it
```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```
Reads every `<id>.in` in the input directory and writes `<id>.out` to the output directory. Node 22, standard library only, no network, deterministic.
<sub>Entered on [make.fun](https://make.fun/?ref=b8pznreq&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
