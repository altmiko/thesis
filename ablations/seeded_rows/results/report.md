# Fresh-row Prim-PGD untargeted joint comparison

Each attack seed (42, 2024, 2026) selects a fresh, random, clean-correct 800 flows per attack class per victim. Each budget uses identical rows within a seed; seed 42 exactly matches FINAL selection. Other seeds have partly overlapping, **not paired**, cohorts. Victims are reported separately.

## Three-seed rates

Rates are class-balanced within each victim (four classes × 800). SD is the sample SD across three seed rates (ddof=1), not an uncertainty interval. Clopper–Pearson (CP), Wilson and class-stratified bootstrap 95% intervals refer to the reference seed 42 only (n=3,200 per victim). The 10,000-replicate bootstrap resamples flows within each class at that seed, conditional on this test split and frozen victim. These intervals do not measure variation across attack seeds or new campaigns; the binomial intervals assume independent flows. Cohorts overlap, so three-seed pooled-flow intervals would overstate precision.

|Dataset|Victim|Budget|Outcome|Seed 42 / 2024 / 2026|Mean ± SD|CP 95%|Wilson 95%|Stratified bootstrap 95%|
|---|---|---|---|---|---|---|---|---|
|cicids2017_distrinet|mlp|p75|raw|4.09% / 4.06% / 4.19%|4.11% ± 0.07%|3.43%–4.84%|3.46%–4.84%|3.44%–4.78%|
|cicids2017_distrinet|mlp|p75|valid|4.09% / 4.06% / 4.19%|4.11% ± 0.07%|3.43%–4.84%|3.46%–4.84%|3.44%–4.78%|
|cicids2017_distrinet|mlp|unbounded|raw|22.97% / 23.62% / 22.44%|23.01% ± 0.59%|21.52%–24.47%|21.54%–24.46%|21.78%–24.16%|
|cicids2017_distrinet|mlp|unbounded|valid|22.97% / 23.62% / 22.44%|23.01% ± 0.59%|21.52%–24.47%|21.54%–24.46%|21.81%–24.12%|
|cicids2017_distrinet|cnn|p75|raw|13.47% / 12.47% / 12.88%|12.94% ± 0.50%|12.30%–14.70%|12.33%–14.70%|12.38%–14.56%|
|cicids2017_distrinet|cnn|p75|valid|13.47% / 12.47% / 12.88%|12.94% ± 0.50%|12.30%–14.70%|12.33%–14.70%|12.41%–14.56%|
|cicids2017_distrinet|cnn|unbounded|raw|59.94% / 59.19% / 59.88%|59.67% ± 0.42%|58.22%–61.64%|58.23%–61.62%|58.97%–60.94%|
|cicids2017_distrinet|cnn|unbounded|valid|59.94% / 59.19% / 59.88%|59.67% ± 0.42%|58.22%–61.64%|58.23%–61.62%|58.94%–60.94%|
|cicids2017_distrinet|ft_transformer|p75|raw|0.12% / 0.16% / 0.16%|0.15% ± 0.02%|0.03%–0.32%|0.05%–0.32%|0.03%–0.25%|
|cicids2017_distrinet|ft_transformer|p75|valid|0.12% / 0.16% / 0.16%|0.15% ± 0.02%|0.03%–0.32%|0.05%–0.32%|0.03%–0.25%|
|cicids2017_distrinet|ft_transformer|unbounded|raw|0.59% / 0.62% / 0.59%|0.60% ± 0.02%|0.36%–0.93%|0.38%–0.93%|0.34%–0.88%|
|cicids2017_distrinet|ft_transformer|unbounded|valid|0.59% / 0.62% / 0.59%|0.60% ± 0.02%|0.36%–0.93%|0.38%–0.93%|0.34%–0.88%|
|cicids2018_distrinet|mlp-s42|p75|raw|2.53% / 2.25% / 1.88%|2.22% ± 0.33%|2.02%–3.14%|2.04%–3.14%|2.00%–3.06%|
|cicids2018_distrinet|mlp-s42|p75|valid|2.53% / 2.25% / 1.88%|2.22% ± 0.33%|2.02%–3.14%|2.04%–3.14%|2.00%–3.06%|
|cicids2018_distrinet|mlp-s42|unbounded|raw|44.31% / 44.75% / 44.25%|44.44% ± 0.27%|42.58%–46.05%|42.60%–46.04%|43.34%–45.28%|
|cicids2018_distrinet|mlp-s42|unbounded|valid|44.31% / 44.75% / 44.25%|44.44% ± 0.27%|42.58%–46.05%|42.60%–46.04%|43.34%–45.25%|
|cicids2018_distrinet|cnn-s42|p75|raw|1.16% / 1.19% / 1.00%|1.11% ± 0.10%|0.82%–1.59%|0.84%–1.59%|0.81%–1.53%|
|cicids2018_distrinet|cnn-s42|p75|valid|1.16% / 1.19% / 1.00%|1.11% ± 0.10%|0.82%–1.59%|0.84%–1.59%|0.81%–1.53%|
|cicids2018_distrinet|cnn-s42|unbounded|raw|26.28% / 26.34% / 26.34%|26.32% ± 0.04%|24.76%–27.84%|24.79%–27.83%|25.91%–26.69%|
|cicids2018_distrinet|cnn-s42|unbounded|valid|26.28% / 26.34% / 26.34%|26.32% ± 0.04%|24.76%–27.84%|24.79%–27.83%|25.91%–26.69%|
|cicids2018_distrinet|ft_transformer-s42|p75|raw|0.00% / 0.09% / 0.03%|0.04% ± 0.05%|0.00%–0.12%|0.00%–0.12%|0.00%–0.00%|
|cicids2018_distrinet|ft_transformer-s42|p75|valid|0.00% / 0.09% / 0.03%|0.04% ± 0.05%|0.00%–0.12%|0.00%–0.12%|0.00%–0.00%|
|cicids2018_distrinet|ft_transformer-s42|unbounded|raw|0.12% / 0.28% / 0.38%|0.26% ± 0.13%|0.03%–0.32%|0.05%–0.32%|0.03%–0.25%|
|cicids2018_distrinet|ft_transformer-s42|unbounded|valid|0.12% / 0.28% / 0.38%|0.26% ± 0.13%|0.03%–0.32%|0.05%–0.32%|0.03%–0.25%|

Per-class three-seed rates and intervals are in `summary.csv`.

## Reference seed 42: paired tests

Only valid-ASR p75 vs unbounded has a prespecified six-victim Holm family (α=0.05). McNemar is exact binomial below 25 discordants, otherwise continuity-corrected chi-square; differences have 95% Newcombe paired intervals. Raw-vs-valid within each budget and the raw p75-vs-unbounded comparisons are **descriptive, unadjusted diagnostics**; their p-values are not multiplicity-controlled findings. No Cochran Q or cross-seed paired test is performed.

|Dataset|Victim|Comparison|Budget|Outcome|Difference pp [95% CI]|A-only / B-only|p|Holm p|Decision|
|---|---|---|---|---|---|---|---|---|---|
|cicids2017_distrinet|mlp|raw vs valid|p75|untargeted|0.00 [-0.13, 0.13]|0 / 0|1|not adjusted|descriptive only|
|cicids2017_distrinet|mlp|raw vs valid|unbounded|untargeted|0.00 [-0.09, 0.09]|0 / 0|1|not adjusted|descriptive only|
|cicids2017_distrinet|mlp|p75 vs unbounded|both|raw|-18.87 [-20.26, -17.54]|0 / 604|6.144e-133|not adjusted|descriptive only|
|cicids2017_distrinet|mlp|p75 vs unbounded|both|valid|-18.87 [-20.26, -17.54]|0 / 604|6.144e-133|1.843e-132|reject H0|
|cicids2017_distrinet|cnn|raw vs valid|p75|untargeted|0.00 [-0.11, 0.11]|0 / 0|1|not adjusted|descriptive only|
|cicids2017_distrinet|cnn|raw vs valid|unbounded|untargeted|0.00 [-0.07, 0.07]|0 / 0|1|not adjusted|descriptive only|
|cicids2017_distrinet|cnn|p75 vs unbounded|both|raw|-46.47 [-48.17, -44.71]|0 / 1487|<1e-300 (underflow)|not adjusted|descriptive only|
|cicids2017_distrinet|cnn|p75 vs unbounded|both|valid|-46.47 [-48.17, -44.71]|0 / 1487|<1e-300 (underflow)|<1e-300 (underflow)|reject H0|
|cicids2017_distrinet|ft_transformer|raw vs valid|p75|untargeted|0.00 [-0.13, 0.13]|0 / 0|1|not adjusted|descriptive only|
|cicids2017_distrinet|ft_transformer|raw vs valid|unbounded|untargeted|0.00 [-0.13, 0.13]|0 / 0|1|not adjusted|descriptive only|
|cicids2017_distrinet|ft_transformer|p75 vs unbounded|both|raw|-0.47 [-0.78, -0.24]|0 / 15|6.104e-05|not adjusted|descriptive only|
|cicids2017_distrinet|ft_transformer|p75 vs unbounded|both|valid|-0.47 [-0.78, -0.24]|0 / 15|6.104e-05|0.0001221|reject H0|
|cicids2018_distrinet|mlp-s42|raw vs valid|p75|untargeted|0.00 [-0.13, 0.13]|0 / 0|1|not adjusted|descriptive only|
|cicids2018_distrinet|mlp-s42|raw vs valid|unbounded|untargeted|0.00 [-0.06, 0.06]|0 / 0|1|not adjusted|descriptive only|
|cicids2018_distrinet|mlp-s42|p75 vs unbounded|both|raw|-41.78 [-43.49, -40.07]|0 / 1337|2.8e-292|not adjusted|descriptive only|
|cicids2018_distrinet|mlp-s42|p75 vs unbounded|both|valid|-41.78 [-43.49, -40.07]|0 / 1337|2.8e-292|1.4e-291|reject H0|
|cicids2018_distrinet|cnn-s42|raw vs valid|p75|untargeted|0.00 [-0.13, 0.13]|0 / 0|1|not adjusted|descriptive only|
|cicids2018_distrinet|cnn-s42|raw vs valid|unbounded|untargeted|0.00 [-0.08, 0.08]|0 / 0|1|not adjusted|descriptive only|
|cicids2018_distrinet|cnn-s42|p75 vs unbounded|both|raw|-25.13 [-26.72, -23.55]|32 / 836|1.426e-163|not adjusted|descriptive only|
|cicids2018_distrinet|cnn-s42|p75 vs unbounded|both|valid|-25.13 [-26.72, -23.55]|32 / 836|1.426e-163|5.706e-163|reject H0|
|cicids2018_distrinet|ft_transformer-s42|raw vs valid|p75|untargeted|0.00 [-0.12, 0.12]|0 / 0|1|not adjusted|descriptive only|
|cicids2018_distrinet|ft_transformer-s42|raw vs valid|unbounded|untargeted|0.00 [-0.13, 0.13]|0 / 0|1|not adjusted|descriptive only|
|cicids2018_distrinet|ft_transformer-s42|p75 vs unbounded|both|raw|-0.12 [-0.32, 0.02]|0 / 4|0.125|not adjusted|descriptive only|
|cicids2018_distrinet|ft_transformer-s42|p75 vs unbounded|both|valid|-0.12 [-0.32, 0.02]|0 / 4|0.125|0.125|do not reject H0|

Observed seed-cohort rates differ in 12/12 victim–budget conditions; the three rates supply descriptive variance, not a valid cross-seed paired hypothesis test. Seed-42 p75 versus unbounded Valid ASR differs after Holm in 5/6 victims. These tests compare budgets, not attack methods or independent row-selection effects. The existing FINAL Experiment A baseline tests remain applicable to the seed-42 cohort only.


## Cohort overlap and historical context

`audit.json` records exact pairwise sample-ID intersections and Jaccard overlap per dataset × victim × class. Overlap alone does not turn distinct seed cohorts into a fully paired repeated-measures panel.

|Dataset|Victim|Seed pair|Shared IDs (out of 3200)|Jaccard|
|---|---|---|---:|---:|
|cicids2017_distrinet|mlp|42/2024|705|0.1238|
|cicids2017_distrinet|mlp|42/2026|722|0.1272|
|cicids2017_distrinet|mlp|2024/2026|730|0.1287|
|cicids2017_distrinet|cnn|42/2024|708|0.1244|
|cicids2017_distrinet|cnn|42/2026|724|0.1276|
|cicids2017_distrinet|cnn|2024/2026|735|0.1297|
|cicids2017_distrinet|ft_transformer|42/2024|703|0.1234|
|cicids2017_distrinet|ft_transformer|42/2026|720|0.1268|
|cicids2017_distrinet|ft_transformer|2024/2026|728|0.1283|
|cicids2018_distrinet|mlp-s42|42/2024|129|0.0206|
|cicids2018_distrinet|mlp-s42|42/2026|146|0.0233|
|cicids2018_distrinet|mlp-s42|2024/2026|140|0.0224|
|cicids2018_distrinet|cnn-s42|42/2024|128|0.0204|
|cicids2018_distrinet|cnn-s42|42/2026|146|0.0233|
|cicids2018_distrinet|cnn-s42|2024/2026|139|0.0222|
|cicids2018_distrinet|ft_transformer-s42|42/2024|128|0.0204|
|cicids2018_distrinet|ft_transformer-s42|42/2026|146|0.0233|
|cicids2018_distrinet|ft_transformer-s42|2024/2026|137|0.0219|

The earlier `outputs/expA_primattack_full_pool/row_resampling.json` resamples rows from an already-attacked full pool at p75; it measures row-selection variation **conditional on fixed attack outcomes**, not independent attack-seed variability, and cannot replace these three fresh attack runs. Its historical valid p75 figures (canonical / resample mean / resample SD / 2.5–97.5%):

|Dataset|Victim|Canonical|Resample mean ± SD|Resample 95% range|
|---|---|---|---|---|
|cicids2017_distrinet|mlp|4.09%|4.03% ± 0.32%|3.41%–4.69%|
|cicids2017_distrinet|cnn|13.47%|13.51% ± 0.53%|12.50%–14.56%|
|cicids2017_distrinet|ft_transformer|0.12%|0.13% ± 0.04%|0.06%–0.22%|
|cicids2018_distrinet|mlp-s42|2.53%|2.15% ± 0.25%|1.66%–2.66%|
|cicids2018_distrinet|cnn-s42|1.16%|1.20% ± 0.19%|0.84%–1.59%|
|cicids2018_distrinet|ft_transformer-s42|0.00%|0.08% ± 0.05%|0.00%–0.19%|

All successes are offline feature-space proxies on held-out chronological-within-label test splits. No PCAP replay, packet-level realizability, new-campaign generalization, or unconditional seed robustness is established. Non-rejection by McNemar does not prove equivalence.
