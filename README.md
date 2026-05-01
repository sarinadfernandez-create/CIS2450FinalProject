# CIS 2450 Final Project
**Carly Googel & Sarina Fernandez-Grinshpun**

## What this is

A system that tries to predict which emerging tech topics expert newsletter editors are about to cover, before they actually cover them. We watch arXiv, Semantic Scholar, GitHub, Reddit, and Hacker News for signals, and we use newsletter appearance (TLDR AI, TLDR Tech, TLDR Fintech, TLDR Founders, Import AI, Bits in Bio, The Batch) as our ground-truth label. The idea is that newsletter editors are domain experts doing active curation, so if a topic makes it into their issue, it was worth paying attention to. We want to flag those topics one to two weeks earlier.

The business question we kept coming back to: out of all the noise being published this week across research, developer, and community platforms, which topics are actually building toward something, and which are just hype that will fade?

## Data sources

We pull from two categories. The newsletter side is our **labels** — seven scraped newsletters going back to 2023. If a topic shows up in any of them in week `t+1` or `t+2`, that topic-week is a positive example. The external signal side is our **features** — arXiv and Semantic Scholar for academic research (sampled at 20,000 rows each with a fixed seed for reproducibility), Hacker News for tech discussion, Reddit for community chatter, and GitHub for developer activity (all available rows pulled). All timestamps get normalized to ISO week format so we can join everything cleanly on a `(topic, week)` key.

## How the pipeline works

Everything flows through the same four-stage pipeline. Raw newsletter and external CSVs come in, KeyBERT pulls candidate phrases out of every document, the entity linker maps those phrases to a shared canonical topic vocabulary using sentence-transformer embeddings, and then the burst-score and historical features get computed on the per-topic-week level and fed into the classifier.

## EDA

The unit of analysis is a `(canonical_topic, week)` pair. Each pair has a small set of engineered features and one binary label: did this topic show up in a newsletter in the next one or two weeks?

The features that matter are `burst_score` (a z-score of this week's mentions against the topic's own trailing 8-week mean and std—captures acceleration, not raw volume), `mentions` (raw count of times the topic was mentioned across signal sources that week), `nl_rate_8wk` (fraction of the topic's last 8 active weeks that ended up in a newsletter), `nl_rate_all` (same thing but over the topic's full history), `weeks_since_nl` (how long it's been since this topic last got newsletter coverage, capped at 200), `is_novel` (a binary flag for topics that have literally never been in a newsletter), and `past_nl_count` (total newsletter appearances before this week).

The biggest thing EDA told us was that the dataset is brutally imbalanced — roughly 2-4% of topic-weeks are positives, which is roughly a 1:25 to 1:30 ratio. A model that always predicts "no" gets 96% accuracy, which is exactly why we throw out accuracy as a metric and lean on F1 and Precision@K instead. We caught this early in the Decision Tree experiments and it shaped every modeling decision after.

EDA also flagged that `nl_rate_8wk` and `nl_rate_all` are correlated, which pushed us toward L2 regularization for Logistic Regression. `burst_score` had near-zero Gini importance in the initial Decision Tree, which was surprising given it's basically the headline feature of the project. We followed up on that with a point-biserial correlation analysis and found it's more useful for novel topics than established ones, which actually makes sense — editors lean on track record for known topics, but need a velocity signal for ones they've never covered. `weeks_since_nl` came in negatively associated with the label, which lines up with the editorial-cooldown intuition that editors don't immediately re-cover something they just wrote about.

All EDA was done with a strict temporal lens. The dataset spans 2023 to present, features for week `t` only see data from weeks before `t`, and labels look forward (so a label of 1 means "in newsletter at `t+1` or `t+2`," never `t` itself, to avoid leakage).

## Pre-processing & feature engineering

Data collection is straightforward—APIs and scrapers feeding into `data/raw/`. The interesting preprocessing happens in the entity linker. Raw text from every source goes through three steps. First, KeyBERT (`all-MiniLM-L6-v2`) pulls candidate keyphrases out of each document with MMR diversity (`diversity=0.5`) so we don't get a bunch of near-identical phrases. Second, those phrases get embedded alongside our 100+ hand-curated seed topics, we compute pairwise cosine similarity, and each phrase gets assigned to its best seed topic only if similarity is at least 0.72. Third, anything that didn't map (sub-0.72 similarity, but with a KeyBERT confidence score above 0.60) goes into agglomerative clustering with cosine distance and average linkage. Clusters with at least 5 phrases generate new "discovered" canonical topics, with the most centroid-proximate phrase used as the label.

There are a few layers of noise filtering on top of that. We have a sponsor brand blocklist (50+ recurring sponsor names like Rippling, Vanta, Notion—they're useless tech topics from a forecasting perspective and they were polluting our feature matrix). We have junk regex patterns to strip newsletter metadata, dates, single-word tokens, and engagement boilerplate. Phrase length is capped at 2-5 words (single words too ambiguous, long phrases too specific to generalize). And we only keep phrases scoring above 0.50 from KeyBERT.

Null handling is mostly straightforward—null dates filtered out before feature construction, `fill_null(0)` after left-joins (a missing value just means the topic had no signal that week), `weeks_since_nl` capped at 200 to handle topics that have literally never been covered, and `burst_score` and `mentions` clipped at the 99th percentile to keep viral outliers from dominating. NaN/Inf embeddings get caught and dropped before clustering.

For imbalanced data, every model uses `class_weight='balanced'` (computed via `sklearn.utils.class_weight.compute_class_weight`). Without this, all three models collapse to predicting the majority class. Logistic Regression also gets `StandardScaler` (LR isn't scale-invariant; DT and RF are, so they get raw features). Scaling is fit on train only, then applied to val and test — no leakage.

The split is temporal. 70% train, 15% val, 15% test, by week cutoff, never shuffled. Features for week `t` only see weeks before `t`. The val set exists exclusively for hyperparameter tuning; the test set is touched once at the end.

## Modeling
We trained three models so we could compare interpretability, calibration, and ensemble performance side by side.

The **Decision Tree** (`model1_dt_rough.py`) is our interpretable baseline. The reason to start here is that you can directly read off which features and thresholds the model is splitting on, which is a sanity check on whether the model is learning anything meaningful. We tuned `max_depth` over `[2, 3, 4, 5, 6, 8, None]` using F1 on the validation set and 8 was best. `nl_rate_8wk` dominated the splits, which lined up with EDA, and `burst_score` showed nearly zero Gini importance, which kicked off the deeper investigation into when burst actually matters. The DT overfit hard at higher depths and made hard threshold decisions instead of probabilistic ones, which is part of why we moved to RF.

**Logistic Regression** (`model2_lr_rough.py`) gives us calibrated probabilities and signed coefficients. Since the practical output of this system is a per-week ranking of topics, calibration matters more than raw accuracy. We tuned `C` over `[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]` on val F1, and compared L1 vs L2—L1 zeroed out `burst_score` entirely and had worse F1, so we kept L2. The coefficients confirm what EDA suggested: `nl_rate_8wk` is the largest positive, `weeks_since_nl` is negative (editorial cooldown), and `is_novel` ended up small after the bug fix (novel topics aren't inherently more or less likely to be covered). We also evaluated calibration with `CalibrationDisplay.from_predictions`.

**Random Forest** (`model3_rf.py`) is our primary model. It addresses DT's overfitting by averaging across many trees and captures feature interactions that LR can't see without explicit interaction terms. We did a grid search over `n_estimators ∈ {100, 200, 300}` × `max_depth ∈ {4, 6, 8}` on val F1 with `n_jobs=-1` for parallelism. We use F1 as the primary tuning metric, but report ROC-AUC, AUPRC (more honest than AUC under imbalance), Precision@K and Recall@K (the practically relevant metrics, since editors pick a fixed number of topics per issue), the confusion matrix (interpreted in domain terms — false positives are hype editors ignored, false negatives are signal we missed), and a lead-time analysis (for topics we correctly flagged, how many weeks ahead did we catch them).

## The three "difficult" inclusions
These are the three rubric difficulty topics this project takes on, and where they live in the codebase.

**1. Ensemble Model.** Random Forest is our primary production model (`model3_rf.py`, also compared head-to-head in `modeling_rough.py`). It's an ensemble of `n_estimators` decision trees, each trained on a bootstrap sample with random feature subsets at each split, with predictions averaged across all trees. The reason ensembling specifically matters here is that a single Decision Tree was unstable—small perturbations in training data flipped its splits—and couldn't capture interactions between features (a topic with both high mentions *and* a long gap since last newsletter is a different signal than either feature alone). RF handles both via bagging and feature randomization, and its probability outputs are better calibrated than a single tree, which matters for the per-week Precision@K ranking. RF outperformed both DT and LR on AUC, AUPRC, and Precision@K, and the gap over LR specifically is evidence that those interaction effects matter—including the burst-score-by-novelty interaction we'd flagged earlier.

**2. NLP Entity Linking.** This is `src/entity-linking/`, primarily `entity-linker.py` plus the two topic extraction scripts. The pipeline is KeyBERT extraction, then MiniLM-L6-v2 sentence embeddings, then cosine similarity mapping at a 0.72 threshold against our 100+ seed topics, then agglomerative clustering on the leftovers to discover new topics. This is the most novel piece of the project. The whole thing is necessary because we're combining newsletters, arXiv abstracts, Reddit posts, GitHub repos, and HN stories, all of which talk about the same things in totally different vocabulary. "LLM" in an arXiv abstract should match "foundation model" in a newsletter, and without entity linking it just doesn't. We tuned the seed topic list and the similarity threshold pretty heavily—at thresholds below 0.72, sponsor copy started leaking into real topics (we had "electrolyte drink" mapping to biotech), which is why we kept it strict.

**3. Temporal Learning + Imbalance.** This is the modeling-side difficulty, and it's spread across all three model files plus `modeling_rough.py`. The class ratio is roughly 1:23, every feature is computed strictly causally with no lookahead (week `t` features see only weeks before `t`, labels look forward to `t+1` or `t+2`), and the train/val/test split is 70/15/15 by week with no shuffling at any point. This combination changes basically every modeling choice. We can't use accuracy because predicting all-zero hits 96%. We use `class_weight='balanced'` everywhere, fit `StandardScaler` on train only, tune hyperparameters only on val, and touch the test set exactly once. The forward-looking label was the single most important call here—when we'd previously labeled with same-week, burst score looked statistically significant (p=0.026), but that turned out to be leakage. With proper forward labels, p jumped to 0.54, which was honestly demoralizing for about a day until we realized the editorial-history features are doing the real predictive work.

## Course topics applied
Most of these come up multiple times across the pipeline—listing where they live and why they matter genearlly.

**Polars.** Every script (`entity-linker.py`, `modeling_rough.py`, etc.) uses Polars as the primary DataFrame library—loading, filtering, grouping, joining, CSV I/O. We lean on lazy filter chains (`df.filter(pl.col("score") >= 0.50).filter(pl.col("date") >= START_DATE)`), group-by aggregations on `(canonical_topic, week)`, multi-source joins, and schema overrides for messy CSVs. The signal data hits the millions of rows, so the columnar execution and lazy evaluation actually matter—pandas would be painful here.

**Text representations, embeddings, and LLMs.** Lives in `entity-linker.py` and the two topic-extraction scripts. KeyBERT (BERT-based phrase scoring against the document embedding) does the candidate extraction. SentenceTransformers (`all-MiniLM-L6-v2`) embeds keyphrases and seed topics into 384-dimensional dense vectors, and cosine similarity over those embeddings is what powers the entity linker. Everything is batch-encoded (`batch_size=256`, `normalize_embeddings=True`) for efficiency. Without these embeddings, "retrieval augmented generation" and "RAG pipeline" would never collapse onto the same canonical topic, and the whole multi-source aggregation idea falls apart.

**Unsupervised learning (clustering).** `cluster_unmapped()` in `entity-linker.py`. Agglomerative clustering with `metric="cosine"` and `linkage="average"`, using a `distance_threshold=0.20` rather than a fixed `n_clusters`. The reason we picked agglomerative over K-Means is that we don't know in advance how many novel topics show up in a given batch—we want clusters to stop merging when they get semantically heterogeneous, not when we hit some arbitrary K. Cluster labels are assigned as the phrase closest to the centroid. This is what lets us discover niche topics that weren't in our seed list (especially in biotech and fintech sub-domains).

**Supervised learning.** Across all three model files. Three classifiers—Decision Tree, Logistic Regression (L2, scaled), Random Forest (parallelized)—trained on the binary `label_next2wk`. Temporal split, validation-set tuning, held-out test evaluation. The whole prediction task is binary classification with a labeled history, so this is the right hammer.

**Time series.** Burst score, historical rate features, the temporal split, and the lead-time analysis are all in `modeling_rough.py`. Burst score is a rolling z-score of weekly mentions against a trailing 8-week window. `nl_rate_8wk` and `nl_rate_alltime` use expanding windows over only past data. The split itself is by week cutoff (not random)—train ends at week `t`, val runs `t` to `t+δ`, test runs `t+δ` onward. Treating this as i.i.d. classification would leak future newsletter appearances into features and make the metrics look way better than they actually are.

**Joins (multi-source integration).** In `modeling_rough.py`. The whole feature matrix is a series of left joins on `(canonical_topic, week)`—burst features joined to historical rate features, signal data grouped and joined across source types, newsletter weekly aggregates joined to construct forward labels, and `fill_null(0)` to handle topics with no signal in a given week. The whole project's premise is that signals from external sources predict newsletter coverage, and those two streams live in totally separate DataFrames. The join is literally the system.

**Hyperparameter tuning.** All model files. We did manual grid search on the validation set across all three models—DT over 7 `max_depth` candidates, LR over 7 `C` values plus an L1-vs-L2 comparison, RF over a 3×3 grid of `n_estimators` × `max_depth`. Selection metric is val F1. The test set is never used for tuning. We chose F1 over accuracy because at a 3% positive rate, accuracy is meaningless—F1 balances precision and recall, which is what actually matters for a ranking system.

## Results & conclusions
The headline finding is that `nl_rate_8wk` is the dominant predictor across all three models. Topics that have been in newsletters recently are the best candidates for future coverage—editorial interest is genuinely sticky. `weeks_since_nl` being strongly negative confirms the editorial cooldown effect, where editors deprioritize topics they've recently covered.

The burst score story is more interesting than we initially thought. It has low Gini importance in tree models on its own, but the point-biserial analysis shows it's actually predictive specifically for novel topics (`is_novel = 1`), much less so for established ones. The RF model picks up on this through interaction effects—burst score ends up with 22% feature importance in RF despite a near-zero standalone correlation, because the model is using it in combination with editorial history. The story is roughly: editors trust track record for established topics, but for novel ones, they need an external velocity signal to even notice them. This validates the original design intuition, just in a more nuanced way than we expected.

The novel-vs-established split shows meaningfully different AUC between the two subsets, which suggests they're really separate prediction problems. Future work could split into two models or build explicit interaction terms.

Lead time turned out to be the most satisfying metric. At an `rf_prob ≥ 0.3` threshold, the model flagged topics on average 9 weeks before they actually appeared in a newsletter, with a max of 20 weeks. That's the value proposition working—we're not just catching contemporaneous signal, we're getting genuine lead time on editorial coverage.

The entity linking quality piece is worth flagging too. Going from ~50 to 100+ seed topics meaningfully improved newsletter mapping coverage, and the agglomerative clustering step pulled in real niche topics (especially in biotech and fintech) that the seed list would have missed. Discovered topics ended up contributing non-trivially to the final feature matrix.

### Limitations
The label is coarse. "Appeared in a newsletter in the next two weeks" lumps together coverage that happened for very different reasons—actual editorial interest, but also sponsorships, calendar-driven slots, and serendipity. There's a ceiling on what we can predict.

Some canonical topics ended up too broad ("AI coding assistant tool" aggregates a lot of heterogeneous stuff together) even after our filtering. A more aggressive filtering pass or a finer-grained seed list would probably help.

The system performs best on topics with historical newsletter coverage, which means truly novel breakthroughs—the actual black swans we'd most want to catch—are the hardest case. This is partly fundamental (you can't predict editorial coverage of a topic that has zero history) and partly something more data could fix. I think increasing our topic data to industry specific blogs and a more diverse array of source types would solve this and lead to better coverage of signal sources.

## Dependencies

Install with:
```bash
pip install polars scikit-learn sentence-transformers keybert numpy pandas matplotlib seaborn scipy requests beautifulsoup4 streamlit plotly
```

## How to run
Assuming you're at the repo root (`CIS2450FinalProject/`)
### Quickstart (just look at the results)
```bash
# 1. Open the modeling notebook to see all three models, EDA, and final metrics
jupyter notebook src/models/modeling_final_clean.ipynb

# 2. Or open the EDA notebook
jupyter notebook eda_final.ipynb

# 3. Or launch the dashboard
streamlit run src/dashboard.py
```

