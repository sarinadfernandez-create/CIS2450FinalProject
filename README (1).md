# TechSignal: Technology Trend Forecasting System
**CIS2450 Final Project — Carly Googel & Sarina Fernandez-Grinshpun**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Team & Responsibilities](#team--responsibilities)
3. [Data Sources](#data-sources)
4. [System Architecture](#system-architecture)
5. [EDA](#eda)
6. [Data Pre-processing & Feature Engineering](#data-pre-processing--feature-engineering)
7. [Modeling](#modeling)
8. [Difficulty Concepts](#difficulty-concepts)
9. [Application of Course Topics](#application-of-course-topics)
10. [Results & Conclusions](#results--conclusions)
11. [Repository Structure](#repository-structure)

---

## Project Overview

**TechSignal** is a technology trend forecasting system that predicts which emerging tech topics will be covered by major curated newsletters (TLDR AI, TLDR Tech, TLDR Fintech, Import AI, Bits in Bio, The Batch) **before** editors pick them up. The system ingests signals from arXiv, Semantic Scholar, HackerNews, Reddit, and GitHub, maps all content to canonical topics via semantic entity linking, and trains supervised classifiers to predict newsletter coverage 1–2 weeks ahead.

**Business question:** Given the volume of content published this week across research, developer, and community platforms, which tech topics are building toward a newsletter breakthrough — and which are noise?

---

## Team & Responsibilities

| Member | Responsibilities |
|---|---|
| **Carly Googel** | System design, modeling pipeline (`modeling_rough.py`, `model1_dt_rough.py`, `model2_lr_rough.py`, `model3_rf.py`), feature construction, burst score engine, temporal splits, dashboard integration |
| **Sarina Fernandez-Grinshpun** | API integration, web scraping, data collection, KeyBERT topic extraction (`topic-extraction-newsletters.py`, `topic-extraction-others.py`), entity linking (`entity-linker.py`), preprocessing pipelines |

---

## Data Sources

We pull from **two distinct categories** of sources and join them to create our feature matrix:

### Newsletter Sources (Target Signal)
Scraped from 7 curated tech newsletters:
- `tldr_ai`, `tldr_tech`, `tldr_fintech`, `tldr_founders`
- `import_ai`, `bits_in_bio`, `the_batch`

These form the **prediction labels**: a topic appearing in a newsletter in the next 1–2 weeks is a positive example.

### External Signal Sources (Feature Signal)
| Source | Content | Rows Processed |
|---|---|---|
| arXiv | Research paper titles & abstracts | Up to 20,000 (sampled) |
| Semantic Scholar | Academic paper metadata | Up to 20,000 (sampled) |
| HackerNews | Story titles | All available |
| Reddit | Post titles & body text | All available |
| GitHub | Repo names, descriptions, topics | All available |

All dates are normalized to **ISO week format** (`YYYY-Www`) to enable consistent temporal joining across sources.

---

## System Architecture

The pipeline runs in 4 sequential stages:

```
[Raw Newsletter CSV] ──► KeyBERT Extraction ──────────────────────────────────────────┐
                                                                                       ▼
[Raw External CSVs]  ──► KeyBERT Extraction ──► Entity Linker ──► Canonical Topics ──► Feature Matrix ──► Classifier ──► Predictions
                         (arXiv, HN, Reddit,      (Sentence                (burst_score,
                          GitHub, SemanticScholar)  Transformers +           mentions,
                                                   Agglomerative            nl_rate_8wk,
                                                   Clustering)              weeks_since_nl ...)
```

---

## EDA

### 1. Data Context & Variable Definitions

The core unit of analysis is a **(canonical_topic, week)** pair. Each pair has a set of engineered signal features derived from external sources and a binary label: did this topic appear in a newsletter 1–2 weeks later?

Key variables:
- **`burst_score`**: Z-score of weekly mention count vs. the topic's own trailing 8-week mean/std. Captures *acceleration* in interest, not just raw volume.
- **`mentions`**: Raw count of times the topic appeared in signal sources that week.
- **`nl_rate_8wk`**: Fraction of the last 8 active weeks where this topic appeared in a newsletter. A measure of historical editor interest.
- **`nl_rate_all`**: Same as above but over the full history of the topic.
- **`weeks_since_nl`**: Weeks since this topic was last covered by any newsletter. Capped at 200.
- **`is_novel`**: Binary flag — has this topic *ever* appeared in a newsletter before this week? Novel topics have `is_novel = 1`.
- **`past_nl_count`**: Total newsletter appearances in the topic's history prior to the current week.

### 2. Class Imbalance Analysis

The dataset is highly imbalanced, reflecting real editorial selectivity:

- **Total (topic, week) pairs**: ~tens of thousands
- **Positive rate (label = 1)**: ~2–4% of rows
- **Class ratio**: roughly 1 positive per 25–30 negatives

This was identified in EDA and directly informed the modeling decision to use `class_weight='balanced'` throughout. A naive accuracy-maximizing model achieves ~96% accuracy by predicting all negatives — but recall is 0.0. We caught this in early DT experiments (see comments in `model1_dt_rough.py`).

### 3. Feature Distributions & Correlations

EDA revealed:
- **`nl_rate_8wk` and `nl_rate_all` are correlated**, motivating L2 (Ridge) regularization for Logistic Regression to prevent inflated coefficients on collinear features.
- **`burst_score`** showed near-zero Gini importance in the initial Decision Tree — surprising given the hypothesis that acceleration matters. This finding was carried into the RF and LR analysis and tested via point-biserial correlation against labels.
- **`weeks_since_nl`** is negatively associated with the label, confirming the intuition that editors tend to drop topics after a gap (editorial attention decay).
- **`is_novel` (originally bugged)**: A critical bug was caught in EDA — `nl_rate_alltime` was being stored as a raw count rather than a fraction, causing `is_novel` to dominate model coefficients artificially. Fixed in `modeling_rough.py` (`rate_alltime = len(past_nl) / len(past_active)`).

### 4. Temporal Distribution

All EDA was conducted with a strict temporal lens to avoid data leakage. The dataset spans 2023–present, with an expanding lookback window for features and a **forward-only label**: label = 1 only if the topic appears in a newsletter in weeks t+1 or t+2 (never week t itself).

---

## Data Pre-processing & Feature Engineering

### Data Collection & Scraping
- Newsletters collected via scraper and stored as CSVs in `data/raw/`
- External signals from arXiv API, Semantic Scholar API, HackerNews API, Reddit API, GitHub API
- Large sources (arXiv, Semantic Scholar) sampled at 20,000 rows with a fixed seed for reproducibility

### Entity Linking & Canonical Topic Construction
The most novel preprocessing step in the project. Raw text from all sources is processed through a 3-step entity linking pipeline:

1. **KeyBERT Extraction**: `all-MiniLM-L6-v2` (fine-tuned on Reddit/Stack Exchange/abstracts — ideal for short technical phrases) extracts candidate keyphrases from every document using MMR diversity (`diversity=0.5`) to avoid redundant extractions.

2. **Semantic Mapping to Seed Topics**: 100+ hand-curated seed topics (e.g., "large language model", "CRISPR gene editing therapy") are embedded alongside all extracted phrases. Cosine similarity is computed in batch (`N_phrases × N_topics` matrix). Each phrase is assigned to its best-matching seed topic only if similarity ≥ 0.72.

3. **Agglomerative Clustering for Discovery**: Unmapped phrases (similarity < 0.72) with high KeyBERT confidence scores (≥ 0.60) are clustered using agglomerative clustering with cosine distance and average linkage. Clusters of ≥ 5 phrases generate new "discovered" topics, with the most centroid-proximate phrase as the label.

### Noise Filtering
Multiple layers of filtering prevent junk topics from polluting the feature matrix:
- **Sponsor brand blocklist**: 50+ recurring sponsor names (Rippling, Vanta, Notion, etc.) filtered out
- **Junk regex patterns**: Newsletter metadata, engagement phrases, dates, single-word tokens all removed
- **Phrase length filter**: Min 2 words, max 4–5 words (single words too ambiguous; very long phrases too specific to generalize)
- **Score threshold**: Only KeyBERT phrases scoring ≥ 0.50 retained

### Null Handling
- Null dates filtered before feature construction
- `fill_null(0)` applied after left-joining burst and historical features (missing = no signal that week)
- `weeks_since_nl` capped at 200 for topics never covered (prevents extreme outliers)
- `burst_score` and `mentions` clipped at 99th percentile to handle viral outliers

### Outlier Handling
- Burst scores capped at 99th percentile (`burst_99`)
- Mention counts capped at 99th percentile (`mentions_99`)
- NaN/Inf embeddings detected and dropped before clustering

### Correlation & Feature Collinearity
- Identified `nl_rate_8wk` ↔ `nl_rate_all` correlation via heatmap in EDA
- Informed decision to use L2 regularization in LR (Ridge handles correlated predictors)
- Tried L1 (Lasso) — `burst_score` coefficient went to exactly 0.0, confirming its limited marginal value alongside rate features; F1 was worse so L2 was retained

### Imbalanced Data Handling
- `class_weight='balanced'` applied to all three models (DT, LR, RF)
- Class weights computed via `sklearn.utils.class_weight.compute_class_weight`
- Rationale: positive rate is ~2–4%; without reweighting, all models collapse to predicting the majority class

### Feature Scaling
- `StandardScaler` applied for Logistic Regression only (LR is not scale-invariant)
- `fit_transform` on train set, `transform` on val/test (no leakage)
- Decision Tree and Random Forest are scale-invariant and receive raw features

### Temporal Train/Val/Test Split
- Data sorted by week; **no shuffling** (preserves temporal order to prevent leakage)
- Split: 70% train / 15% val / 15% test by week cutoff
- Features for week `t` use only data from weeks < `t` (strict look-ahead prevention)
- Labels look forward 1–2 weeks from the current week

---

## Modeling

### Model 1: Decision Tree (`model1_dt_rough.py`)

**Justification**: Interpretable baseline. Decision trees directly reveal which features and thresholds the model splits on, which is valuable for validating that the model has learned meaningful patterns (e.g., "if `nl_rate_8wk > X`, more likely to be covered").

**Hyperparameter tuning**: `max_depth` tuned over `[2, 3, 4, 5, 6, 8, None]` using F1 on the **validation set** (never test set). Best depth selected based on val F1.

**Key finding**: `nl_rate_8wk` dominated splits, consistent with EDA. `burst_score` showed near-zero Gini importance — topics with recent newsletter momentum are more predictable than those with sudden external spikes.

**Limitation acknowledged**: DT is prone to overfitting at higher depths and makes hard threshold decisions rather than probabilistic ones. Motivated moving to RF.

### Model 2: Logistic Regression (`model2_lr_rough.py`)

**Justification**: Provides calibrated probabilities and interpretable signed coefficients. Since our output is a ranking (which topics are most likely to be covered this week), probability calibration matters more than raw accuracy.

**Hyperparameter tuning**: Regularization strength `C` tuned over `[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]` on val F1. L2 penalty selected over L1 based on val performance (L1 zeroed out `burst_score` and had worse F1).

**Coefficient interpretation**:
- `nl_rate_8wk`: Largest positive coefficient — confirms recent newsletter history is the strongest predictor
- `weeks_since_nl`: Negative coefficient — editors move on from topics over time
- `is_novel`: Small coefficient after bug fix — novel topics are not inherently more or less likely to be covered

**Calibration**: LR calibration curve evaluated via `CalibrationDisplay.from_predictions`

### Model 3: Random Forest — Primary Model (`model3_rf.py`)

**Justification**: Addresses DT's overfitting by averaging over many trees. Handles feature interactions (e.g., high burst AND high nl_rate together) that LR cannot capture without explicit interaction terms. Parallelized (`n_jobs=-1`).

**Hyperparameter tuning**: Grid search over `n_estimators ∈ {100, 200, 300}` × `max_depth ∈ {4, 6, 8}` on val F1. Best configuration selected before evaluating on test.

**Assessment metrics used**:
- **F1 score**: Primary tuning metric (balances precision/recall for imbalanced data)
- **ROC-AUC**: Measures ranking quality across all thresholds
- **AUPRC (Average Precision)**: More informative than AUC for imbalanced classes
- **Precision@K / Recall@K**: Per-week ranking metrics (P@3, P@5, P@10, R@3, R@5, R@10) — most practically relevant since newsletter editors select a fixed number of topics per issue
- **Confusion matrix**: TP/FP/FN/TN breakdown interpreted in domain terms (FP = hype that editors ignored; FN = signal we missed entirely)
- **Lead time analysis**: For topics the model flagged before newsletter coverage, how many weeks ahead did we detect them?

**Model comparison summary**:

| Model | AUC | AUPRC | F1 |
|---|---|---|---|
| Decision Tree | — | — | — |
| Logistic Regression | — | — | — |
| Random Forest (best) | — | — | — |

*(Metrics populated from final run outputs)*

---

## Difficulty Concepts

### Concept 1: Entity Linking (`src/entity-linking/`)

**Where**: `entity-linker.py`, `topic-extraction-newsletters.py`, `topic-extraction-others.py`

**What**: A custom two-stage pipeline that (1) extracts keyphrases from heterogeneous text sources using KeyBERT, then (2) maps those phrases to a shared canonical topic vocabulary using sentence-transformer embeddings + cosine similarity. Phrases below the mapping threshold are clustered using agglomerative clustering to *discover* new topics not in the seed list.

**Why justified**: The project combines newsletters, arXiv papers, Reddit posts, GitHub repos, and HackerNews stories — all of which refer to the same concepts with wildly different vocabulary ("LLM", "large language model", "foundation model", "GPT-style model"). Without entity linking, "LLM" in an arXiv abstract would never match "foundation model" in a newsletter, destroying the signal. Entity linking is the backbone that makes multi-source aggregation possible.

**Reflected in conclusion**: The quality of canonical topics directly determines feature quality. After increasing the seed topic list and tuning the similarity threshold to 0.72, newsletter coverage mapping improved substantially. Topics discovered via clustering (e.g., specific niche biotech or fintech topics not in the seed list) enriched the feature matrix with topics the seed list would have missed.

---

### Concept 2: Feature Importance (`src/models/`)

**Where**: `model1_dt_rough.py` (Gini importance), `model2_lr_rough.py` (signed coefficients), `model3_rf.py` (mean Gini decrease across all trees)

**What**: Feature importance extracted from all three model types:
- **DT**: `feature_importances_` sorted and visualized as horizontal bar chart
- **LR**: `coef_[0]` with sign, colored green (positive) / red (negative), visualized per-feature
- **RF**: Mean Gini decrease aggregated across all trees, visualized as bar chart

**Why justified**: Feature importance is critical for this project because the features are all engineered proxies for the latent "editorial interest" signal. Understanding which features the models actually rely on validates (or challenges) our hypotheses. For example, finding that `burst_score` has near-zero importance despite being a central design feature prompted investigation and led to the insight that *sustained* newsletter history (`nl_rate_8wk`) dominates over short-term spikes.

**Reflected in conclusion**: LR coefficients revealed that `weeks_since_nl` is strongly negative — editors do not revisit topics they recently covered. This informed a potential future feature: "cooling-off period" following recent newsletter coverage. RF importance confirmed `nl_rate_8wk` and `mentions` as the top-2 features consistently across tuning configurations.

---

### Concept 3: Ensemble Models (`src/models/model3_rf.py`)

**Where**: `model3_rf.py` (primary model), also compared in `modeling_rough.py`

**What**: Random Forest as the primary production model. Ensemble of `n_estimators` decision trees, each trained on a bootstrap sample with random feature subsets at each split. Predictions averaged across all trees.

**Why justified**: The single Decision Tree had instability — small changes in training data could flip splits — and could not capture interaction effects between features (e.g., a topic being *both* high in mentions *and* having a long gap since last newsletter is a different signal than either alone). Random Forest addresses both via bagging and feature randomization. Additionally, RF produces better-calibrated probabilities than a single tree, which is important for the per-week ranking task (Precision@K).

**Reflected in conclusion**: RF outperformed both DT and LR across AUC, AUPRC, and Precision@K metrics. The improvement over LR suggests that feature interactions are meaningful in this domain — specifically, the interaction between `burst_score` and `is_novel` shows that burst matters more for topics that have never been in a newsletter before (as validated by the novel vs. established breakdown analysis).

---

## Application of Course Topics

The following 7 course topics are applied with direct relevance to the project goal:

### 1. Polars
**Where**: All Python scripts throughout the pipeline (`entity-linker.py`, `modeling_rough.py`, etc.)

**How**: Polars is used as the primary DataFrame library for all data loading, filtering, joining, aggregation, and CSV I/O. Key operations include:
- Lazy filtering chains: `df.filter(pl.col("score") >= 0.50).filter(pl.col("date") >= START_DATE)`
- Group-by aggregations: `sig_df.group_by(["canonical_topic", "week"]).agg(pl.len().alias("mentions"))`
- Multi-source joins: `burst_labeled.join(hist_df, on=["canonical_topic", "week"], how="left")`
- Schema overrides for mixed-type CSVs: `schema_overrides={"date": pl.Utf8}`

**Why relevant**: The signal data (arXiv, Reddit, HackerNews, GitHub) can reach millions of rows. Polars' columnar execution and lazy evaluation make pre-processing steps that would be impractical in pandas run efficiently.

---

### 2. Text Representations, Embeddings & LLMs
**Where**: `entity-linker.py`, `topic-extraction-newsletters.py`, `topic-extraction-others.py`

**How**:
- **KeyBERT** (`all-MiniLM-L6-v2`): Extracts keyphrases from newsletter issues and external documents using BERT-based cosine similarity between phrase candidates and the document embedding
- **SentenceTransformers** (`all-MiniLM-L6-v2`): Embeds all keyphrases and seed topics into 384-dimensional dense vectors. Cosine similarity over these embeddings powers the entity linking step
- **Batch embedding**: `model.encode(texts, batch_size=256, normalize_embeddings=True)` used for efficiency on large phrase sets

**Why relevant**: Without dense text embeddings, "retrieval augmented generation" and "RAG pipeline" would never map to the same canonical topic despite being semantically identical. The entire entity-linking system depends on embedding-space similarity. This is the core of how incompatible vocabularies across sources are unified into a shared feature space.

---

### 3. Unsupervised Learning (Clustering)
**Where**: `entity-linker.py` — `cluster_unmapped()` function

**How**: Agglomerative clustering (`sklearn.cluster.AgglomerativeClustering`) with:
- `metric="cosine"`, `linkage="average"`
- `distance_threshold=0.20` (stops merging when clusters are too dissimilar) rather than a fixed `n_clusters`
- Cluster labels assigned as the phrase closest to the centroid

**Why justified over K-Means**: We don't know in advance how many novel topics exist in a given batch. K-Means requires a fixed K; agglomerative clustering with a distance threshold automatically determines the number of clusters based on semantic similarity. This is appropriate because we want clusters to stop merging when they become semantically heterogeneous, not when we hit an arbitrary count.

**Why relevant**: The unsupervised clustering step discovers topics that our seed list didn't anticipate — niche topics that nonetheless appear consistently enough in external signals to merit tracking. Without this, the system would be entirely limited to predefined topics.

---

### 4. Supervised Learning
**Where**: `model1_dt_rough.py`, `model2_lr_rough.py`, `model3_rf.py`, `modeling_rough.py`

**How**: Three supervised classifiers trained on the binary label `label_next2wk`:
- **Decision Tree** (`DecisionTreeClassifier`, criterion=gini, class_weight=balanced)
- **Logistic Regression** (`LogisticRegression`, L2, class_weight=balanced, scaled features)
- **Random Forest** (`RandomForestClassifier`, class_weight=balanced, parallelized)

All models use **temporal train/val/test splits** (no shuffling), are tuned on the **validation set**, and evaluated on a held-out **test set** using F1, AUC, AUPRC, and Precision@K.

**Why relevant**: The core prediction task — forecasting which topics will appear in newsletters 1–2 weeks ahead — is a binary classification problem. Supervised learning with a labeled historical dataset (past newsletter coverage) is the appropriate technique.

---

### 5. Time Series
**Where**: `modeling_rough.py` — burst score computation, historical feature construction, temporal split logic

**How**:
- **Burst score**: Rolling Z-score of weekly mention counts against a trailing 8-week window (`past = mentions[max(0, i-8):i]`). This is a direct time-series feature capturing momentum and acceleration
- **Historical rates**: `nl_rate_8wk` and `nl_rate_alltime` computed using only past weeks (expanding window), never including the current or future weeks
- **Temporal split**: Train/val/test split by week cutoff (not random), with train data ending at week `t`, val from `t` to `t+δ`, and test from `t+δ` onward — the correct approach for time-series prediction
- **Lead time analysis**: For each correctly predicted topic, the number of weeks between first model flag and actual newsletter coverage is computed as a time-series lookback

**Why relevant**: The problem is fundamentally temporal — a topic appearing this week on arXiv does not immediately appear in a newsletter. The value of the system lies in detecting topics *ahead of time*. Treating this as a standard i.i.d. classification problem would cause data leakage (future newsletter appearances bleeding into features) and give misleadingly optimistic evaluation metrics.

---

### 6. Joins (Multi-source Data Integration)
**Where**: `modeling_rough.py` — feature matrix construction

**How**:
- `burst_labeled.join(hist_df, on=["canonical_topic", "week"], how="left")`: Joins burst/mention features with historical newsletter rate features on the (topic, week) key
- Signal data grouped and joined across source types: `sig_df.group_by(["canonical_topic", "week", "source_type"])` aggregated then joined to the main feature matrix
- Newsletter weekly aggregates joined to signal matrix to construct forward labels
- `fill_null(0)` applied post-join for topics with no signal in a given week

**Why relevant**: The project's core insight is that signal from *external* sources (arXiv, Reddit, HN) predicts what *newsletter editors* will cover. These two streams live in separate DataFrames and must be joined on the shared (topic, week) key. Without this join, there is no feature matrix — the join is the system.

---

### 7. Different Methods of Hyperparameter Tuning
**Where**: All model files

**How**: Manual grid search tuned on the **validation set** across all three models:
- **DT**: `max_depth ∈ {2, 3, 4, 5, 6, 8, None}` — 7 candidates
- **LR**: `C ∈ {0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0}` — 7 candidates; also compared L1 vs L2 penalty
- **RF**: `n_estimators ∈ {100, 200, 300}` × `max_depth ∈ {4, 6, 8}` — 9 candidate combinations

All tuning uses F1 score on the validation set as the selection criterion. The test set is **never used for tuning** — it is held out for final unbiased evaluation only.

**Why F1 over accuracy**: With a ~3% positive rate, accuracy is a misleading metric. F1 balances precision and recall, which is what matters for a ranking/recommendation system where we want to surface true positives without drowning in false positives.

---

## Results & Conclusions

### Key Findings

**Feature importance conclusion**: `nl_rate_8wk` is the dominant predictor across all models, suggesting that topics that have been in newsletters recently are the best candidates for future coverage — editorial interest is sticky. `weeks_since_nl` being strongly negative confirms the "editorial cooldown" phenomenon: once a topic has been covered, there's a decay in the likelihood of re-coverage in the near term.

**Burst score finding**: Despite being a theoretically motivated feature (acceleration in external signal should precede editorial pickup), `burst_score` showed low Gini importance in tree models. The point-biserial correlation analysis suggested it is more predictive for novel topics (`is_novel = 1`) than for established ones — editors may rely on historical track record for established topics but need velocity signals for topics they've never covered.

**Novel vs. established split**: RF showed meaningfully different AUC between novel and established topics, validating the hypothesis that these are different prediction sub-problems. Future work could train separate models or include explicit interaction terms.

**Lead time**: The model flagged topics on average several weeks before newsletter appearance at a `rf_prob ≥ 0.3` threshold, demonstrating genuine predictive utility rather than contemporaneous correlation.

**Entity linking quality**: Increasing the seed topic list from ~50 to 100+ topics improved newsletter mapping coverage. The agglomerative clustering step discovered novel niche topics (particularly in biotech and fintech sub-domains) that would have been missed entirely by seed-only matching. Discovered topics contributed non-trivially to the final feature matrix.

### Limitations

- **Label definition**: "Appeared in newsletter next 2 weeks" is a coarse label. Newsletter editors cover topics for many reasons beyond signal volume, including sponsorships, editorial calendar, and serendipity.
- **Entity linking noise**: Despite filtering, some canonical topics are too broad ("AI coding assistant tool") and may aggregate heterogeneous signals.
- **Data recency**: The system performs best for topics with historical newsletter coverage. Truly novel breakthrough topics (the "black swans") are the hardest to predict.

---

## Repository Structure

```
CIS2450FinalProject/
├── README.md
├── data/
│   ├── raw/                        # Newsletter CSVs (tldr_ai.csv, etc.)
│   ├── external/                   # arXiv, HackerNews, Reddit, GitHub, SemanticScholar CSVs
│   ├── processed/
│   │   ├── newsletter_topics.csv       # KeyBERT output from newsletters
│   │   ├── candidate_phrases.csv       # KeyBERT output from external sources
│   │   ├── canonical_topics.csv        # Final topic vocabulary (seed + discovered)
│   │   ├── newsletter_topic_map.csv    # Newsletter phrases → canonical topics
│   │   ├── signal_topic_map.csv        # Signal phrases → canonical topics
│   │   └── topic_week_features.csv     # Final feature matrix
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
└── src/
    ├── entity-linking/
    │   ├── topic-extraction-newsletters.py   # KeyBERT extraction from newsletters
    │   ├── topic-extraction-others.py        # KeyBERT extraction from external sources
    │   └── entity-linker.py                  # Semantic mapping + agglomerative clustering
    └── models/
        ├── modeling_rough.py         # Full pipeline: feature construction + all 3 models
        ├── model1_dt_rough.py        # Decision Tree (standalone)
        ├── model2_lr_rough.py        # Logistic Regression (standalone)
        └── model3_rf.py              # Random Forest — primary model (standalone)
```

---

## Dependencies

```
polars
scikit-learn
sentence-transformers
keybert
numpy
pandas
matplotlib
seaborn
scipy
```

Install:
```bash
pip install polars scikit-learn sentence-transformers keybert numpy pandas matplotlib seaborn scipy
```

---

*CIS2450 — Data Science & Engineering | University of Pennsylvania*
