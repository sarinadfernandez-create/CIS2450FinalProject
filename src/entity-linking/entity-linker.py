import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import polars as pl
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

#path setup
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

NEWSLETTER_IN = os.path.join(PROCESSED_DIR, "newsletter_topics.csv")
SIGNAL_IN = os.path.join(ROOT, "src", "data", "processed", "candidate_phrases.csv")
TOPICS_OUT = os.path.join(PROCESSED_DIR, "canonical_topics.csv")
NEWSLETTER_MAP_OUT = os.path.join(PROCESSED_DIR, "newsletter_topic_map.csv")
SIGNAL_MAP_OUT = os.path.join(PROCESSED_DIR, "signal_topic_map.csv")

#configs
MIN_SCORE_NEWSLETTER = 0.50
MIN_SCORE_SIGNAL = 0.45
START_DATE = "2023-01-01"
SIMILARITY_THRESHOLD = 0.55
CLUSTER_DISTANCE = 0.35
MIN_CLUSTER_SIZE = 3
SIGNAL_BATCH_SIZE = 50000

SEED_TOPICS = [
    #AI/ML
    "large language model",
    "machine learning",
    "natural language processing",
    "diffusion model",
    "reinforcement learning",
    "transformer architecture",
    "retrieval augmented generation",
    "fine tuning",
    "prompt engineering",
    "computer vision",
    "speech recognition",
    "multimodal model",
    "mixture of experts",
    "speculative decoding",
    "quantization",
    "model alignment",
    "AI safety",
    "AI agents",
    "autonomous agents",
    "vector database",
    "model distillation",
    "federated learning",
    #dev/infra
    "developer tools",
    "WebAssembly",
    "Kubernetes",
    "serverless computing",
    "open source",
    "API development",
    "cloud computing",
    "edge computing",
    #fintech
    "fintech",
    "algorithmic trading",
    "quantitative finance",
    "decentralized finance",
    "stablecoin",
    "payments",
    "blockchain",
    "cryptocurrency",
    #Biotech
    "CRISPR",
    "gene therapy",
    "protein folding",
    "genomics",
    "drug discovery",
    "biotech",
    "synthetic biology",
    "GLP-1",
    "mRNA",
    "cancer immunotherapy",
    "cell therapy",
    "antibody drug conjugate",
    #Startups
    "venture capital",
    "startup funding",
    "Y Combinator",
    "IPO",
    "acquisition",
    "seed round",
    #General
    "robotics",
    "quantum computing",
    "semiconductor",
    "GPU",
    "autonomous vehicle",
    "augmented reality",
    "cybersecurity",
    "software engineering",
]

def week_of(date_str: str) -> str:
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        iso = d.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    except Exception:
        return ""
    
def load_newsletter_phrases() -> pl.DataFrame:
    df = pl.read_csv(NEWSLETTER_IN)
    df = df.filter(pl.col("score") >= MIN_SCORE_NEWSLETTER)
    df = df.filter(pl.col("date") >= START_DATE)
    df = df.filter(pl.col("date").is_not_null())
    df = df.filter(pl.col("phrase").str.len_chars() > 3)
    df = df.filter(~pl.col("phrase").str.contains("2023|2024|2025|2026"))
    df = df.filter(~pl.col("phrase").str.contains("tldr|import ai|bits bio|the batch"))
    df = df.filter(~pl.col("phrase").str.contains("subscribe|click here|read more|this week|last week"))
    df = df.filter(pl.col("phrase").str.split(" ").list.len() <= 4)
    print(f"  Loaded {len(df)} newsletter phrases after filtering")
    return df

def load_signal_phrases() -> pl.DataFrame:
    if not os.path.exists(SIGNAL_IN):
        print(f"  Signal phrases not found at {SIGNAL_IN}")
        return pl.DataFrame()
    df = pl.read_csv(SIGNAL_IN, infer_schema_length=10000, schema_overrides={"date": pl.Utf8})
    df = df.filter(pl.col("score") >= MIN_SCORE_SIGNAL)
    df = df.filter(pl.col("date") >= START_DATE)
    df = df.filter(pl.col("date").is_not_null())
    df = df.filter(pl.col("phrase").str.len_chars() > 3)
    df = df.filter(pl.col("phrase").str.split(" ").list.len() <= 4)
    df = df.unique(subset=["phrase", "source", "week"])
    print(f"  Loaded {len(df)} signal phrases after filtering")
    return df


#https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 this model was actually trained on reddit comments, abstracts, and stack exchange and is good at short phrases, so should work well for our use case
def embed(model: SentenceTransformer, texts: list[str], batch_size: int = 256) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

def map_to_topics(
    phrase_embeddings: np.ndarray,
    topic_embeddings: np.ndarray,
    topics: list[str],
) -> tuple[list[str | None], list[float]]:
    sims = cosine_similarity(phrase_embeddings, topic_embeddings)
    best_idx = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)
    mapped = [topics[i] if s >= SIMILARITY_THRESHOLD else None
              for i, s in zip(best_idx, best_sim)]
    return mapped, best_sim.tolist()


def cluster_unmapped(
    phrase_embeddings: np.ndarray,
    unmapped_indices: list[int],
    phrases: list[str],
) -> dict[int, str]:
    if len(unmapped_indices) < MIN_CLUSTER_SIZE:
        return {}

    unmapped_embs = phrase_embeddings[unmapped_indices]
    valid_mask = np.isfinite(unmapped_embs).all(axis=1)
    if not valid_mask.all():
        print(f"  Dropping {(~valid_mask).sum()} NaN/inf embeddings")
        unmapped_embs = unmapped_embs[valid_mask]
        unmapped_indices = [unmapped_indices[i] for i, v in enumerate(valid_mask) if v]

    if len(unmapped_indices) < MIN_CLUSTER_SIZE:
        return {}

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=CLUSTER_DISTANCE,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(unmapped_embs)

    cluster_map = {}
    for cluster_id in set(labels):
        mask = labels == cluster_id
        if mask.sum() < MIN_CLUSTER_SIZE:
            continue
        cluster_embs = unmapped_embs[mask]
        centroid = cluster_embs.mean(axis=0, keepdims=True)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            continue
        centroid = centroid / norm
        best = cosine_similarity(centroid, cluster_embs)[0].argmax()
        orig_indices = [unmapped_indices[j] for j, m in enumerate(mask) if m]
        label = phrases[orig_indices[best]]
        for idx in orig_indices:
            cluster_map[idx] = label
    return cluster_map


def main():
    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Done.\n")

    #LOAD NEWSLETTER PHRASES AND MAP TO CANONICAL TOPICS
    #load newsletter phrases
    nl_df = load_newsletter_phrases()
    nl_phrases = nl_df["phrase"].to_list()

    #embed seed topics
    seed_embs = embed(model, SEED_TOPICS)

    #embed newsletter phrases
    nl_embs = embed(model, nl_phrases)

    #map to seed topics
    nl_mapped, nl_sims = map_to_topics(nl_embs, seed_embs, SEED_TOPICS)

    mapped_count = sum(1 for t in nl_mapped if t is not None)
    unmapped_idx = [i for i, t in enumerate(nl_mapped) if t is None]
    print(f"  Mapped: {mapped_count} ({mapped_count/len(nl_phrases)*100:.1f}%)")
    print(f"  Unmapped: {len(unmapped_idx)}")

    print(f"\nClustering {len(unmapped_idx)} unmapped phrases...")
    cluster_map = cluster_unmapped(nl_embs, unmapped_idx, nl_phrases)
    discovered = list(set(cluster_map.values()))
    print(f"  Discovered {len(discovered)} new topics")

    # finalize newsletter mapping
    final_nl_topics = []
    for i, topic in enumerate(nl_mapped):
        if topic is not None:
            final_nl_topics.append(topic)
        elif i in cluster_map:
            final_nl_topics.append(cluster_map[i])
        else:
            final_nl_topics.append(None)

    nl_map_df = nl_df.with_columns([
        pl.Series("canonical_topic", final_nl_topics),
        pl.Series("similarity", nl_sims),
    ]).filter(pl.col("canonical_topic").is_not_null())

    nl_map_df.write_csv(NEWSLETTER_MAP_OUT)
    print(f"Saved newsletter map: {len(nl_map_df)} rows -> {NEWSLETTER_MAP_OUT}")

    # build canonical topic list
    all_canonical = SEED_TOPICS + discovered
    canonical_df = pl.DataFrame({
        "canonical_topic": all_canonical,
        "type": ["seed"] * len(SEED_TOPICS) + ["discovered"] * len(discovered),
    }).unique(subset=["canonical_topic"])
    canonical_df.write_csv(TOPICS_OUT)
    print(f"Saved {len(canonical_df)} canonical topics -> {TOPICS_OUT}")

    # ── STEP 2: signal phrases → canonical topics ─────────────────────────
    print("\n=== STEP 2: Signal phrases → canonical topics ===\n")

    print("Loading signal phrases...")
    sig_df = load_signal_phrases()

    if sig_df.is_empty():
        print("No signal phrases — skipping step 2")
        return

    print(f"\nEmbedding {len(all_canonical)} canonical topics...")
    canonical_embs = embed(model, all_canonical)

    sig_phrases = sig_df["phrase"].to_list()
    all_mapped = []
    all_sims = []

    print(f"\nMapping {len(sig_phrases)} signal phrases in batches...")
    for start in range(0, len(sig_phrases), SIGNAL_BATCH_SIZE):
        end = min(start + SIGNAL_BATCH_SIZE, len(sig_phrases))
        batch = sig_phrases[start:end]
        print(f"  Batch {start}-{end}...")
        batch_embs = embed(model, batch, batch_size=512)
        batch_mapped, batch_sims = map_to_topics(batch_embs, canonical_embs, all_canonical)
        all_mapped.extend(batch_mapped)
        all_sims.extend(batch_sims)

    sig_map_df = sig_df.with_columns([
        pl.Series("canonical_topic", all_mapped),
        pl.Series("similarity", all_sims),
    ]).filter(pl.col("canonical_topic").is_not_null())

    sig_map_df.write_csv(SIGNAL_MAP_OUT)
    print(f"Saved signal map: {len(sig_map_df)} rows -> {SIGNAL_MAP_OUT}")

    # ── SUMMARY ───────────────────────────────────────────────────────────
    print("\n=== SUMMARY ===")
    print(f"Canonical topics: {len(canonical_df)} ({len(SEED_TOPICS)} seed + {len(discovered)} discovered)")
    print(f"Newsletter phrases mapped: {len(nl_map_df)}")
    print(f"Signal phrases mapped: {len(sig_map_df)}")

    print("\nTop topics by newsletter mentions:")
    print(
        nl_map_df.group_by("canonical_topic")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(15)
    )

    print("\nTop topics by signal mentions:")
    print(
        sig_map_df.group_by("canonical_topic")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(15)
    )


if __name__ == "__main__":
    main()