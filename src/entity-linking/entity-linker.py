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
MIN_SCORE_SIGNAL = 0.50
START_DATE = "2023-01-01"
SIMILARITY_THRESHOLD = 0.72
CLUSTER_DISTANCE = 0.20
MIN_CLUSTER_SIZE = 5
SIGNAL_BATCH_SIZE = 50000
MIN_DISCOVERY_SCORE = 0.60

#noticed recurring sponsorship info that was useless and messing wtih data set so we are removing the common ones
SPONSOR_BRANDS = {
    "drata", "workos", "swarmia", "rippling", "warp", "sentry", "vanta",
    "lmnt", "snhu", "pacaso", "jurny", "brex", "retool", "notion",
    "linear", "coda", "vercel", "planetscale", "neon", "supabase",
    "clerk", "convex", "tigris", "axiom", "depot", "buildkite",
    "render", "railway", "fly.io", "koyeb", "zeet",
    "incogni", "surfshark", "nordvpn", "expressvpn",
    "grammarly", "jasper", "copy.ai", "writesonic",
    "masterclass", "brilliant", "skillshare", "coursera",
    "athletic greens", "ag1", "helix", "eight sleep",
    "1inch", "everyrealm", "futurex",
    "modern treasury", "mercury", "ramp",
    "anyscale", "modal", "baseten", "replicate",
    "butter", "contentful", "sanity",
    "cdata", "sage intacct",
}

JUNK_PATTERNS = [
    #newsletter metadatata and generic engagement phrases that aren't really topics and just add noise
    r"subscribe|click here|read more|this week|last week|sign up",
    r"minute read|big read|quick read|deep dive",
    r"sponsor|sponsored|partner|advertisement",
    r"daily email|weekly email|newsletter|readers",
    r"hiring|remote|salary|interview|resume",
    #dates and years
    r"2023|2024|2025|2026",
    #newsletter names
    r"tldr|import ai|bits bio|the batch|alphasense",
    #too generic to be useful (always trending and not really a topic on its own)
    r"^tech$|^ai$|^data$|^cloud$|^code$|^app$|^tool$|^api$",
    r"coffee break|behaviors make|feed uses|event start",
    r"development time|component library",
    #product review/specs language
    r"specs features|colors price|benchmark report",
    r"reviewers rated|stars rating",
    #drinks/lifestyle (LMNT sponsor content)
    r"electrolyte|sparkling|hydration|drinks try",
    r"vacation|save \dk",
]

#increased seed topics from first iteratino. This gives more chances for phrases to map to a known topic and reduces noise in the clustering step, which should improve discovered topics. We thought about the risk of confirmation bias but the original seed topics were pretty broad and generic, so adding more specific ones actually helps surface more specific mappings and lets the clustering focus on truly novel topics instead of just splitting hairs between broad ones. We also made sure to include a wide range of topics across AI, fintech, biotech, consumer tech, and more to cover the diversity of the newsletter content.
SEED_TOPICS = [
    #──AI models & architectures──
    "large language model",
    "GPT-4 capabilities",
    "Claude AI assistant",
    "Gemini multimodal model",
    "Llama open source model",
    "Mistral language model", 
    "diffusion model image generation",
    "text to video generation",
    "text to speech synthesis",
    "mixture of experts architecture",
    "transformer neural network",
    "vision language model",
    "small language model on device", 
    "retrieval augmented generation",  
    #──AI applications(specific)——
    "AI coding assistant tool",  
    "autonomous AI agent framework",
    "AI chatbot customer service",
    "AI image generation art",
    "AI drug discovery research",
    "AI chip hardware accelerator",
    "AI voice assistant",
    "AI search engine", 
    "AI video editing tool",
    # ──AI industry/policy──
    "OpenAI company news",
    "Google DeepMind research",
    "Anthropic AI safety",
    "AI safety alignment research",
    "AI regulation government policy",
    "EU AI Act regulation",
    "AI copyright training data",
    "open source model weights release",
    "AI compute GPU shortage",
    "AI energy consumption datacenter",
    # ──ML techniques──
    "reinforcement learning from human feedback",
    "fine tuning language model",
    "prompt engineering techniques",
    "synthetic data generation training",
    "model quantization compression",
    "multimodal AI system",
    #──Dev/infra──
    "Kubernetes container orchestration",
    "WebAssembly runtime performance",
    "developer tools productivity",
    "serverless cloud computing",
    "edge computing deployment",
    "observability monitoring platform",
    "CI CD pipeline automation",
    "infrastructure as code",
    "vector database embeddings",
    "API gateway management",
    #──Fintech──
    "algorithmic trading strategy",
    "decentralized finance DeFi protocol",
    "cryptocurrency Bitcoin price",
    "stablecoin payment regulation",
    "fintech startup neobank",
    "venture capital fundraising",
    "private equity investment",
    "banking API open finance",
    #──Biotech──
    "CRISPR gene editing therapy",
    "GLP-1 obesity weight loss drug",
    "protein structure AlphaFold prediction",
    "mRNA vaccine technology",
    "drug discovery clinical trial",
    "longevity aging research",
    "synthetic biology engineering",
    #──Startups/business──
    "AI startup funding round",
    "startup seed series fundraising",
    "Y Combinator accelerator batch",
    "tech IPO public offering",
    "big tech layoffs restructuring",
    "remote work return office",
    #──Hardware/chips──
    "semiconductor chip manufacturing",
    "NVIDIA GPU datacenter",
    "Apple silicon M-series chip",
    "quantum computing qubit",
    "RISC-V processor architecture",
    # ──Consumer tech──
    "Apple Vision Pro headset",
    "Tesla autopilot self driving",
    "Tesla Cybertruck vehicle",
    "electric vehicle EV battery",
    "robotics humanoid robot",
    "smart glasses wearable AR",
    "satellite internet Starlink",
    #──Security──
    "cybersecurity ransomware attack",
    "zero trust security architecture",
    "data privacy GDPR compliance",
    #─Platforms──
    "TikTok ban regulation",
    "Twitter X platform changes",
    "Meta Threads social network",
    "Reddit community platform",
]

#extracting week from date for weekly aggregation in signal phrases. This is important because signal phrases can be more noisy and we want to aggregate them at the week level to get more stable mappings. The newsletter phrases are already aggregated at the week level since they come from the weekly newsletter, but the signal phrases are more granular and can vary day to day, so we want to group them by week to find consistent signals that align with our canonical topics.
def week_of(date_str: str) -> str:
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        iso = d.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    except Exception:
        return ""

#preprocessing step
def contains_sponsor(phrase: str) -> bool:
    """Check if a phrase mentions a known sponsor brand to filter out."""
    phrase_lower = phrase.lower()
    for brand in SPONSOR_BRANDS:
        if brand in phrase_lower:
            return True
    return False

#preprocessing and filtering steps for newsletter and signal phrases. We apply a series of filters to clean the data before embedding and mapping. This includes removing low-score phrases, filtering by date, removing short phrases, applying regex filters to remove junk patterns, and finally filtering out any phrases that mention known sponsor brands. This ensures that the remaining phrases are more likely to be meaningful topics rather than noise or promotional content.
def load_newsletter_phrases() -> pl.DataFrame:
    df = pl.read_csv(NEWSLETTER_IN)
    df = df.filter(pl.col("score") >= MIN_SCORE_NEWSLETTER)
    df = df.filter(pl.col("date") >= START_DATE)
    df = df.filter(pl.col("date").is_not_null())
    df = df.filter(pl.col("phrase").str.len_chars() > 3)
 
    #combined all junk patterns into one regex pass
    junk_regex = "|".join(JUNK_PATTERNS)
    df = df.filter(~pl.col("phrase").str.to_lowercase().str.contains(junk_regex))
 
    df = df.filter(pl.col("phrase").str.split(" ").list.len() <= 5)  #allow up to 5 words for more specific phrases
    df = df.filter(pl.col("phrase").str.split(" ").list.len() >= 2)  #single words are too ambiguous
 
    #filter out sponsor brand mentions
    phrases = df["phrase"].to_list()
    sponsor_mask = [not contains_sponsor(p) for p in phrases]
    df = df.filter(pl.Series(sponsor_mask))
 
    print(f"  Loaded {len(df)} newsletter phrases after filtering")
    return df

#filtering and preprocessing the signal phrases
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
    phrases = df["phrase"].to_list()
    sponsor_mask = [not contains_sponsor(p) for p in phrases]
    df = df.filter(pl.Series(sponsor_mask))

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
    #comutes cosine_similarity between all phrases and topics by their embeddings according to the sentence transformers model
    sims = cosine_similarity(phrase_embeddings, topic_embeddings) #This computes cosine similarity between every phrase and every topic simultaneously. phrase_embeddings is a matrix of shape (N_phrases × 384) and topic_embeddings is (N_topics × 384). The result sims is a matrix of shape (N_phrases × N_topics) where sims[i][j] is the similarity between phrase i and topic j. So if you have 6000 phrases and 219 topics, sims is a 6000×219 matrix.
    best_idx = sims.argmax(axis=1) #For each phrase (each row), find the column index of the highest similarity score. 
    best_sim = sims.max(axis=1) #For each phrase (each row), find the highest similarity score
    mapped = [topics[i] if s >= SIMILARITY_THRESHOLD else None #For each phrase, look up the topic name using the index. But only assign it if the similarity score clears the threshold (0.55).
              for i, s in zip(best_idx, best_sim)]
    return mapped, best_sim.tolist() #returns the list of assigned topic names (or None), and the list of similarity scores. The scores get stored in output CSV as the similarity column/how confident each mapping was


def cluster_unmapped(
    phrase_embeddings: np.ndarray,
    unmapped_indices: list[int],
    phrases: list[str],
    scores: list[float],
) -> dict[int, str]:
    #filters by MIN_DISCOVERY_SCORE. Only high-confidence KeyBERT phrases can define new topic clusters. This prevents low-quality phrases from polluting discovered topics, which we had an issue with earlier.
    if len(unmapped_indices) < MIN_CLUSTER_SIZE:
        return {}
 
    # Only cluster phrases with high enough KeyBERT scores
    # low-score phrases are too generic to define meaningful new topics
    quality_mask = [scores[i] >= MIN_DISCOVERY_SCORE for i in unmapped_indices]
    quality_indices = [idx for idx, keep in zip(unmapped_indices, quality_mask) if keep]
    
    print(f"  Quality filter: {len(unmapped_indices)} unmapped -> {len(quality_indices)} with score >= {MIN_DISCOVERY_SCORE}")
    
    if len(quality_indices) < MIN_CLUSTER_SIZE:
        return {}
 
    unmapped_embs = phrase_embeddings[quality_indices]
 
    # filter NaN/inf embeddings
    valid_mask = np.isfinite(unmapped_embs).all(axis=1)
    if not valid_mask.all():
        print(f"  Dropping {(~valid_mask).sum()} NaN/inf embeddings")
        unmapped_embs = unmapped_embs[valid_mask]
        quality_indices = [quality_indices[i] for i, v in enumerate(valid_mask) if v]
    if len(quality_indices) < MIN_CLUSTER_SIZE:
        return {}
 
    #agglomerative clustering with cosine distance, felt would work better than k-means since it is more important to stop clustering once topics are a specific similarity, rather than how many clusters we have (some would be arbitrary)
    #https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=CLUSTER_DISTANCE,
        metric="cosine",
        linkage="average",
    ).fit_predict(unmapped_embs)
 
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
        centroid = centroid/norm
 
        # find the phrase closest to the centroid as the cluster label
        best = cosine_similarity(centroid, cluster_embs)[0].argmax()
        orig_indices = [quality_indices[j] for j, m in enumerate(mask) if m]
        label = phrases[orig_indices[best]]
 
        #Just to make sure it is appropriate, we validate the discovered topic before accepting it
        if contains_sponsor(label):
            continue
        #skip single-word labels (too generic as we were noticing in the results)
        if len(label.split()) < 2:
            continue
        #skip labels that are just generic phrases
        label_lower = label.lower()
        if any(label_lower.startswith(w) for w in ["tech startups", "big tech", "minute read"]):
            continue
        
        for idx in orig_indices:
            cluster_map[idx] = label
    return cluster_map


def filter_discovered_topics(
    model: SentenceTransformer,
    discovered: list[str],
    min_tech_similarity: float = 0.30,
) -> list[str]:
    #Filter discovered topics by checking they're actually about technology. Embeds anchor phrases like "technology software hardware" and drops discovered topics that are too far from this anchor. This catches things like "save 2k vacation" and "electrolyte drink".
    if not discovered:
        return []
 
    tech_anchors = [
        "technology software engineering",
        "artificial intelligence machine learning",
        "startup company funding investment",
        "scientific research breakthrough",
        "hardware chip processor computing",
        "cybersecurity data privacy",
        "biotechnology medicine pharmaceutical",
        "cryptocurrency blockchain finance",
    ]
    
    anchor_embs = model.encode(tech_anchors, normalize_embeddings=True)
    anchor_centroid = anchor_embs.mean(axis=0, keepdims=True)
    anchor_centroid = anchor_centroid / np.linalg.norm(anchor_centroid)
 
    topic_embs = model.encode(discovered, normalize_embeddings=True)
    sims = cosine_similarity(topic_embs, anchor_centroid).flatten()
 
    kept = []
    dropped = []
    for topic, sim in zip(discovered, sims):
        if sim >= min_tech_similarity:
            kept.append(topic)
        else:
            dropped.append((topic, sim))
 
    if dropped:
        print(f"\n  Dropped {len(dropped)} non-tech discovered topics:")
        for topic, sim in sorted(dropped, key=lambda x: x[1]):
            print(f"    {sim:.3f}  {topic}")
 
    return kept

def main():
    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Done.\n")
 
    # ──STEP 1: newsletter phrases → canonical topics─────────────────────
    print("=== STEP 1: Newsletter phrases → canonical topics ===\n")
 
    nl_df = load_newsletter_phrases()
    nl_phrases = nl_df["phrase"].to_list()
    nl_scores = nl_df["score"].to_list()
 
    print(f"\nEmbedding {len(SEED_TOPICS)} seed topics...")
    seed_embs = embed(model, SEED_TOPICS)
 
    print(f"Embedding {len(nl_phrases)} newsletter phrases...")
    nl_embs = embed(model, nl_phrases)
 
    print("\nMapping newsletter phrases to seed topics...")
    nl_mapped, nl_sims = map_to_topics(nl_embs, seed_embs, SEED_TOPICS)
 
    mapped_count = sum(1 for t in nl_mapped if t is not None)
    unmapped_idx = [i for i, t in enumerate(nl_mapped) if t is None]
    print(f"  Mapped: {mapped_count} ({mapped_count/len(nl_phrases)*100:.1f}%)")
    print(f"  Unmapped: {len(unmapped_idx)}")
 
    #Show top matched topics for diagnostics
    from collections import Counter
    topic_counts = Counter(t for t in nl_mapped if t is not None)
    print("\n  Top 10 matched seed topics:")
    for topic, count in topic_counts.most_common(10):
        print(f"    {count:4d}  {topic}")
 
    #Cluster unmapped phrases (now with quality filtering)
    print(f"\nClustering {len(unmapped_idx)} unmapped phrases...")
    cluster_map = cluster_unmapped(nl_embs, unmapped_idx, nl_phrases, nl_scores)
    discovered_raw = list(set(cluster_map.values()))
    print(f"  Discovered {len(discovered_raw)} raw cluster topics")
 
    #filter discovered topics for tech relevance
    print("\nFiltering discovered topics for tech relevance...")
    discovered = filter_discovered_topics(model, discovered_raw)
    print(f"  Kept {len(discovered)} / {len(discovered_raw)} discovered topics")
 
    #remove cluster assignments for topics that were filtered out
    kept_set = set(discovered)
    cluster_map = {k: v for k, v in cluster_map.items() if v in kept_set}
 
    #finalize newsletter mapping
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
    print(f"\nSaved newsletter map: {len(nl_map_df)} rows -> {NEWSLETTER_MAP_OUT}")
 
    #build canonical topic list
    all_canonical = SEED_TOPICS + discovered
    canonical_df = pl.DataFrame({
        "canonical_topic": all_canonical,
        "type": ["seed"] * len(SEED_TOPICS) + ["discovered"] * len(discovered),
    }).unique(subset=["canonical_topic"])
    canonical_df.write_csv(TOPICS_OUT)
    print(f"Saved {len(canonical_df)} canonical topics -> {TOPICS_OUT}")
 
    #──STEP 2: signal phrases → canonical topics─────────────────────────
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
 
    #──SUMMARY───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Canonical topics: {len(canonical_df)} ({len(SEED_TOPICS)} seed + {len(discovered)} discovered)")
    print(f"Newsletter phrases mapped: {len(nl_map_df)}")
    print(f"Signal phrases mapped: {len(sig_map_df)}")
 
    print("\nTop 15 topics by newsletter mentions:")
    print(
        nl_map_df.group_by("canonical_topic")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(15)
    )
 
    print("\nTop 15 topics by signal mentions:")
    print(
        sig_map_df.group_by("canonical_topic")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(15)
    )
 
#------extra stats so we can better undrsatand results----
    #Show coverage diagnostics
    print("\n--- DIAGNOSTICS ---")
    total_nl = len(nl_df)
    mapped_nl = len(nl_map_df)
    print(f"Newsletter coverage: {mapped_nl}/{total_nl} ({mapped_nl/total_nl*100:.1f}%)")
    
    if not sig_df.is_empty():
        total_sig = len(sig_df)
        mapped_sig = len(sig_map_df)
        print(f"Signal coverage: {mapped_sig}/{total_sig} ({mapped_sig/total_sig*100:.1f}%)")
 
    #show similarity distribution
    nl_sim_vals = nl_map_df["similarity"].to_list()
    if nl_sim_vals:
        print(f"\nNewsletter similarity stats:")
        print(f"  Mean: {np.mean(nl_sim_vals):.3f}")
        print(f"  Median: {np.median(nl_sim_vals):.3f}")
        print(f"  Min: {np.min(nl_sim_vals):.3f}")
        print(f"  P25: {np.percentile(nl_sim_vals, 25):.3f}")
        print(f"  P75: {np.percentile(nl_sim_vals, 75):.3f}")
 
 
if __name__ == "__main__":
    main()