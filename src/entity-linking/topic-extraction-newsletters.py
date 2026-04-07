import os
import polars as pl
from datetime import datetime
from keybert import KeyBERT

#Path setup
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(ROOT, "data", "raw")
OUT_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "newsletter_topics.csv")

#Newsletter sources to extract from
NEWSLETTER_SOURCES = [
    "tldr_ai",
    "tldr_tech",
    "tldr_fintech",
    "tldr_founders",
    "import_ai",
    "bits_in_bio",
    "the_batch",
]
#KeyBERT config
TOP_N = 10
NGRAM_RANGE = (1, 3)
DIVERSITY = 0.5

def week_of(date_str: str) -> str:
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        iso = d.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    except Exception:
        return ""


def load_newsletter(source: str) -> pl.DataFrame:
    path = os.path.join(RAW_DIR, f"{source}.csv")
    if not os.path.exists(path):
        print(f"  Skipping {source} — file not found")
        return pl.DataFrame()
    df = pl.read_csv(path, infer_schema_length=1000)
    if "text" not in df.columns:
        print(f"  Skipping {source} — no text column")
        return pl.DataFrame()
    df = df.filter(pl.col("text").str.len_chars() > 100)
    return df

def extract_topics(kw_model: KeyBERT, df: pl.DataFrame, source: str) -> list[dict]:
    records = []
    texts = df["text"].to_list()
    dates = df["date"].to_list() if "date" in df.columns else [""] * len(texts)

    for i, (text, date) in enumerate(zip(texts, dates)):
        if not text or len(text) < 100:
            continue
        try:
            keywords = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=NGRAM_RANGE,
                stop_words="english",
                use_mmr=True,
                diversity=DIVERSITY,
                top_n=TOP_N,
            )
            for phrase, score in keywords:
                records.append({
                    "phrase": phrase.lower().strip(),
                    "score": round(score, 4),
                    "source": source,
                    "date": str(date)[:10],
                    "week": week_of(str(date)),
                })
        except Exception as e:
            print(f"  Error on row {i}: {e}")
            continue

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(texts)} issues")

    return records


def main():
    print("Loading KeyBERT model...")
    kw_model = KeyBERT(model="all-MiniLM-L6-v2")
    print("Model loaded.\n")
    all_records = []
    for i, source in enumerate(NEWSLETTER_SOURCES):
        print(f"[{i+1}/{len(NEWSLETTER_SOURCES)}] {source}")
        df = load_newsletter(source)
        if df.is_empty():
            continue
        print(f"  {len(df)} issues to process")
        records = extract_topics(kw_model, df, source)
        print(f"  Extracted {len(records)} phrases")
        all_records.extend(records)

    if not all_records:
        print("No records extracted—check that data/raw/ CSVs exist")
        return

    combined = pl.DataFrame(all_records)
    combined = combined.sort(["date", "score"], descending=[False, True])
    combined.write_csv(OUT_PATH)

    print(f"\nDone. Saved {len(combined)} phrases -> {OUT_PATH}")
    print(f"Unique phrases: {combined['phrase'].n_unique()}")
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    print("\nPhrases per source:")
    print(
        combined.group_by("source")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )


if __name__ == "__main__":
    main()