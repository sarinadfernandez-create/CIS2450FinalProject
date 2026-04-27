import polars as pl

newsletter = pl.read_csv("newsletter_topic_map.csv", 
                         schema_overrides={"date": pl.Utf8})
signal = pl.read_csv("signal_topic_map.csv",
                     schema_overrides={"date": pl.Utf8})

# count newsletter mentions per topic
nl_counts = (
    newsletter.group_by("canonical_topic")
    .agg(pl.len().alias("newsletter_mentions"))
    .sort("newsletter_mentions", descending=True)
)

# count signal mentions per topic
sig_counts = (
    signal.group_by("canonical_topic")
    .agg(pl.len().alias("signal_mentions"))
)

# join them
combined = nl_counts.join(sig_counts, on="canonical_topic", how="left").with_columns(
    pl.col("signal_mentions").fill_null(0)
)

print("Topics with newsletter mentions but low signal:")
print(
    combined.filter(pl.col("signal_mentions") < 50)
    .sort("newsletter_mentions", descending=True)
    .head(20)
)