import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import argparse
import os
import re
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)
 
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
 
def get(url, retries=3, delay=2):
    """GET with retries and polite delay."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            time.sleep(delay)
            return r
        except Exception as e:
            print(f"  Attempt {attempt+1} failed for {url}: {e}")
            time.sleep(delay * 2)
    return None

TLDR_EDITIONS = {
    "tldr_tech":     "https://tldr.tech/tech/archives",
    "tldr_ai":       "https://tldr.tech/ai/archives",
    "tldr_fintech":  "https://tldr.tech/fintech/archives",
    "tldr_founders": "https://tldr.tech/founders/archives",
}

#-------------SCRAPING FROM TLDR--------------------------
 
# TLDR issue URLs follow a predictable pattern: tldr.tech/{slug}/{YYYY-MM-DD}
# The archive page only shows ~20 recent issues (JS-rendered), so instead we
# generate all weekday dates in our range and probe each one directly.
# Issues that don't exist return a non-200 or redirect — we skip those.
 
TLDR_EDITIONS = {
    "tldr_tech":     "tech",
    "tldr_ai":       "ai",
    "tldr_fintech":  "fintech",
    "tldr_founders": "founders",
}
 
# Date range to scrape — adjust as needed
TLDR_START_DATE = "2023-01-01"
TLDR_END_DATE   = datetime.today().strftime("%Y-%m-%d")
 
def generate_weekdays(start: str, end: str):
    """Generate all weekday dates between start and end (inclusive)."""
    from datetime import timedelta
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    dates = []
    current = s
    while current <= e:
        if current.weekday() < 5:  # Mon–Fri only
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates
 
def scrape_tldr_archive(edition_name, slug):
    """
    Probe every weekday URL for a TLDR edition.
    Issues that don't exist will 404 or redirect — skip them.
    Returns list of dicts.
    """
    print(f"\nScraping TLDR: {edition_name} ({TLDR_START_DATE} to {TLDR_END_DATE})")
    dates = generate_weekdays(TLDR_START_DATE, TLDR_END_DATE)
    print(f"  Probing {len(dates)} weekday dates...")
 
    records = []
    for i, date in enumerate(dates):
        url = f"https://tldr.tech/{slug}/{date}"
        r = get(url, delay=1)  # shorter delay since most will 404 fast
        if not r:
            continue  # 404 or timeout — issue doesn't exist for this date
 
        # Confirm it's actually an issue page, not a redirect to homepage
        if "tldr.tech" not in r.url or date not in r.url:
            continue
 
        rec = scrape_tldr_issue(edition_name, url, date, r)
        if rec:
            records.append(rec)
            print(f"  [+] {date} — found ({len(records)} total)")
 
    return records
 
 
def scrape_tldr_issue(source, url, date, r):
    """Parse a single TLDR issue from a pre-fetched response."""
    soup = BeautifulSoup(r.text, "html.parser")
 
    # Title from <h1> or <title>
    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
 
    # Main content
    content_div = soup.find("div", class_=re.compile(r"content|article|body|main", re.I))
    if content_div:
        text = content_div.get_text(separator=" ", strip=True)
    else:
        text = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))
 
    if len(text) < 100:
        return None  # probably a redirect or empty page
 
    return {
        "source": source,
        "date": date,
        "title": title,
        "url": url,
        "text": text,
    }

#-------------SCRAPING FROM SUBSTACK--------------------------
SUBSTACK_SOURCES = {
    "import_ai":   "https://importai.substack.com",
    "bits_in_bio": "https://bitsinbio.substack.com",
}
 
def scrape_substack_archive(source_name, base_url, max_posts=200):
    """
    Scrape Substack archive using the JSON API endpoint.
    Substack exposes: /api/v1/archive?sort=new&offset=N&limit=12
    Returns list of dicts.
    """
    print(f"\nScraping Substack: {source_name}")
    api = f"{base_url}/api/v1/archive?sort=new&limit=12&offset="
 
    all_posts = []
    offset = 0
 
    while True:
        r = get(api + str(offset), delay=1.5)
        if not r:
            break
        try:
            posts = r.json()
        except Exception:
            break
 
        if not posts:
            break
 
        all_posts.extend(posts)
        print(f"  Fetched {len(all_posts)} posts so far...")
 
        if len(posts) < 12 or len(all_posts) >= max_posts:
            break
        offset += 12
 
    print(f"  Total posts found: {len(all_posts)}")
 
    records = []
    for i, post in enumerate(all_posts):
        url = post.get("canonical_url") or post.get("url", "")
        print(f"  [{i+1}/{len(all_posts)}] {url}")
        rec = scrape_substack_post(source_name, post, url)
        if rec:
            records.append(rec)
 
    return records
 
 
def scrape_substack_post(source_name, post_meta, url):
    """
    Fetch full text of a Substack post.
    Falls back to post_meta description if full fetch fails.
    """
    # Try to get full text from the post page
    r = get(url, delay=1.5)
    text = ""
    if r:
        soup = BeautifulSoup(r.text, "html.parser")
        # Substack wraps post body in .available-content or .post-content
        body = (
            soup.find("div", class_="available-content")
            or soup.find("div", class_="post-content")
            or soup.find("article")
        )
        if body:
            text = body.get_text(separator=" ", strip=True)
 
    # Fallback to description from API metadata
    if not text:
        text = post_meta.get("description", "") or post_meta.get("subtitle", "")
 
    published = post_meta.get("post_date", "") or post_meta.get("publishedBylines", "")
    # Normalize date to YYYY-MM-DD
    if published:
        try:
            published = datetime.fromisoformat(published[:10]).strftime("%Y-%m-%d")
        except Exception:
            published = published[:10]
 
    return {
        "source": source_name,
        "date": published,
        "title": post_meta.get("title", ""),
        "url": url,
        "text": text,
    }

#-------------SCRAPING FROM THE BATCH--------------------------
THE_BATCH_ARCHIVE = "https://www.deeplearning.ai/the-batch/"
 
def scrape_the_batch(max_issues=100):
    """
    Scrape The Batch archive from deeplearning.ai.
    Archive page lists all issues as links.
    """
    print("\nScraping The Batch (deeplearning.ai)")
    r = get(THE_BATCH_ARCHIVE)
    if not r:
        return []
 
    soup = BeautifulSoup(r.text, "html.parser")
 
    # Issue links look like /the-batch/issue-NNN/
    pattern = re.compile(r"/the-batch/issue-\d+")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if pattern.search(href):
            full = "https://www.deeplearning.ai" + href if href.startswith("/") else href
            if full not in links:
                links.append(full)
 
    links = links[:max_issues]
    print(f"  Found {len(links)} issues")
 
    records = []
    for i, url in enumerate(links):
        print(f"  [{i+1}/{len(links)}] {url}")
        r = get(url)
        if not r:
            continue
 
        soup = BeautifulSoup(r.text, "html.parser")
 
        # Date—look for <time> tag or date-like text in meta
        date = ""
        time_tag = soup.find("time")
        if time_tag:
            date = time_tag.get("datetime", time_tag.get_text(strip=True))[:10]
 
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else ""
 
        # Body content
        body = soup.find("article") or soup.find("main")
        text = body.get_text(separator=" ", strip=True) if body else ""
 
        records.append({
            "source": "the_batch",
            "date": date,
            "title": title,
            "url": url,
            "text": text,
        })
 
    return records

#-------------SAVING EVERYTHING--------------------------

def save(records, name):
    if not records:
        print(f"  No records to save for {name}")
        return
    df = pd.DataFrame(records)
    # Drop rows with no text
    df = df[df["text"].str.len() > 50].reset_index(drop=True)
    path = os.path.join(RAW_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows -> {path}")



ALL_SOURCES = list(TLDR_EDITIONS.keys()) + list(SUBSTACK_SOURCES.keys()) + ["the_batch"]
 
def run(source=None):
    targets = [source] if source else ALL_SOURCES
 
    for name in targets:
        if name in TLDR_EDITIONS:
            records = scrape_tldr_archive(name, TLDR_EDITIONS[name])
            save(records, name)
 
        elif name in SUBSTACK_SOURCES:
            records = scrape_substack_archive(name, SUBSTACK_SOURCES[name])
            save(records, name)
 
        elif name == "the_batch":
            records = scrape_the_batch()
            save(records, name)
 
        else:
            print(f"Unknown source: {name}. Options: {ALL_SOURCES}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Newsletter scraper")
    parser.add_argument(
        "--source",
        choices=ALL_SOURCES,
        default=None,
        help="Scrape one source only. Omit to scrape all.",
    )
    args = parser.parse_args()
    run(args.source)
    