import time
from datetime import datetime
import polars as pl
import requests
import xml.etree.ElementTree as ET

import urllib.parse
import urllib.request
# arXiv API
#https://info.arxiv.org/help/api/user-manual.html
def fetch_arxiv_data(query: str, max_results: int=50) -> pl.DataFrame:
    base_url = "http://export.arxiv.org/api/query?"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results
    }

    url = base_url + urllib.parse.urlencode(params)
    
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}")
    
    namespacePrefix = {"atom": "http://www.w3.org/2005/Atom"}

    #https://docs.python.org/3/library/xml.etree.elementtree.html (an API for parsing and creating XML data)
    root = ET.fromstring(response.text)

    data = [
        #https://info.arxiv.org/help/api/user-manual.html
        {
            "source": "arxiv",
            "query": query,
            "title": entry.find("atom:title", namespacePrefix).text.strip(),
            "abstract": entry.find("atom:summary", namespacePrefix).text.strip(),
            "published": entry.find("atom:published", namespacePrefix).text[:10],
            "citation_count": None,
            "url": entry.find("atom:id", namespacePrefix).text.strip()
        }
        for entry in root.findall("atom:entry",namespacePrefix)
    ] 

    return pl.DataFrame(data)

# Semantic Scholar API
#WAITING ON API KEY BC CANT REALLY DO ANYTHING WITH THESE RATE LIMITS
#https://www.semanticscholar.org/product/api%2Ftutorial
def fetch_SS_data(query: str, limit: int=50) -> pl.DataFrame:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    query_params = {
        "query": query,
        "limit": limit,
        "fields": "title,year,abstract,citationCount"
                    }
    response = None
    for attempt in range(3):
        response = requests.get(url, params=query_params)
        if response.status_code == 429:
            wait = 5 * (attempt + 1)
            print(f"Rate limited, waiting {wait}s...")
            time.sleep(wait)
        else:
            break
    
    response.raise_for_status()

    return [
        #https://info.arxiv.org/help/api/user-manual.html
        {
            "source": "semantic_scholar",
            "query": query,
            "title": p.get("title", ""),
            "abstract": p.get("abstract", ""),
            "published": p.get("publicationDate") or str(p.get("year", "")),
            "citation_count": p.get("citationCount", 0),
            "url": f"https://www.semanticscholar.org/paper/{p['paperId']}",
        }
        for p in response.json().get("data", [])
    ]



# X API


# Reddit API?


if __name__ == "__main__":
    df = fetch_arxiv_data("AI agents", max_results=5)
    df1 = fetch_SS_data("AI agents", limit=5)
    print(df)
    print(df1)