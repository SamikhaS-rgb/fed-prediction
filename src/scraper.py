"""
scraper.py
----------
Downloads all FOMC meeting statements from federalreserve.gov.
Stores each statement as a plain text file in data/raw/fomc/.

HOW TO RUN:
    python src/scraper.py

No API key required — all data is publicly available on federalreserve.gov.
"""

import os
import re
import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

#Paths
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_FOMC = os.path.join(ROOT, "data", "raw", "fomc")
os.makedirs(RAW_FOMC, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (research project; contact: student@university.edu)"}
SLEEP   = 1.5   # seconds between requests — be polite to the Fed's servers


#1. Discover statement URLs

def get_statement_urls(start_year: int, end_year: int) -> list[dict]:
    """
    Scrapes the Fed's historical FOMC calendar pages to find all
    press-release / statement links for each meeting.

    Returns a list of dicts: [{date, year, url}, ...]
    """
    found = []

    for year in range(start_year, end_year + 1):
        if year <= 2010:
            url = f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
        else:
            url = f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"  [!] Could not fetch archive for {year}: {e}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True).lower()
            href = a["href"]

            is_statement = (
                ("statement" in text or "press release" in text)
                and ("monetary" in text or "fomcpr" in href.lower() or "press" in href.lower())
            )

            if not is_statement:
                is_statement = bool(re.search(r"fomcpr\d{8}", href, re.IGNORECASE))

            if is_statement:
                full_url = href if href.startswith("http") else "https://www.federalreserve.gov" + href
                match    = re.search(r"(\d{8})", href)
                date_str = match.group(1) if match else f"{year}0101"
                found.append({"date": date_str, "year": year, "url": full_url})

        time.sleep(SLEEP)

    seen = set()
    unique = []
    for rec in found:
        if rec["url"] not in seen:
            seen.add(rec["url"])
            unique.append(rec)

    print(f"[scraper] Found {len(unique)} FOMC statement links ({start_year}–{end_year})")
    return sorted(unique, key=lambda r: r["date"])


# 2. Download each statement

def download_statements(records: list[dict]):
    """Downloads each FOMC statement and saves as plain text."""
    print("[scraper] Downloading FOMC statements...")
    for rec in tqdm(records, desc="Statements"):
        fname = os.path.join(RAW_FOMC, f"fomc_statement_{rec['date']}.txt")
        if os.path.exists(fname):
            continue  # already downloaded

        try:
            resp = requests.get(rec["url"], headers=HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Try to extract just the statement body
            content = (
                soup.find("div", {"id": "content"})
                or soup.find("div", {"class": "col-xs-12 col-sm-8 col-md-8"})
                or soup.find("article")
                or soup.body
            )
            text = content.get_text(separator="\n", strip=True) if content else resp.text

            with open(fname, "w", encoding="utf-8") as f:
                f.write(f"DATE: {rec['date']}\nURL: {rec['url']}\n\n{text}")

        except Exception as e:
            print(f"  [!] Failed {rec['date']}: {e}")

        time.sleep(SLEEP)


# Entry point

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1994,
                        help="First year to scrape (default: 1994)")
    parser.add_argument("--end",   type=int, default=2024,
                        help="Last year to scrape (default: 2024)")
    args = parser.parse_args()

    print(f"[scraper] Scraping FOMC statements {args.start}–{args.end}...")
    records = get_statement_urls(args.start, args.end)
    download_statements(records)
    print(f"[scraper] Done. Statements saved to {RAW_FOMC}")


if __name__ == "__main__":
    main()
