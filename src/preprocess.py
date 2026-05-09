"""
preprocess.py
-------------
Cleans raw downloaded text files:
  - Strips boilerplate (page numbers, legal headers, repeated disclaimers)
  - Splits FOMC minutes into prepared_remarks vs Q&A sections
  - Normalises whitespace and encoding

HOW TO RUN:
    python src/preprocess.py
"""

import os
import re
import json
import pandas as pd
from tqdm import tqdm

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_FOMC    = os.path.join(ROOT, "data", "raw", "fomc")
RAW_BEIGE   = os.path.join(ROOT, "data", "raw", "beige_book")
RAW_REGIONAL= os.path.join(ROOT, "data", "raw", "regional")
PROCESSED   = os.path.join(ROOT, "data", "processed")
os.makedirs(PROCESSED, exist_ok=True)


#Text cleaning helpers

BOILERPLATE_PATTERNS = [
    r"page \d+ of \d+",
    r"^\s*\d+\s*$",                    # lone page numbers
    r"for release.*?embargo",          # embargo notices
    r"federal open market committee",  # repeated headers
    r"class i fomc - restricted",      # classification headers
    r"\[end of document\]",
    r"https?://\S+",                   # URLs
    r"\*{3,}",                         # decorative asterisks
]
BOILERPLATE_RE = re.compile(
    "|".join(BOILERPLATE_PATTERNS), re.IGNORECASE | re.MULTILINE
)


def clean_text(raw: str) -> str:
    """Apply all cleaning steps to a raw text string."""
    raw = re.sub(r"^DATE:.*?\n\nURL:.*?\n\n", "", raw, flags=re.DOTALL)

    raw = raw.replace("\u2019", "'").replace("\u2014", "--").replace("\u201c", '"').replace("\u201d", '"')

    raw = BOILERPLATE_RE.sub(" ", raw)

    raw = re.sub(r"\n{3,}", "\n\n", raw)

    raw = re.sub(r" {2,}", " ", raw)

    return raw.strip()



QA_MARKERS = [
    "by way of background",
    "question and answer",
    "questions and answers",
    "the chairman then",
    "mr. chairman",
    "madam chair",
    "open for questions",
    "discussion of",
    "participants discussed",
]
QA_RE = re.compile("|".join(QA_MARKERS), re.IGNORECASE)


def split_fomc_minutes(text: str) -> dict:
    """
    Splits a cleaned FOMC minutes text into:
      - prepared_remarks: the scripted policy statement section
      - discussion:       the open discussion / Q&A portion
    Returns a dict with both parts (may be empty strings if split not found).
    """
    match = QA_RE.search(text)
    if match:
        split_pos = match.start()
        return {
            "prepared_remarks": text[:split_pos].strip(),
            "discussion":       text[split_pos:].strip(),
            "full_text":        text,
        }
    else:
        return {
            "prepared_remarks": text,
            "discussion":       "",
            "full_text":        text,
        }


def extract_date_from_filename(fname: str) -> str:
    """Extracts date string like '20190320' from filenames."""
    match = re.search(r"(\d{8})", fname)
    return match.group(1) if match else "unknown"


#Process FOMC Minutes

def process_fomc_minutes() -> pd.DataFrame:
    """Cleans all FOMC minutes files and returns a DataFrame."""
    files = [f for f in os.listdir(RAW_FOMC) if f.endswith(".txt")]
    records = []

    print(f"[preprocess] Processing {len(files)} FOMC minutes files...")
    for fname in tqdm(files, desc="FOMC"):
        fpath = os.path.join(RAW_FOMC, fname)
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

        cleaned  = clean_text(raw)
        split    = split_fomc_minutes(cleaned)
        date_str = extract_date_from_filename(fname)

        records.append({
            "doc_id":           fname.replace(".txt", ""),
            "source":           "fomc_minutes",
            "date":             date_str,
            "year":             int(date_str[:4]) if date_str != "unknown" else None,
            "month":            int(date_str[4:6]) if date_str != "unknown" else None,
            "prepared_remarks": split["prepared_remarks"],
            "discussion":       split["discussion"],
            "full_text":        split["full_text"],
            "word_count":       len(split["full_text"].split()),
        })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    out_path = os.path.join(PROCESSED, "fomc_minutes_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"[preprocess] Saved {len(df)} FOMC records → {out_path}")
    return df



DISTRICT_HEADERS = {
    "boston":       ["first district", "boston"],
    "new_york":     ["second district", "new york"],
    "philadelphia": ["third district", "philadelphia"],
    "cleveland":    ["fourth district", "cleveland"],
    "richmond":     ["fifth district", "richmond"],
    "atlanta":      ["sixth district", "atlanta"],
    "chicago":      ["seventh district", "chicago"],
    "st_louis":     ["eighth district", "st. louis", "st louis"],
    "minneapolis":  ["ninth district", "minneapolis"],
    "kansas_city":  ["tenth district", "kansas city"],
    "dallas":       ["eleventh district", "dallas"],
    "san_francisco":["twelfth district", "san francisco"],
}


def split_beige_book_by_district(text: str) -> dict[str, str]:
    """
    Splits a Beige Book text into one section per Federal Reserve district.
    Returns a dict: {bank_name: section_text}
    """
    # Build a pattern that finds any district header
    all_headers = []
    header_to_bank = {}
    for bank, headers in DISTRICT_HEADERS.items():
        for h in headers:
            all_headers.append(re.escape(h))
            header_to_bank[h] = bank

    pattern = re.compile(
        r"(" + "|".join(all_headers) + r")",
        re.IGNORECASE
    )

    matches = list(pattern.finditer(text))
    if not matches:
        return {"full": text}

    sections = {}
    for i, match in enumerate(matches):
        header_text = match.group(1).lower()
        bank = next((header_to_bank[h] for h in header_to_bank if h in header_text), "unknown")
        start = match.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[bank] = text[start:end].strip()

    return sections


def process_beige_books() -> pd.DataFrame:
    """Cleans Beige Book files and splits them by district."""
    files = [f for f in os.listdir(RAW_BEIGE) if f.endswith(".txt")]
    records = []

    print(f"[preprocess] Processing {len(files)} Beige Book files...")
    for fname in tqdm(files, desc="Beige Books"):
        fpath = os.path.join(RAW_BEIGE, fname)
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

        cleaned  = clean_text(raw)
        date_str = extract_date_from_filename(fname)
        districts = split_beige_book_by_district(cleaned)

        for bank, section_text in districts.items():
            records.append({
                "doc_id":    f"beige_{date_str}_{bank}",
                "source":    "beige_book",
                "bank":      bank,
                "date":      date_str,
                "year":      int(date_str[:4]) if date_str != "unknown" else None,
                "month":     int(date_str[4:6]) if date_str != "unknown" else None,
                "full_text": section_text,
                "word_count":len(section_text.split()),
            })

    df = pd.DataFrame(records).sort_values(["date", "bank"]).reset_index(drop=True)
    out_path = os.path.join(PROCESSED, "beige_books_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"[preprocess] Saved {len(df)} Beige Book district records → {out_path}")
    return df


#Process Regional Fed publications

def process_regional_publications() -> pd.DataFrame:
    """Cleans all regional bank text files."""
    records = []
    regional_root = RAW_REGIONAL

    print("[preprocess] Processing regional Fed publications...")
    for bank_name in os.listdir(regional_root):
        bank_dir = os.path.join(regional_root, bank_name)
        if not os.path.isdir(bank_dir):
            continue
        for fname in os.listdir(bank_dir):
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(bank_dir, fname)
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()
            cleaned  = clean_text(raw)
            date_str = extract_date_from_filename(fname)
            records.append({
                "doc_id":    fname.replace(".txt", ""),
                "source":    "regional_publication",
                "bank":      bank_name,
                "date":      date_str,
                "year":      int(date_str[:4]) if len(date_str) >= 4 and date_str[:4].isdigit() else None,
                "full_text": cleaned,
                "word_count":len(cleaned.split()),
            })

    if not records:
        print("[preprocess] No regional publication files found — skipping.")
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values(["bank", "date"]).reset_index(drop=True)
    out_path = os.path.join(PROCESSED, "regional_publications_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"[preprocess] Saved {len(df)} regional publication records → {out_path}")
    return df


#Entry Point

def main():
    fomc_df     = process_fomc_minutes()
    beige_df    = process_beige_books()
    regional_df = process_regional_publications()

    print("\n=== Preprocessing complete ===")
    print(f"  FOMC minutes:          {len(fomc_df)} records")
    print(f"  Beige Book districts:  {len(beige_df)} records")
    print(f"  Regional publications: {len(regional_df)} records")
    print(f"  All saved to: {PROCESSED}\n")


if __name__ == "__main__":
    main()
