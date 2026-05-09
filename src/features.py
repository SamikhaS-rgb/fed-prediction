"""
features.py
-----------
All preprocessing and feature engineering for the FOMC prediction project.

Produces:
  data/processed/fomc_statements_clean.csv   — cleaned statement text + metadata
  data/processed/market_features.csv         — 21 market features per FOMC meeting
  data/processed/tone_scores.csv             — hawkishness / uncertainty / concern scores
  data/processed/topic_distributions.csv     — 8-topic LDA weight vectors
  data/processed/embeddings.npy             — Sentence-BERT 384-dim embeddings
  data/processed/embedding_doc_ids.csv       — row-to-doc_id mapping for embeddings
  data/processed/master_dataset.csv          — all features + targets combined

HOW TO RUN:
    python src/features.py
"""

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# NLP
import nltk
from sentence_transformers import SentenceTransformer
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords as nltk_stopwords

nltk.download("punkt",     quiet=True)
nltk.download("stopwords", quiet=True)

warnings.filterwarnings("ignore")

#Paths
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_FOMC  = os.path.join(ROOT, "data", "raw", "fomc")
RAW_MKT   = os.path.join(ROOT, "data", "raw", "market")
RAW_LEX   = os.path.join(ROOT, "data", "raw", "lexicon")
PROCESSED = os.path.join(ROOT, "data", "processed")
os.makedirs(PROCESSED, exist_ok=True)

SBERT_MODEL  = "all-MiniLM-L6-v2"    # 384-dim, fast, free
N_TOPICS     = 8                       # per proposal
RANDOM_STATE = 42

CHAIR_ERAS = {
    (1994, 2006): "greenspan",
    (2006, 2014): "bernanke",
    (2014, 2018): "yellen",
    (2018, 2099): "powell",
}

RECESSION_PERIODS = [
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]

HIKING_CYCLES = [
    ("1999-06-30", "2000-05-16"),
    ("2004-06-30", "2006-06-29"),
    ("2015-12-16", "2018-12-19"),
    ("2022-03-16", "2023-07-26"),
]


# PART 1: Clean FOMC statement text

BOILERPLATE_RE = re.compile(
    r"(page \d+ of \d+|for release at|embargo|https?://\S+|\*{3,}|\[\d+\])",
    re.IGNORECASE,
)


def clean_statement(raw: str) -> str:
    """Strips scraper metadata headers and boilerplate from raw statement text."""
    # Remove scraper-added header
    raw = re.sub(r"^DATE:.*?\n\nURL:.*?\n\n", "", raw, flags=re.DOTALL)
    # Fix common encoding artifacts
    for old, new in [("\u2019", "'"), ("\u2014", "--"), ("\u201c", '"'), ("\u201d", '"')]:
        raw = raw.replace(old, new)
    raw = BOILERPLATE_RE.sub(" ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    raw = re.sub(r" {2,}", " ", raw)
    return raw.strip()


def load_and_clean_statements() -> pd.DataFrame:
    """Reads all raw FOMC statement .txt files and returns a clean DataFrame."""
    files = sorted(glob.glob(os.path.join(RAW_FOMC, "fomc_statement_*.txt")))
    if not files:
        raise FileNotFoundError(
            f"No statement files found in {RAW_FOMC}. Run scraper.py first."
        )

    records = []
    for fpath in files:
        fname = os.path.basename(fpath)
        match = re.search(r"(\d{8})", fname)
        date_str = match.group(1) if match else "unknown"

        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

        text = clean_statement(raw)
        if len(text.split()) < 20:   
            continue

        year  = int(date_str[:4])   if date_str != "unknown" else None
        month = int(date_str[4:6])  if date_str != "unknown" else None
        date  = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")

        chair = "unknown"
        for (s, e), name in CHAIR_ERAS.items():
            if year and s <= year < e:
                chair = name

        is_crisis = any(
            pd.to_datetime(s) <= date <= pd.to_datetime(e)
            for s, e in RECESSION_PERIODS
        )
        is_hiking = any(
            pd.to_datetime(s) <= date <= pd.to_datetime(e)
            for s, e in HIKING_CYCLES
        )
        is_post_covid = date >= pd.to_datetime("2020-03-01")

        records.append({
            "doc_id":        fname.replace(".txt", ""),
            "date":          date,
            "date_str":      date_str,
            "year":          year,
            "month":         month,
            "quarter":       (month - 1) // 3 + 1 if month else None,
            "chair_era":     chair,
            "is_crisis":     int(is_crisis),
            "is_hiking":     int(is_hiking),
            "is_post_covid": int(is_post_covid),
            "text":          text,
            "word_count":    len(text.split()),
        })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    chair_dummies = pd.get_dummies(df["chair_era"], prefix="chair").astype(int)
    quarter_dummies = pd.get_dummies(df["quarter"].astype(str), prefix="q").astype(int)
    df = pd.concat([df, chair_dummies, quarter_dummies], axis=1)

    out = os.path.join(PROCESSED, "fomc_statements_clean.csv")
    df.to_csv(out, index=False)
    print(f"[features] Saved fomc_statements_clean.csv  ({len(df)} statements)")
    return df


# PART 2: Market feature engineering


MARKET_FILES = {
    "treasury_2yr.csv":       "treasury_2yr",
    "treasury_10yr.csv":      "treasury_10yr",
    "move_index.csv":         "move_index",
    "inflation_breakeven.csv":"breakeven_5yr",
    "hy_spread.csv":          "hy_spread",
    "vix.csv":                "vix",
    "dxy.csv":                "dxy",
}

WINDOW_DAYS = 42 


def load_market_series() -> dict[str, pd.Series]:
    """Loads all market CSVs into a dict of date-indexed Series."""
    series = {}
    for filename, col in MARKET_FILES.items():
        fpath = os.path.join(RAW_MKT, filename)
        if not os.path.exists(fpath):
            print(f"  [warn] {filename} not found — run market_data.py first")
            continue
        df = pd.read_csv(fpath, parse_dates=["date"])
        df = df.dropna().sort_values("date").set_index("date")
        series[col] = df[col].astype(float)
    return series


def compute_window_features(series: pd.Series, meeting_date: pd.Timestamp,
                             window_days: int = WINDOW_DAYS) -> dict:
    """
    Computes level, trend (OLS slope), and volatility (std dev) for a
    market series over the window_days ending the day before meeting_date.
    Returns {name_level, name_trend, name_vol}.
    """
    end   = meeting_date - pd.Timedelta(days=1)
    start = meeting_date - pd.Timedelta(days=window_days)
    window = series.loc[start:end].dropna()

    if len(window) < 5:
        return {}

    name = series.name
    level = float(window.mean())
    vol   = float(window.std())

    # OLS slope (trend)
    x = np.arange(len(window), dtype=float)
    x -= x.mean()
    y  = window.values - window.values.mean()
    trend = float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) > 0 else 0.0

    return {
        f"{name}_level": round(level, 6),
        f"{name}_trend": round(trend, 6),
        f"{name}_vol":   round(vol,   6),
    }


def build_market_features(stmt_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each FOMC meeting, computes level/trend/vol for all 7 market variables
    over the 6-week pre-meeting window.

    Returns DataFrame with ~21 market features per meeting, min-max scaled.
    """
    print("\n[features] Building market features (6-week pre-meeting windows)...")
    market = load_market_series()

    rows = []
    for _, row in tqdm(stmt_df.iterrows(), total=len(stmt_df), desc="Market features"):
        date = row["date"]
        feat = {"date_str": row["date_str"]}

        for col, series in market.items():
            series.name = col
            window_feat = compute_window_features(series, date)
            feat.update(window_feat)

        rows.append(feat)

    df = pd.DataFrame(rows)


    feature_cols = [c for c in df.columns if c != "date_str"]
    valid_feat   = [c for c in feature_cols if df[c].notna().sum() > 0]

    train_mask = df["date_str"].str[:4].astype(int) <= 2019
    scaler = MinMaxScaler()
    df_filled = df[valid_feat].fillna(df[valid_feat].median())
    scaler.fit(df_filled[train_mask])
    df[valid_feat] = scaler.transform(df_filled)

    out = os.path.join(PROCESSED, "market_features.csv")
    df.to_csv(out, index=False)
    print(f"[features] Saved market_features.csv  ({len(df)} rows × {len(df.columns)} cols)")
    return df

# PART 3: Tone scores (hawkishness, uncertainty, concern)


HAWKISH_WORDS = {
    "tighten", "tightening", "tightened", "restrict", "restrictive",
    "raise", "raised", "raising", "hike", "hikes", "hiking",
    "increase", "increases", "increased", "increasing",
    "vigilant", "vigilance", "inflation", "inflationary", "overheat",
    "overheating", "firming", "firm", "elevated",
}


UNCERTAINTY_SEED = {
    "approximately", "around", "assume", "assumed", "could", "depend",
    "doubt", "estimate", "estimated", "expect", "expected", "fluctuate",
    "generally", "if", "indicate", "likelihood", "likely", "may", "might",
    "mostly", "nearly", "occasional", "often", "perhaps", "possible",
    "possibly", "potential", "potentially", "predict", "probable", "probably",
    "range", "roughly", "seem", "should", "sometimes", "suggest", "tentative",
    "uncertain", "uncertainty", "unclear", "unexpected", "unpredictable",
    "unusual", "usually", "variable", "volatile", "volatility", "would",
}

CONCERN_WORDS = {
    "concern", "concerns", "concerning", "downside", "deteriorate",
    "deteriorating", "deterioration", "weaken", "weakening", "weakness",
    "risk", "risks", "risky", "stress", "stressful", "tension", "tensions",
    "slowdown", "contraction", "recession", "decline", "declining",
    "adverse", "adversely", "negative", "negatively", "turbulence",
    "uncertainty", "volatile", "volatility", "disruption", "headwind",
}


def load_lm_words() -> tuple[set, set]:
    """
    Loads Loughran-McDonald uncertainty and negative word lists.
    Falls back to hardcoded seed sets if the CSV is not present.

    Download the full LM dictionary from: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
    Save as: data/raw/lexicon/lm_dictionary.csv
    """
    lm_path = os.path.join(RAW_LEX, "lm_dictionary.csv")
    if os.path.exists(lm_path):
        df = pd.read_csv(lm_path)
        df["Word"] = df["Word"].str.lower()
        unc  = set(df[df["Uncertainty"] > 0]["Word"].tolist()) if "Uncertainty" in df.columns else UNCERTAINTY_SEED
        neg  = set(df[df["Negative"]    > 0]["Word"].tolist()) if "Negative"    in df.columns else CONCERN_WORDS
        print(f"[features] Loaded LM dictionary: {len(unc)} uncertainty, {len(neg)} negative words")
        return unc, neg
    else:
        print("[features] LM CSV not found — using hardcoded word lists")
        print("           For best results, download from sraf.nd.edu")
        return UNCERTAINTY_SEED, CONCERN_WORDS


def compute_tone_scores(text: str, hawkish_words: set,
                         uncertainty_words: set, concern_words: set) -> dict:
    """Returns hawkishness, uncertainty, and concern scores (per 100 tokens)."""
    if not text or len(text.strip()) == 0:
        return {"hawkishness": 0.0, "uncertainty": 0.0, "concern": 0.0}

    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    n = len(tokens)
    if n == 0:
        return {"hawkishness": 0.0, "uncertainty": 0.0, "concern": 0.0}

    return {
        "hawkishness": round(sum(1 for t in tokens if t in hawkish_words) / n * 100, 4),
        "uncertainty": round(sum(1 for t in tokens if t in uncertainty_words) / n * 100, 4),
        "concern":     round(sum(1 for t in tokens if t in concern_words) / n * 100, 4),
    }


def build_tone_scores(stmt_df: pd.DataFrame) -> pd.DataFrame:
    """Computes tone scores for all statements and saves to CSV."""
    print("\n[features] Computing tone scores...")
    unc_words, neg_words = load_lm_words()

    records = []
    for _, row in tqdm(stmt_df.iterrows(), total=len(stmt_df), desc="Tone scores"):
        scores = compute_tone_scores(row["text"], HAWKISH_WORDS, unc_words, neg_words)
        records.append({"date_str": row["date_str"], **scores})

    df = pd.DataFrame(records)
    out = os.path.join(PROCESSED, "tone_scores.csv")
    df.to_csv(out, index=False)
    print(f"[features] Saved tone_scores.csv")
    return df


# PART 4: LDA topic distributions
CUSTOM_STOPWORDS = set(nltk_stopwords.words("english")) | STOPWORDS | {
    "said", "also", "would", "could", "may", "one", "two", "three",
    "percent", "year", "month", "quarter", "committee", "federal",
    "reserve", "bank", "noted", "indicated", "meeting", "members",
    "participants", "staff", "rdf", "obj", "endobj", "xmp", "stream", "xref", "pdf", "xmlns",
    "metadata", "adobe", "startxref",
}


def tokenize_for_lda(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [t for t in tokens if t not in CUSTOM_STOPWORDS]


def build_lda_topics(train_texts: list[str], all_texts: list[str],
                     n_topics: int = N_TOPICS) -> pd.DataFrame:
    """
    Trains LDA exclusively on train_texts (1994–2019 split).
    Applies the fitted model to all_texts (full corpus).
    Returns topic weight DataFrame — one row per document.

    The optimal number of topics (default 8) is chosen based on the proposal.
    In practice, you can sweep N_TOPICS from 5–12 and compare perplexity.
    """
    print(f"\n[features] Training LDA ({n_topics} topics) on training set only...")

    train_tokens = [tokenize_for_lda(t) for t in train_texts]
    dictionary   = corpora.Dictionary(train_tokens)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    train_corpus = [dictionary.doc2bow(t) for t in train_tokens]

    lda = LdaModel(
        corpus=train_corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=RANDOM_STATE,
        passes=15,
        alpha="auto",
        eta="auto",
    )
    lda.save(os.path.join(PROCESSED, "lda_model"))

    # Print top words per topic for interpretability
    print("[features] Top 5 words per LDA topic:")
    for i in range(n_topics):
        words = ", ".join(w for w, _ in lda.show_topic(i, topn=5))
        print(f"  Topic {i:02d}: {words}")

    # Apply to full corpus (no refitting — prevents leakage)
    def get_topic_vec(text: str) -> list[float]:
        tokens = tokenize_for_lda(text)
        bow    = dictionary.doc2bow(tokens)
        topics = dict(lda.get_document_topics(bow, minimum_probability=0.0))
        return [topics.get(i, 0.0) for i in range(n_topics)]

    print("[features] Computing topic distributions for all statements...")
    topic_vecs = [get_topic_vec(t) for t in tqdm(all_texts, desc="LDA inference")]
    topic_df   = pd.DataFrame(topic_vecs, columns=[f"topic_{i}" for i in range(n_topics)])
    return topic_df



# PART 5: Sentence-BERT embeddings
def build_embeddings(texts: list[str], train_mask: np.ndarray = None) -> np.ndarray:
    """
    Encodes all statements into 384-dim Sentence-BERT vectors.
    Returns shape (N, 384). Saves full array to embeddings.npy.
    Also saves the first 10 PCA components to processed/ for use as stretch-goal targets.
    """
    print("\n[features] Loading Sentence-BERT model...")
    model = SentenceTransformer(SBERT_MODEL)
    print("[features] Encoding statements (may take a few minutes)...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize → dot product = cosine sim
    )
    np.save(os.path.join(PROCESSED, "embeddings.npy"), embeddings)

    if train_mask is None:
        train_mask = np.ones(len(embeddings), dtype=bool)


    print("[features] Running PCA (384 → 10 dims) fit on train set only...")
    pca = PCA(n_components=10, random_state=RANDOM_STATE)
    pca.fit(embeddings[train_mask])
    pca_10 = pca.transform(embeddings)
    pd.DataFrame(pca_10, columns=[f"sbert_pca_{i}" for i in range(10)]).to_csv(
        os.path.join(PROCESSED, "sbert_pca.csv"), index=False
    )
    print(f"[features] Saved embeddings.npy  shape={embeddings.shape}")
    print(f"[features] Saved sbert_pca.csv   explained_var={pca.explained_variance_ratio_.sum():.3f}")
    return embeddings


# MAIN: orchestrate all steps and assemble master_dataset.csv
def build_features():
    print("=" * 60)
    print(" FOMC Prediction — Feature Engineering Pipeline")
    print("=" * 60)

    stmt_df = load_and_clean_statements()
    texts   = stmt_df["text"].fillna("").tolist()

    train_mask = stmt_df["year"] <= 2019
    train_texts = [t for t, m in zip(texts, train_mask) if m]
    print(f"\n[features] Train: {train_mask.sum()} meetings (1994–2019) | "
          f"Test: {(~train_mask).sum()} meetings (2020–2024)")

    market_df = build_market_features(stmt_df)

    tone_df = build_tone_scores(stmt_df)

    topic_df = build_lda_topics(train_texts, texts, n_topics=N_TOPICS)
    topic_df.insert(0, "date_str", stmt_df["date_str"].values)
    topic_df.to_csv(os.path.join(PROCESSED, "topic_distributions.csv"), index=False)

    embeddings = build_embeddings(texts, train_mask=train_mask.values)
    stmt_df[["doc_id", "date_str"]].to_csv(
        os.path.join(PROCESSED, "embedding_doc_ids.csv"), index=False
    )

    sbert_pca = pd.read_csv(os.path.join(PROCESSED, "sbert_pca.csv"))

    master = (
        stmt_df
        .drop(columns=["text"])             # don't put full text in the wide CSV
        .merge(market_df, on="date_str",    how="left")
        .merge(tone_df,   on="date_str",    how="left")
        .merge(topic_df,  on="date_str",    how="left")
    )
   
    for col in sbert_pca.columns:
        master[col] = sbert_pca[col].values

    out = os.path.join(PROCESSED, "master_dataset.csv")
    master.to_csv(out, index=False)
    print(f"\n[features] Saved master_dataset.csv  ({len(master)} rows × {len(master.columns)} cols)")
    print("[features] Pipeline complete.")
    return master


if __name__ == "__main__":
    build_features()
