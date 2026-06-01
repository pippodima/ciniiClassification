import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import re
from html import unescape
import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm
from utils import load_df, save, _get_device, M2M_LANG_MAP
from config import RDF_PARSED, ENGLISH, OTHER_LANGS
from time import sleep
import torch
tqdm.pandas()

# Load model once globally (fast)
# DEVICE = _get_device()
# print(f"Using device: {DEVICE}")

# tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
# model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(DEVICE)

# ============================================================================
# TEXT CLEANING
# ============================================================================

def clean_abstract(text: str) -> str:
    """Clean and normalize abstract text by removing HTML, LaTeX, and formatting."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r'\\"([a-zA-Z])', lambda m: m.group(1), text)
    text = re.sub(r"\\'", "", text)
    text = re.sub(r"\$(.*?)\$", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\{(.*?)\}", r"\1", text)
    text = re.sub(r"\\(begin|end)\{.*?\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = text.replace("``", '"').replace("''", '"')
    text = re.sub(r"\s+", " ", text)


    return text.strip()


# ============================================================================
# LANGUAGE DETECTION
# ============================================================================

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return None

_CJK_RE = re.compile(
    r"[぀-ゟ"   # Hiragana
    r"゠-ヿ"    # Katakana
    r"一-鿿"    # CJK Unified Ideographs (kanji / hanzi)
    r"㐀-䶿]"   # CJK Extension A
)


def _cjk_ratio(text: str) -> float:
    """Fraction of characters that are CJK/Japanese."""
    if not text:
        return 0.0
    return len(_CJK_RE.findall(text)) / len(text)


def keep_only_english(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep rows where the resolved language is English AND the abstract
    contains fewer than 5% CJK characters.

    The CJK check catches papers where the publisher declared dc:language="en"
    but the abstract is actually written in Japanese — those slip through the
    langdetect fallback because Tier 1 (publisher tag) is trusted unconditionally.
    """
    print("Keeping only English papers")
    before = len(df)

    abs_col = "clean_abstract" if "clean_abstract" in df.columns else "abstract"

    # Tier 1: publisher / langdetect language label
    mask_lang = df["abstract_lang"] == "en"

    # Tier 2: CJK character ratio guard — catches Japanese abstracts labelled "en"
    cjk_ratios = df[abs_col].fillna("").apply(_cjk_ratio)
    mask_cjk   = cjk_ratios < 0.05   # less than 5% CJK chars → treat as English

    mask_en = mask_lang & mask_cjk
    n_cjk_rejected = int((mask_lang & ~mask_cjk).sum())

    df_en    = df[mask_en].copy()
    df_other = df[~mask_en].copy()

    after = len(df_en)
    print(f"  Kept {after:,} / {before:,} rows ({after / before:.1%})")
    if n_cjk_rejected:
        print(f"  CJK-ratio filter removed {n_cjk_rejected:,} additional rows "
              f"(declared 'en' but ≥5% CJK characters — Japanese leakage)")

    return df_en, df_other


# ============================================================================
# DOCUMENT CLASSIFICATION
# ============================================================================

DIAGNOSTIC_PATTERNS = re.compile(
    r'\blhd\s*(bolometer|flxloop|fpellet|interferometer|cxrs|ece|diagnostic|probe)\b|'
    r'^lhd\b|'
    r'\blhd\s*#?\d+',
    flags=re.IGNORECASE
)


def classify_document_type(title: str) -> str:
    if pd.isna(title):
        return "unknown"
    title_clean = str(title).strip().lower()
    return "diagnostic_report" if DIAGNOSTIC_PATTERNS.search(title_clean) else "scientific_paper"


# ============================================================================
# DATAFRAME OPERATIONS
# ============================================================================

def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    print("Dropping rows where title or abstract is empty or null")

    tqdm.pandas(desc="Checking for empty title/abstract")

    df_clean = df.dropna(subset=["title", "abstract"])

    df_clean = df_clean[
        df_clean["title"].progress_apply(lambda x: x.strip() != "") &
        df_clean["abstract"].progress_apply(lambda x: x.strip() != "")
    ]

    return df_clean


def drop_uninformative_abstracts(df: pd.DataFrame) -> pd.DataFrame:
    print("Dropping rows with non-informative abstracts")
    tqdm.pandas(desc="Filtering abstracts")

    META_KEYWORDS = [
        "article", "論文", "type:", "identifier:", "application/",
        "text/", "pdf", "html", "doi", "report"
    ]

    citation_pattern = re.compile(
    r".*(pp?\.?\s*[0-9一二三四五六七八九十]+[-‐~〜][0-9一二三四五六七八九十]+).*;?\s*(\d{4})$"
    )


    def is_informative(text: str) -> bool:
        t = text.strip().lower()

        # 1. Too short to be meaningful
        if len(t) < 100:
            return False

        # 2. Matches known metadata patterns
        for kw in META_KEYWORDS:
            if t.startswith(kw) or t == kw:
                return False

        # 3. Pure metadata (e.g. 論文(Article))
        if re.match(r".*\b(article|論文)\b.*$", t):
            if len(t) < 40:  # still metadata-level
                return False
                
        # 4. CITATION CHECK: The important new part
        if citation_pattern.match(t):
            return False

        # 5. No sentence markers -> very likely metadata
        if not any(p in t for p in [".", "。", "、", ","]):
            return False

        return True

    return df[df["abstract"].progress_apply(is_informative)]


def apply_clean_text_to_df(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning text")
    df["clean_abstract"] = df["abstract"].progress_apply(clean_abstract)
    return df


def _detect_language_safe(text: str) -> str:
    """Standalone function (module-level) so multiprocessing can pickle it."""
    try:
        from langdetect import detect
        from langdetect.lang_detect_exception import LangDetectException
        result = detect(str(text))
        return result if result else "unknown"
    except Exception:
        return "unknown"


def add_abstract_language(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a final language label per row using a fast-then-slow strategy:
      1. Use `language` column from parse (dc:language / xml:lang on abstract) — free, instant
      2. For rows still missing: parallel langdetect on clean_abstract

    At 3M scale, if ~90% have dc:language, only ~300k rows hit langdetect,
    and those are spread across all CPU cores.
    """
    import os
    from multiprocessing import Pool, cpu_count

    print("Resolving language labels")

    # Tier 1: publisher-declared language
    if "language" in df.columns:
        df["abstract_lang"] = df["language"].str.lower().str.strip()
    else:
        df["abstract_lang"] = None

    unknown_mask = df["abstract_lang"].isna()
    n_unknown = int(unknown_mask.sum())
    n_total = len(df)
    print(f"  Publisher-declared language : {n_total - n_unknown:,} / {n_total:,} rows")
    print(f"  Falling back to langdetect  : {n_unknown:,} rows")

    if n_unknown > 0:
        texts = df.loc[unknown_mask, "clean_abstract"].astype(str).tolist()
        n_workers = int(os.getenv("LANGDETECT_WORKERS", str(min(cpu_count(), 8))))
        print(f"  langdetect workers          : {n_workers}")

        if n_workers > 1 and n_unknown > 500:
            with Pool(processes=n_workers) as pool:
                results = list(
                    tqdm(pool.imap(_detect_language_safe, texts, chunksize=200),
                         total=n_unknown, desc="langdetect (parallel)")
                )
        else:
            results = [_detect_language_safe(t) for t in tqdm(texts, desc="langdetect")]

        df.loc[unknown_mask, "abstract_lang"] = results

    return df


def remove_empty_lang(df: pd.DataFrame, title_col: str = 'title_lang',
                      abstract_col: str = 'abstract_lang') -> pd.DataFrame:
    return df.dropna(subset=[title_col, abstract_col])


def classify_documents(df: pd.DataFrame) -> pd.DataFrame:
    print("Classifying document types")
    df["type"] = df["title"].progress_apply(classify_document_type)
    df_diag = df[df["type"] == "diagnostic_report"].copy()
    df_sci = df[df["type"] == "scientific_paper"].copy()

    print(f"Diagnostic reports: {len(df_diag)} | Scientific papers: {len(df_sci)}")
    return df_sci


# ============================================================================
# I/O OPERATIONS
# ============================================================================



def drop_boilerplate_only_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows whose titles and abstracts contain only boilerplate words like
    'type:text', 'application/pdf', 'Departmental Bulletin Paper', etc.
    """
    boilerplate_patterns = re.compile(
        r"^(type|application|departmental|bulletin|text|論文|記事|表紙|裏表紙|奥付|目次)\b",
        flags=re.IGNORECASE
    )

    mask = df["clean_abstract"].fillna("").str.match(boilerplate_patterns)
    dropped = df[mask]
    if len(dropped) > 0:
        print(f"🧹 Dropping {len(dropped)} boilerplate-only rows (no semantic content)")
    return df[~mask].copy()


def normalize_lang(code):
    if code is None:
        return None
    return M2M_LANG_MAP.get(code.lower(), None)


def translate_batch_m2m(text_list, src_lang, batch_size=4):
    """Translate list of texts from src_lang → English using GPU M2M100."""

    def safe_generate(**kwargs):
        try:
            return model.generate(**kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            sleep(1)
            return model.generate(**kwargs)


    if src_lang is None:
        return text_list  # identity: skip

    tokenizer.src_lang = src_lang
    results = []

    for i in tqdm(range(0, len(text_list), batch_size), desc=f"{src_lang} → en"):
        batch = text_list[i:i + batch_size]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(DEVICE)

        generated = safe_generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id("en"),
            num_beams=1,
            max_length=256
        )

        outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)
        results.extend(outputs)

    return results


def translate_to_eng(df):
    print("Translating to English using local M2M100 model")

    df["title_en"] = df["title"].astype(str)
    df["abstract_en"] = df["clean_abstract"].astype(str)

    # ---------- TITLES ----------
    mask_t = df["title_lang"] != "en"
    langs_t = df.loc[mask_t, "title_lang"].tolist()

    for lang in set(langs_t):
        norm = normalize_lang(lang)

        # skip unsupported languages
        if norm is None or norm == "en":
            continue

        idx = df.index[(df["title_lang"] == lang)]
        texts = df.loc[idx, "title"].tolist()

        translated = translate_batch_m2m(texts, norm)
        df.loc[idx, "title_en"] = translated

    # ---------- ABSTRACTS ----------
    mask_a = df["abstract_lang"] != "en"
    langs_a = df.loc[mask_a, "abstract_lang"].tolist()

    for lang in set(langs_a):
        norm = normalize_lang(lang)

        # skip unsupported languages
        if norm is None or norm == "en":
            continue

        idx = df.index[(df["abstract_lang"] == lang)]
        texts = df.loc[idx, "clean_abstract"].tolist()

        translated = translate_batch_m2m(texts, norm)
        df.loc[idx, "abstract_en"] = translated
        torch.cuda.empty_cache()

    return df


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    import os
    _parsed_input  = os.getenv("PARSED_INPUT",  str(RDF_PARSED))
    _english_out   = os.getenv("ENGLISH_OUTPUT", str(ENGLISH))
    _other_out     = os.getenv("OTHER_OUTPUT",   str(OTHER_LANGS))

    # Push the "title AND abstract must be non-null" filter down into the parquet
    # reader so we never load the 48M empty rows into RAM at all.
    # 71.5M total → 23.1M with both fields → saves ~30 GB of pandas RAM.
    import pyarrow.dataset as _ds
    print("  Pre-filtering: dropping rows without title or abstract at read time...")
    _dataset = _ds.dataset(_parsed_input, format="parquet")
    _filt    = (_ds.field("title").is_valid() &
                _ds.field("abstract").is_valid() &
                (_ds.field("title")    != "") &
                (_ds.field("abstract") != ""))
    n_parsed = _ds.dataset(_parsed_input, format="parquet").count_rows()
    df = _dataset.to_table(filter=_filt).to_pandas()
    print(f"  Loaded: {len(df):,} rows  (pre-filtered from {n_parsed:,} total)")

    # drop_empty_rows is now a no-op but kept for safety
    df = drop_empty_rows(df)
    print(f"  After drop_empty     : {len(df):,}  (dropped {n_parsed - len(df):,})")

    n = len(df)
    df = drop_uninformative_abstracts(df)
    print(f"  After drop_uninform  : {len(df):,}  (dropped {n - len(df):,})")

    df = apply_clean_text_to_df(df)
    jats_remaining = df["clean_abstract"].str.contains("<jats:", na=False).sum()
    print(f"  clean_abstract JATS  : {jats_remaining}  (should be 0)")

    df = add_abstract_language(df)

    n = len(df)
    df, df_other_languages = keep_only_english(df)
    print(f"  After keep_english   : {len(df):,}  (dropped {n - len(df):,} non-English)")

    df_sci = classify_documents(df)
    print(f"  After classify_docs  : {len(df_sci):,} scientific papers")

    # ── Stage report ────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  TEXT PROCESSING STAGE REPORT")
    print("=" * 55)
    print(f"  Input (parsed)       : {n_parsed:,}")
    print(f"  Output (scientific EN): {len(df_sci):,}  ({len(df_sci)/n_parsed:.1%} of input)")
    print(f"  Non-English saved    : {len(df_other_languages):,}")
    lang_dist = df_sci["abstract_lang"].value_counts().head(5).to_dict()
    print(f"  Top langs (final)    : {lang_dist}")
    print(f"  Cols in output       : {list(df_sci.columns)}")
    print("=" * 55)

    save(df_sci, _english_out)
    save(df_other_languages, _other_out)



if __name__ == "__main__":
    main()