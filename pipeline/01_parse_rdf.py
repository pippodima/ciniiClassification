import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import xml.etree.ElementTree as ET
import glob
import argparse
import os
import random
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Set, Optional
import pandas as pd
import html
import re
from datetime import datetime
from rdf_configs import NAMESPACES
from utils import save
from config import RAW_DIR, PROCESSED_DIR


# ============================================================================
# RDF PARSING
# ============================================================================

def extract_title(root: ET.Element) -> Optional[str]:
    """Extract title from RDF, preferring non-English versions."""
    titles = root.findall(".//dc:title", NAMESPACES)
    if not titles:
        return None

    # Prefer non-English titles (typically Japanese)
    for title_elem in titles:
        lang = title_elem.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
        if lang != "en":
            return title_elem.text

    # Fallback to first title
    return titles[0].text


def extract_language(root: ET.Element) -> Optional[str]:
    """
    Extract document language using a three-tier fallback:
      1. <dc:language> — publisher-declared, most authoritative
      2. xml:lang on the abstract <notation> element
      3. None (caller may run langdetect as last resort)

    NOTE: xml:lang on <dc:title> is NOT used — CiNii sometimes marks Japanese
          titles as xml:lang="en", making it unreliable.
    """
    # Tier 1: dc:language
    for e in root.iter():
        if localname(e.tag) == "language" and e.text:
            lang = e.text.strip().lower()
            if lang:
                return lang

    # Tier 2: xml:lang on the abstract notation
    XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
    for desc in root.iter():
        if localname(desc.tag) != "description":
            continue
        desc_type = None
        notation_lang = None
        for child in list(desc):
            ln = localname(child.tag)
            if ln == "type" and child.text:
                desc_type = child.text.strip().lower()
            elif ln == "notation":
                notation_lang = child.attrib.get(XML_LANG)
        if desc_type == "abstract" and notation_lang:
            return notation_lang.lower()

    # Fallback: any <notation> with xml:lang
    for e in root.iter():
        if localname(e.tag) == "notation":
            lang = e.attrib.get(XML_LANG)
            if lang:
                return lang.lower()

    return None


def extract_abstract(root: ET.Element) -> Optional[str]:
    # Prefer <description><type>abstract</type><notation>...</notation></description>
    for desc in root.iter():
        if localname(desc.tag) != "description":
            continue

        desc_type = None
        notation_text = None

        for child in list(desc):
            ln = localname(child.tag)
            if ln == "type" and child.text:
                desc_type = child.text.strip().lower()
            elif ln == "notation" and child.text:
                notation_text = child.text

        if desc_type == "abstract" and notation_text:
            return decode_jats_abstract(notation_text).strip()

    # Fallback: any <notation>
    for e in root.iter():
        if localname(e.tag) == "notation" and e.text:
            return decode_jats_abstract(e.text).strip()

    return None

def extract_dates(root: ET.Element):
    created = None
    modified = None
    for e in root.iter():
        ln = localname(e.tag)
        if ln == "createdAt" and e.text:
            created = e.text.strip()
        elif ln == "modifiedAt" and e.text:
            modified = e.text.strip()
    return created, modified


def extract_creators(root: ET.Element):
    names, affs = [], []

    for creator in root.iter():
        if localname(creator.tag) != "creator":
            continue

        for researcher in list(creator):
            if localname(researcher.tag) != "Researcher":
                continue

            name = None
            aff = None
            for child in list(researcher):
                ln = localname(child.tag)
                if ln == "name" and child.text:
                    name = child.text.strip()
                elif ln == "affiliationName" and child.text:
                    aff = child.text.strip()

            if name:
                names.append(name)
            if aff:
                affs.append(aff)

    return names or None, affs or None


def extract_doi(root):
    for id_ in safe_findall(root, ".//productIdentifier/identifier"):
        if "DOI" in id_.attrib.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}datatype",""):
            return safe_text(id_)
    return None


def extract_publication(root: ET.Element):
    pub = None
    for e in root.iter():
        if localname(e.tag) == "publication":
            pub = e
            break
    if pub is None:
        return (None,) * 7

    def child_text(pub_elem, child_local):
        for c in list(pub_elem):
            if localname(c.tag) == child_local and c.text:
                return c.text.strip()
        return None

    return (
        child_text(pub, "publicationName"),   # prism:publicationName
        child_text(pub, "publisher"),         # dc:publisher
        child_text(pub, "publicationDate"),   # prism:publicationDate
        child_text(pub, "volume"),            # prism:volume
        child_text(pub, "number"),            # prism:number
        child_text(pub, "startingPage"),      # prism:startingPage
        child_text(pub, "endingPage"),        # prism:endingPage
    )

def extract_topics(root: ET.Element):
    topics = []
    for e in root.iter():
        if localname(e.tag) != "topic":
            continue
        # attribute is dc:title
        t = e.attrib.get("{http://purl.org/dc/elements/1.1/}title")
        if t:
            topics.append(t)
    return topics or None

def extract_related(root: ET.Element):
    rel = []
    for e in root.iter():
        if localname(e.tag) == "relatedTitle" and e.text:
            rel.append(e.text.strip())
    return rel or None


def empty_row(file_path: str, error: str):
    return {name: None for name in TARGET_SCHEMA.names} | {"file": file_path, "error": error}

def parse_rdf(file_path: str) -> Dict:
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        title = extract_title(root)
        abstract = extract_abstract(root)
        language = extract_language(root)
        doi = extract_doi(root)
        authors, affs = extract_creators(root)

        journal, publisher, pub_date, volume, issue, start, end = extract_publication(root)
        topics = extract_topics(root)
        related = extract_related(root)
        created, modified = extract_dates(root)

        row = {name: None for name in TARGET_SCHEMA.names}
        row.update({
            "file": file_path,
            "title": title,
            "abstract": abstract,
            "language": language,
            "doi": doi,
            "authors": authors,
            "affiliations": affs,
            "journal": journal,
            "publisher": publisher,
            "publication_date": pub_date,
            "volume": volume,
            "issue": issue,
            "start_page": start,
            "end_page": end,
            "topics": topics,
            "related_titles": related,
            "created_at": created,
            "modified_at": modified,
            "error": None,
        })
        return row

    except ET.ParseError as e:
        return empty_row(file_path, f"ParseError: {e}")
    except Exception as e:
        return empty_row(file_path, f"{type(e).__name__}: {e}")
    
        
def safe_text(elem):
    return elem.text.strip() if elem is not None and elem.text else None


def safe_find(root, path):
    return root.find(path, NAMESPACES)


def safe_findall(root, path):
    return root.findall(path, NAMESPACES)


def decode_jats_abstract(text):
    """Decode and clean abstract text: unescape HTML entities, strip XML/JATS tags."""
    if not text:
        return None
    try:
        text = html.unescape(text)
        text = re.sub(r"<[^>]+>", " ", text)   # strip <jats:p>, <jats:bold>, etc.
        text = re.sub(r"\s+", " ", text).strip()
        return text if text else None
    except Exception:
        return text
    
def localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def find_first_by_localname(root: ET.Element, name: str) -> Optional[ET.Element]:
    for e in root.iter():
        if localname(e.tag) == name:
            return e
    return None


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def find_rdf_files(root_folder: str, pattern: str = "*.rdf") -> List[str]:
    """Recursively find all RDF files in a folder and its subfolders."""
    search_pattern = os.path.join(root_folder, "**", pattern)
    return glob.glob(search_pattern, recursive=True)


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def load_checkpoint(checkpoint_file: str) -> Set[int]:
    """Load completed batch numbers from checkpoint file."""
    if not os.path.exists(checkpoint_file):
        return set()

    with open(checkpoint_file, "r", encoding="utf-8") as f:
        return set(int(line.strip()) for line in f if line.strip().isdigit())


def append_to_checkpoint(checkpoint_file: str, batch_number: int) -> None:
    """Record that a batch was completed."""
    with open(checkpoint_file, "a", encoding="utf-8") as f:
        f.write(f"{batch_number}\n")


# ============================================================================
# PARQUET REPAIR + MERGE (SCHEMA-SAFE)
# ============================================================================

TARGET_SCHEMA = pa.schema([
    ("file", pa.string()),
    ("title", pa.string()),
    ("abstract", pa.string()),
    ("language", pa.string()),          # dc:language / xml:lang on abstract
    ("doi", pa.string()),
    ("authors", pa.list_(pa.string())),
    ("affiliations", pa.list_(pa.string())),
    ("journal", pa.string()),
    ("publisher", pa.string()),
    ("publication_date", pa.string()),
    ("volume", pa.string()),
    ("issue", pa.string()),
    ("start_page", pa.string()),
    ("end_page", pa.string()),
    ("topics", pa.list_(pa.string())),
    ("related_titles", pa.list_(pa.string())),
    ("created_at", pa.string()),
    ("modified_at", pa.string()),
    ("error", pa.string()),
])


def cast_table_to_schema(t: pa.Table, schema: pa.Schema) -> pa.Table:
    """
    Ensure all required columns exist and have the exact target types.
    Missing columns are created as all-null arrays with the correct type.
    """
    for name in schema.names:
        if name not in t.column_names:
            t = t.append_column(name, pa.array([None] * t.num_rows, type=schema.field(name).type))
    # Reorder to match schema order, then cast
    t = t.select(schema.names)
    return t.cast(schema, safe=False)


def repair_batch_parquet_files(output_folder: str) -> None:
    """
    One-time (or every run) repair:
    casts every batch parquet file to TARGET_SCHEMA so merging never fails.
    """
    batch_files = sorted(glob.glob(os.path.join(output_folder, "rdf_results_batch_*.parquet")))
    if not batch_files:
        return

    repaired = 0
    for f in batch_files:
        try:
            t = pq.read_table(f)
            t2 = cast_table_to_schema(t, TARGET_SCHEMA)
            pq.write_table(t2, f)  # overwrite
            repaired += 1
        except Exception as e:
            print(f"Warning: could not repair {f}: {e}")
    print(f"Repaired {repaired} batch parquet files to a consistent schema.")


def merge_batch_parquet_files(batch_files: List[str], final_output: str) -> None:
    """
    Streaming merge: write one batch at a time so RAM stays at one-batch peak
    regardless of total corpus size.  Each batch is cast to TARGET_SCHEMA before
    writing so the output schema is guaranteed stable.
    """
    tmp = final_output + ".tmp.parquet"
    writer = None
    try:
        for f in batch_files:
            t = pq.read_table(f)
            t = cast_table_to_schema(t, TARGET_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(tmp, TARGET_SCHEMA, compression="snappy")
            writer.write_table(t)
    except Exception:
        if writer:
            writer.close()
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    if writer:
        writer.close()
    os.replace(tmp, final_output)   # atomic: output is absent or complete, never partial


def cleanup_temporary_files(batch_files: List[str], checkpoint_file: str) -> None:
    """Remove temporary batch files and checkpoint file."""
    for batch_file in batch_files:
        try:
            os.remove(batch_file)
        except Exception as e:
            print(f"Warning: could not delete {batch_file}: {e}")

    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print("Removed checkpoint file.")
        except Exception as e:
            print(f"Warning: could not delete checkpoint file: {e}")


def merge_batches(output_folder: str, final_output: str, checkpoint_file: str) -> None:
    """Merge all batch Parquet files into one Parquet file, then clean up temporary files."""
    batch_files = sorted(glob.glob(os.path.join(output_folder, "rdf_results_batch_*.parquet")))
    if not batch_files:
        print("No batch parquet files found to merge.")
        return

    # Ensure any previously-written batches are schema-consistent
    repair_batch_parquet_files(output_folder)

    # Merge
    merge_batch_parquet_files(batch_files, final_output)
    print(f"Merged {len(batch_files)} batch parquet files into {final_output}")

    # Cleanup
    cleanup_temporary_files(batch_files, checkpoint_file)
    print("Temporary batch files deleted. Cleanup complete.")


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def select_sample_files(rdf_files: List[str], max_docs: Optional[int], random_state: int = 42) -> List[str]:
    """
    Select a random sample of files if max_docs is specified.

    NOTE: We sort rdf_files before sampling to keep sampling stable across runs.
    """
    if max_docs is None or max_docs >= len(rdf_files):
        return rdf_files

    random.seed(random_state)
    return random.sample(rdf_files, max_docs)


def process_batch(batch_files: List[str], output_file: str, checkpoint_file: str,
                  batch_number: int, workers: int = 1) -> None:
    """Process a single batch of RDF files, optionally in parallel."""
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # chunksize reduces IPC overhead: each worker gets 200 files per round-trip
            results = list(executor.map(parse_rdf, batch_files, chunksize=200))
    else:
        results = [parse_rdf(f) for f in batch_files]

    df = pd.DataFrame(results)

    # Force stable schema across batches (prevents 'null' Arrow columns)
    for col in TARGET_SCHEMA.names:
        if col not in df.columns:
            df[col] = None

    save(df, output_file)
    append_to_checkpoint(checkpoint_file, batch_number)
    print(f"Saved batch {batch_number} ({len(results)} files)")


def process_rdf_folder_batches(
    root_folder: str,
    batch_size: int = 1000,
    max_docs: Optional[int] = None,
    output_folder: str = "processed",
    workers: int = 1,
) -> None:
    """Process RDF files recursively in batches with crash recovery."""
    os.makedirs(output_folder, exist_ok=True)
    checkpoint_file = os.path.join(output_folder, "checkpoint.txt")
    completed_batches = load_checkpoint(checkpoint_file)

    # Deterministic order so batch numbers always map to the same files
    rdf_files = sorted(find_rdf_files(root_folder))
    rdf_files = select_sample_files(rdf_files, max_docs)

    total_files = len(rdf_files)
    print(f"Found {total_files} files. Processing in batches of {batch_size} "
          f"using {workers} worker(s)...")

    batch_number = 1
    for i in range(0, total_files, batch_size):
        if batch_number in completed_batches:
            print(f"Skipping batch {batch_number} (already processed)")
            batch_number += 1
            continue

        batch_files = rdf_files[i:i + batch_size]
        output_file = os.path.join(output_folder, f"rdf_results_batch_{batch_number}.parquet")
        process_batch(batch_files, output_file, checkpoint_file, batch_number, workers=workers)
        batch_number += 1

    # Allow caller (or env var) to override the final merged file path
    default_final = os.path.join(output_folder, "rdf_results_final.parquet")
    final_output = os.getenv("PARSED_OUTPUT", default_final)

    merge_batches(output_folder, final_output, checkpoint_file)
    print(f"All results saved to {final_output}")

    # ── Stage report ─────────────────────────────────────────────────────────
    import pandas as pd
    df_report = pd.read_parquet(final_output, columns=["title", "abstract", "error"])
    n = len(df_report)
    errors = df_report["error"].notna().sum()
    has_title = df_report["title"].notna().sum()
    has_abstract = df_report["abstract"].notna().sum()
    jats = df_report["abstract"].str.contains("<jats:", na=False).sum()
    print("\n" + "=" * 55)
    print("  PARSING STAGE REPORT")
    print("=" * 55)
    print(f"  Total rows           : {n:,}")
    print(f"  Parse errors         : {errors:,}  ({errors/n:.1%})")
    print(f"  Has title            : {has_title:,}  ({has_title/n:.1%})")
    print(f"  Has abstract         : {has_abstract:,}  ({has_abstract/n:.1%})")
    print(f"  JATS in abstract     : {jats:,}  ({jats/max(has_abstract,1):.1%}) — should be 0 after fix")
    print("=" * 55)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RDF parsing pipeline")
    parser.add_argument(
        "-max-document",
        "--max-document",
        type=int,
        default=None,
        help="Maximum number of documents to process (default: all)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=50000,
        help="Batch size for processing (default: 50000)",
    )
    parser.add_argument(
        "-r",
        "--root-folder",
        type=str,
        default=str(RAW_DIR),
        help="Root folder containing RDF files",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        type=str,
        default=str(PROCESSED_DIR),
        help="Output folder for processed files",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes for parsing (default: 1). "
             "Set to the number of CPU cores, e.g. --workers 16. "
             "Each worker parses files independently — data is identical to "
             "single-worker output, just faster.",
    )
    args = parser.parse_args()

    process_rdf_folder_batches(
        root_folder=args.root_folder,
        batch_size=args.batch_size,
        max_docs=args.max_document,
        output_folder=args.output_folder,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
