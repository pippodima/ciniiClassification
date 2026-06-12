"""
search/index_meili.py
=====================
Index the classified CiNii corpus into Meilisearch for faceted, category-aware
search. Streams the parquet in batches (never loads the whole corpus into RAM),
configures the index for LCC faceting + trust filtering, and prints the
search-only API key for the web UI.

Prereqies:
    pip install -r search/requirements.txt
    # run Meilisearch locally (Homebrew or Docker), with a master key:
    meilisearch --master-key "MASTER_KEY"            # brew install meilisearch
    # or: docker run -p 7700:7700 getmeili/meilisearch --master-key MASTER_KEY

Usage:
    MEILI_MASTER_KEY=MASTER_KEY python search/index_meili.py \
        --source classified/classified_v3_300k.parquet \
        --index  cinii

    # smaller demo subset:
    ... --limit 200000

Document fields indexed (lean — full_text/embeddings/affiliations dropped):
    id, title, abstract, authors, journal, publisher, year, doi,
    pred_lcc_main, pred_lcc, pred_lcc_div, lcc_label, lcc.{lvl0,lvl1,lvl2},
    conf_div, pred_centroid_sim, trust_tier
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lcc_names as L

READ_COLS = [
    "title", "clean_abstract", "abstract", "authors", "journal", "publisher",
    "publication_date", "doi", "pred_lcc_main", "pred_lcc", "pred_lcc_div",
    "conf_div", "pred_centroid_sim",
]

_YEAR = re.compile(r"(1[5-9]\d{2}|20\d{2})")


def _year(v) -> int | None:
    if v is None:
        return None
    m = _YEAR.search(str(v))
    return int(m.group(1)) if m else None


def _authors(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)) or hasattr(v, "__array__"):
        return "; ".join(str(a) for a in list(v) if a is not None and str(a) != "")
    return str(v)


def _trust_tier(sim) -> str:
    if sim is None:
        return "unknown"
    s = float(sim)
    if s < 0.40:
        return "reject"
    if s < 0.57:
        return "low"
    if s < 0.67:
        return "medium"
    return "trust"


def _clean_str(v) -> str:
    return "" if v is None else str(v).strip()


def _wait(client, task):
    """Block until a Meilisearch task finishes; robust across client versions."""
    uid = getattr(task, "task_uid", None)
    if uid is None:
        uid = getattr(task, "uid", None)
    if uid is None and isinstance(task, dict):
        uid = task.get("taskUid") or task.get("uid")
    while True:
        t = client.get_task(uid)
        status = t.status if hasattr(t, "status") else t["status"]
        if status in ("succeeded", "failed", "canceled"):
            if status != "succeeded":
                err = getattr(t, "error", None) or (t.get("error") if isinstance(t, dict) else None)
                raise RuntimeError(f"Meilisearch task {uid} {status}: {err}")
            return
        time.sleep(0.2)


SETTINGS = {
    "searchableAttributes": ["title", "abstract", "authors", "journal"],
    "filterableAttributes": [
        "pred_lcc_main", "pred_lcc", "pred_lcc_div",
        "lcc.lvl0", "lcc.lvl1", "lcc.lvl2",
        "journal", "publisher", "year", "trust_tier",
        "pred_centroid_sim", "conf_div",
    ],
    "sortableAttributes": ["year", "pred_centroid_sim", "conf_div"],
    "displayedAttributes": [
        "id", "title", "abstract", "authors", "journal", "publisher", "year",
        "doi", "pred_lcc_main", "pred_lcc", "pred_lcc_div", "lcc_label",
        "conf_div", "pred_centroid_sim", "trust_tier",
    ],
    "rankingRules": ["words", "typo", "proximity", "attribute", "sort", "exactness"],
    "pagination": {"maxTotalHits": 100000},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="classified/classified_v3_300k.parquet")
    ap.add_argument("--index", default="cinii")
    ap.add_argument("--host", default=os.getenv("MEILI_HTTP_ADDR", "http://localhost:7700"))
    ap.add_argument("--key", default=os.getenv("MEILI_MASTER_KEY", ""))
    ap.add_argument("--batch-size", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=None, help="cap rows (demo subset)")
    args = ap.parse_args()

    import meilisearch

    client = meilisearch.Client(args.host, args.key or None)
    try:
        client.health()
    except Exception as e:
        sys.exit(f"✗ Meilisearch not reachable at {args.host} ({e}). Start it first.")

    # (re)create index with integer primary key
    try:
        _wait(client, client.delete_index(args.index))
        print(f"  dropped existing index '{args.index}'")
    except Exception:
        pass
    _wait(client, client.create_index(args.index, {"primaryKey": "id"}))
    index = client.index(args.index)
    _wait(client, index.update_settings(SETTINGS))
    print(f"  index '{args.index}' created + configured")

    pf = pq.ParquetFile(args.source)
    total = pf.metadata.num_rows if args.limit is None else min(args.limit, pf.metadata.num_rows)
    print(f"  indexing up to {total:,} docs from {args.source}")

    buf, doc_id, done, skipped = [], 0, 0, 0
    stop = False
    for rb in pf.iter_batches(batch_size=args.batch_size, columns=READ_COLS):
        rows = rb.to_pylist()
        for r in rows:
            title = _clean_str(r.get("title"))
            if not title:
                skipped += 1
                continue
            main, sub, div = (_clean_str(r.get("pred_lcc_main")),
                              _clean_str(r.get("pred_lcc")),
                              _clean_str(r.get("pred_lcc_div")))
            abstract = _clean_str(r.get("clean_abstract")) or _clean_str(r.get("abstract"))
            doc = {
                "id": doc_id,
                "title": title,
                "abstract": abstract,
                "authors": _authors(r.get("authors")),
                "journal": _clean_str(r.get("journal")),
                "publisher": _clean_str(r.get("publisher")),
                "year": _year(r.get("publication_date")),
                "doi": _clean_str(r.get("doi")),
                "pred_lcc_main": main,
                "pred_lcc": sub,
                "pred_lcc_div": div,
                "lcc_label": L.sub_label(sub),
                "conf_div": round(float(r["conf_div"]), 4) if r.get("conf_div") is not None else None,
                "pred_centroid_sim": round(float(r["pred_centroid_sim"]), 4) if r.get("pred_centroid_sim") is not None else None,
                "trust_tier": _trust_tier(r.get("pred_centroid_sim")),
            }
            lv = L.levels(main, sub, div)
            if lv:
                doc["lcc"] = lv
            buf.append(doc)
            doc_id += 1
            if args.limit is not None and doc_id >= args.limit:
                stop = True
                break

        if len(buf) >= args.batch_size:
            _wait(client, index.add_documents(buf, primary_key="id"))
            done += len(buf); buf = []
            print(f"    indexed {done:,} / {total:,}", end="\r", flush=True)
        if stop:
            break

    if buf:
        _wait(client, index.add_documents(buf, primary_key="id"))
        done += len(buf)
    print(f"\n  ✅ indexed {done:,} docs (skipped {skipped:,} without title)")

    # surface the search-only key for the web UI
    try:
        keys = client.get_keys()
        items = keys.results if hasattr(keys, "results") else keys["results"]
        for k in items:
            actions = getattr(k, "actions", None) or (k.get("actions") if isinstance(k, dict) else [])
            name = getattr(k, "name", None) or (k.get("name") if isinstance(k, dict) else "")
            if "search" in actions and "*" not in actions:
                val = getattr(k, "key", None) or k.get("key")
                print(f"\n  Search-only API key (put in search/web/index.html):\n    {val}")
                break
    except Exception:
        print("\n  (could not list keys — if running without a master key, use '' in the UI)")

    print(f"\n  Next: open search/web/index.html (set HOST + SEARCH_KEY at the top).")


if __name__ == "__main__":
    main()
