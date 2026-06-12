# CiNii LCC Explorer

A local, faceted search engine over the classified corpus. Full-text search
(title / abstract / authors / journal) plus filtering by the validated LCC
categories, journal, publisher, year, and the `pred_centroid_sim` **trust tier**.

```
search/
  make_sample.py     # export a small representative slice (run on the server)
  index_meili.py     # stream a parquet → Meilisearch
  lcc_names.py       # LCC code → human-readable names (facet labels)
  requirements.txt   # indexer deps (UI is CDN-based, no build)
  web/index.html     # self-contained InstantSearch explorer
```

## Recommended: demo on a laptop (no Meilisearch on the server)

Meilisearch only has to run where the *index* lives. For a demo, export a small
representative sample on the server and do everything else locally.

```bash
# on the SERVER — small, quick-to-download slice (no embeddings in the parquet)
python search/make_sample.py \
    --source classified/classified_v3_300k.parquet \
    --out cinii_sample.parquet --n 150000 --stratify

# download it
scp server:/path/to/cinii_sample.parquet .

# locally: steps 1–3 below, but point the indexer at the sample
python search/index_meili.py --source cinii_sample.parquet --index cinii
```

A 150k stratified sample indexes in ~2 min, exercises every facet, and is a
faithful demo. The full corpus works identically — just bigger.

## 1. Run Meilisearch locally

```bash
# macOS
brew install meilisearch
meilisearch --master-key "DEV_MASTER_KEY"

# or Docker
docker run -it --rm -p 7700:7700 getmeili/meilisearch:v1.10 --master-key DEV_MASTER_KEY
```

## 2. Index the corpus

```bash
pip install -r search/requirements.txt

MEILI_MASTER_KEY=DEV_MASTER_KEY python search/index_meili.py \
    --source classified/classified_v3_300k.parquet \
    --index  cinii

# smaller demo subset:
MEILI_MASTER_KEY=DEV_MASTER_KEY python search/index_meili.py --limit 200000
```

The indexer streams in batches (RAM-safe), configures faceting/sorting, and
prints the **search-only API key** at the end.

Indexing all ~3.6M docs with full abstracts takes a while and produces a
multi-GB index — make sure the Meilisearch data dir has space. Use `--limit`
for a quick demo.

## 3. Open the explorer

Edit the top of [web/index.html](web/index.html):

```js
const HOST = "http://localhost:7700";
const SEARCH_KEY = "...";   // the search-only key printed in step 2
const INDEX = "cinii";
```

Then serve the folder (the UI calls Meilisearch from the browser):

```bash
python -m http.server 8080 --directory search/web
# open http://localhost:8080
```

## Notes / best practices applied

- **Lean index:** `full_text`, embeddings, and affiliations are excluded; only
  display/search/filter fields are stored.
- **Hierarchical LCC facet:** drill down main → subclass → division
  (`lcc.lvl0/1/2`) with human-readable names from `lcc_names.py`.
- **Trust as a filter:** `trust_tier` (reject/low/medium/trust) and the numeric
  `pred_centroid_sim` are filterable/sortable — surface only confident
  classifications, the payoff of the evaluation work.
- **Security:** the browser uses a **search-only** key, never the master key.
  For a public deployment, also restrict `displayedAttributes` and put
  Meilisearch behind a proxy.
- **Re-indexing** drops and recreates the index, so it is idempotent.
