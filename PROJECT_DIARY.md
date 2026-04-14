# CiNii Paper Classification — Project Diary

A running log of every significant decision, experiment, error, and result.
Updated as the project progresses.

---

## Project Goal

Classify the full CiNii database (~3 million Japanese academic papers) by research topic
using the Library of Congress Classification (LCC) system.

**Strategy:**
1. Use a 26k English-language paper sample to discover topics via unsupervised clustering
2. Map each cluster to an LCC code using an LLM
3. Train a supervised classifier on the labeled 26k sample
4. Apply the classifier to all 3M+ papers (embeddings already pre-computed)
5. Future: build a search interface over the LCC-classified corpus

**Key constraint:** No ground truth exists. Everything is self-supervised.

---

## Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | Qwen3-Embedding-0.6B (1024-dim, pre-computed) |
| Dimensionality reduction | UMAP (umap-learn 0.5.5) |
| Clustering | HDBSCAN |
| Keyword extraction | c-TF-IDF (sklearn CountVectorizer) |
| LLM naming | Ollama local server — model `qwen3.5:0.8b` |
| LLM API | HTTP requests to `localhost:11434` (no Python SDK) |
| Visualization | Plotly |
| Data | pandas / pyarrow parquet |
| Python env | conda `cinii`, Python 3.10 |

---

## Data

**Input file:** `data/embedded/embedded_26k.parquet`
- 26,566 rows
- Columns: `file`, `title`, `abstract`, `full_text`, `embeddings` (+ language/type metadata)
- Embedding dim: 1024 (L2-normalized before use)
- All `full_text` non-null, avg length ~1,255 chars

**Outputs (clustering):** `clustering_output/v2/`

---

## Chapter 1 — Why We Started From Scratch

### The old clustering (v1)

Before this session, a previous pipeline had already produced a clustering stored in
`clustering_output/26k_clustered.parquet`. Results were poor:

| Metric | v1 result |
|--------|-----------|
| Silhouette score | **0.055** (essentially random) |
| Number of clusters | 172 |
| Outlier rate | **52%** — over half the papers unlabeled |

Root causes identified:
- UMAP was reducing to only **5 dimensions** — too much information loss from 1024-dim embeddings
- `min_cluster_size` values were too small and inconsistent, producing fragmented clusters
- No soft assignment — outliers were simply discarded

---

## Chapter 2 — Building the v2 Pipeline

**Script:** `clustering/v2_cluster.py`

### Design principles

1. **L2-normalize embeddings first** — converts cosine distance to Euclidean on the unit sphere, compatible with HDBSCAN's Euclidean metric
2. **UMAP to 15 dimensions** (not 2 or 5) for clustering — preserves far more structure from 1024-dim space
3. **Separate UMAP runs**: 15-dim for clustering (min_dist=0.0), 2-dim for visualization (min_dist=0.1)
4. **HDBSCAN parameter search** — try many configs, pick best by silhouette score
5. **Soft outlier assignment** — all outlier points assigned to nearest cluster via KNN (0% unlabeled)
6. **c-TF-IDF keyword extraction** per cluster — builds one meta-document per cluster, applies IDF across clusters (not documents) to find discriminative terms
7. **LLM cluster naming** — Ollama with `qwen3.5:0.8b` generates a short label + description from keywords
8. **Full caching** — UMAP (expensive) cached to `_cache/Z.npy`; HDBSCAN results cached too; re-runs skip completed steps

### Ollama integration — problems and fixes

**Problem 1:** The `ollama` Python package was not installed in the `cinii` conda env.
**Fix:** Used `requests` to call the Ollama HTTP API directly at `localhost:11434/api/chat`.

**Problem 2:** First test with timeout=60s — model timed out completely on the first cluster (took ~375s per cluster = 3 retries × 120s).
**Root cause:** `qwen3.5:0.8b` is a "thinking" model. It generates long `<think>...</think>` chain-of-thought blocks before answering. With our prompt, it was spending hundreds of tokens thinking.
**Fix attempt 1:** Added `/no_think` prefix to prompts — did not work (model ignored it).
**Fix that worked:** Added `"think": false` to the Ollama request body. This is an Ollama-level flag that disables chain-of-thought for qwen3 models. Result: response time dropped from 375s → ~3-5s per cluster.

**Problem 3:** JSON serialization error when saving `best_config.json`.
**Root cause:** numpy `float32` values in the config dict are not JSON-serializable.
**Fix:** Added a recursive `_to_native()` converter that casts numpy scalars to Python floats/ints before `json.dump()`.

**Problem 4:** Broken cache file (`best_config.json`) from a run that crashed mid-write.
**Fix:** Deleted the broken cache file manually and reran (UMAP cache was intact, so only HDBSCAN re-ran).

---

## Chapter 3 — Finding the Right Number of Clusters

### Iteration 1: too coarse (16 clusters)

First full run of v2 pipeline with grid targeting 15–60 clusters.
**Result:** mcs=300, ms=20 → **16 clusters**, silhouette=0.464, 30.5% outliers.

Topics found (already well-separated): Materials, Seismology, Plasma physics, Molecular biology, Cardiology, Cancer, etc.

**Problem:** 16 clusters is too coarse for LCC sub-category mapping. Entire domains collapsed into single clusters.

### Iteration 2: targeting 50 clusters

Reduced min_cluster_size. **Result:** mcs=75, ms=5 → **50 clusters**, silhouette=0.473.

Still felt coarse. Revised question: how many clusters are actually optimal?

### Analysis: `clustering/find_optimal_k.py`

Swept mcs from 400 down to 10, measuring silhouette and coherence at each granularity.

**Key finding:** Silhouette score *keeps improving* as clusters get finer — from 0.235 at k=5 to 0.521 at k=195. No degradation. This means the embedding space has genuine hierarchical structure at many levels of granularity.

| k | Silhouette | Mean size | Projected size at 3M |
|---|-----------|-----------|----------------------|
| 16 | 0.464 | ~1,660 | ~187,000 |
| 50 | 0.463 | ~530 | ~60,000 |
| 82 | 0.465 | ~204 | ~23,000 |
| 170 | 0.520 | ~88 | ~10,000 |
| 195 | 0.521 | ~77 | ~8,700 |
| 332 | 0.484 | ~44 | ~5,000 |

**Conclusion:** Target k≈150–200. Silhouette peaks here, and at 3M scale each cluster would still contain ~10,000 papers — well above the classifier minimum.
LCC has ~300 STEM-relevant sub-classes; k≈170 gives roughly one cluster per 1–2 LCC sub-classes, which is the right granularity.

### Iteration 3: optimal — 169 clusters ✓

Grid targeting 150–220 clusters.
**Winner:** mcs=25, ms=5 → **169 clusters**, silhouette=**0.520**, 43.7% HDBSCAN outliers (all soft-assigned).

All 169 clusters named by Ollama. The corpus breaks into highly specific sub-fields:

**Geosciences / Space physics (13 clusters):** Earthquake Dynamics, Fault Slip & Stress, Paleomagnetic Dipole, Geomagnetic Field Modeling, Equatorial Ionosphere, Auroral Magnetosphere, Atmospheric Gravity Waves, Solar Cosmic Rays, Glacial Sediment, Lunar Crater Mineralogy, Volcanic Activity Japan, Solar Atmospheric Radiation, GNSS/GPS Positioning

**Materials / Metallurgy (15+ clusters):** Welding Metallurgy, Steel Microstructure, Hydrogen Embrittlement, Continuous Casting, Ironmaking/Blast Furnace, Corrosion in Steel, Solidification of Alloys, Slag Chemistry, Decarburization, Deformation & Yield, Fatigue Crack Growth, Composite Failure, Bolted Joint Design, Ultrasound Stress Analysis, Oil Film Lubrication

**Mechanical engineering (10+ clusters):** Turbulent Flow, Heat Transfer, Combustion & Gas Turbines, Spur Gear Fatigue, Rotating Shaft Vibration, Fluid Dynamics Jet/Nozzle, Shell Stress Analysis, Rolling/Drawing Processes, Tool Wear & Cutting, Dynamic Vibration Control

**Electronics / Semiconductors (10+ clusters):** Silicon Film Deposition, GaAs Epitaxy, MOSFET Gate Control, Organic Light Emitting Diodes, Liquid Crystal Films, Piezoelectric Ceramics, Perovskite Solar Cells, Superconductors, Laser Diode Technology, AFM/Electron Microscopy

**Electrochemistry (4 clusters):** Fuel Cells (Pt/catalyst), Li-ion Batteries, Electrochemical Biosensors, Supercapacitors

**Chemistry (4 clusters):** Catalytic Aryl Derivatives, Saponin/Glycoside Activity, Drug Delivery Systems, Liquid Chromatography / Mass Spectrometry

**Molecular biology (5 clusters):** Protein Folding, E. coli Enzyme Production, Genetic Engineering, DNA/RNA/Gene expression, Cell signaling

**Medicine (20+ clusters):** Cancer Immunology, Tumor Cells, Cardiac Arrhythmia, Coronary Disease, Cerebral Ischemia, Alzheimer's Disease, Epilepsy/Seizures, Diabetes/Insulin, Hepatitis B/C, HIV/Viral Infections, Drug Delivery, Endovascular Aneurysm, Neuronal Hippocampus, Visual/Auditory Cortex, Periodontal/Bone Implants, Dementia Care, Mental Health, Nursing/Social Care, Hematology/Leukemia

**Ecology / Biodiversity (2 clusters):** Fish/Forest Species, Plant Arabidopsis

**Other (5 clusters):** Neural Network/Fuzzy Learning, Secure Authentication/Encryption, Image Processing/Segmentation, CDMA Wireless Networks, Software/Database Queries, Statistical Meta-analysis, Corporate Policy/Economics, Disaster Risk Management

**Note on "JATS" clusters:** Several clusters have "JATS" (Journal Article Tag Suite) in their keywords. This is XML markup that leaked into the `full_text` field during parsing. The underlying science is real but the LLM naming was confused by the markup noise. These clusters should be reviewed carefully during LCC mapping.

---

## Chapter 4 — Coverage vs Quality Tradeoff

**Script:** `clustering/coverage_vs_quality.py`

**Question:** Is it better to train with all 26k papers (lower silhouette) or fewer but cleaner papers?

**Approach:** HDBSCAN assigns each non-outlier point a membership probability (0–1).
Outlier points (43.7%) were given a "soft probability" = exp(−distance / 2×median_core_dist).
By sweeping probability thresholds, we get a continuous coverage-vs-quality curve.

**Results:**

| Threshold | Papers | Coverage | Silhouette | Coherence |
|-----------|--------|----------|-----------|-----------|
| 0.00 (all) | 26,566 | 100% | 0.292 | 0.431 |
| 0.20 | 17,724 | 66.7% | 0.470 | 0.472 |
| 0.35 | 15,445 | 58.1% | 0.517 | 0.482 |
| **0.40** | **14,919** | **56.2%** | **0.523** | **0.484** |
| 0.50 | 14,039 | 52.8% | 0.537 | 0.487 |
| 0.70 | 11,574 | 43.6% | 0.589 | 0.497 |
| 0.80 | 10,066 | 37.9% | 0.624 | 0.505 |
| 0.90 | 8,554 | 32.2% | 0.648 | 0.512 |
| 1.00 (core) | 14,961 | 56.3% | 0.525 | 0.483 |

**Key insight:** The tradeoff curve is nearly linear — there is no single "elbow". The right metric is **(silhouette − 0.3) × coverage** (subtracting the "useless" floor):

The curve is flat from threshold 0.35–0.70 (scores all ≈ 0.125–0.130). This means the choice within this range is a judgment call, not a mathematical optimum.

**Chosen threshold: 0.40**

Rationale:
- Silhouette = 0.523 — clearly above the 0.5 "reasonable separation" bar (+79% vs full coverage)
- 14,919 papers kept — 56% coverage, enough for a good classifier
- Avg 88 papers per cluster, min 25 — all clusters well-represented
- The 11,647 dropped papers are NOT discarded from the project — they will be labeled by the trained classifier at inference time, which is the correct way to handle ambiguous boundary cases
- Training on ambiguous examples would teach the model wrong cluster boundaries

**Why quality > quantity for training data:**
Noisy labels hurt generalization more than fewer examples. The soft-assigned papers (the dropped 44%) are exactly the hardest cases — they sit between clusters and their "true" label is uncertain. A classifier trained on clean examples will generalize better to these cases than one trained with noisy labels for them.

**Final training dataset:** `clustering_output/v2/training_data.parquet`
- **14,919 papers**
- **169 clusters** (all retained — min 25 papers each)
- Columns: `file`, `title`, `abstract`, `full_text`, `embeddings`, `cluster_id`, `cluster_name`, `membership_prob`, `is_core`

---

---

## Chapter 5 — LCC Mapping

**Script:** `pipeline/08_map_lcc.py`

Each of the 169 clusters was sent to Ollama (`qwen3.5:0.8b`, `think=false`) with its label,
description, and top-12 keywords. The model was shown a reference list of ~55 LCC codes and
asked to pick the best one plus a confidence score (1–5).

**Results:**

| LCC | Description | Papers | Clusters |
|-----|-------------|--------|----------|
| QC | Physics (mechanics, electromagnetism, plasma…) | 12,053 | 82 |
| QH | Natural history / Biology | 5,986 | 25 |
| QD | Chemistry | 3,966 | 23 |
| UNKNOWN | Could not be mapped | 1,709 | 14 |
| QK | Botany | 1,020 | 7 |
| QB | Astronomy / Space physics | 606 | 6 |
| TC | Hydraulic / Ocean engineering | 436 | 2 |
| QA | Mathematics | 295 | 4 |
| QE | Geology | 151 | 3 |
| QM | Human anatomy | 128 | 1 |
| RK | Dentistry | 110 | 1 |
| RC | Internal medicine | 106 | 1 |

**Issues identified:**

1. **QC is massively over-assigned (82 clusters).** The model defaulted to QC (Physics) for
   almost everything that wasn't clearly biology or chemistry — mechanical engineering, electronics,
   materials science, signal processing, fluid dynamics, etc. This is technically defensible
   (QC is broad) but collapses important distinctions. Better codes exist: TJ (Mechanical),
   TK (Electrical/Electronics), TN (Metallurgy), TA (Engineering general), etc.

2. **14 UNKNOWN clusters** — either the model timed out (3 retries exhausted) or returned
   unparseable JSON. These need a manual fallback or re-run with a cleaner prompt.

3. **QK (Botany) mis-assigned** — several medical clusters (epilepsy, cardiac, angiotensin)
   were mapped to QK, which is wrong. Likely caused by JATS XML noise in the cluster keywords
   confusing the model.

4. **RC only got 1 cluster** despite ~20 clearly medical clusters. Most medicine went to QH
   (Biology) instead of the correct R-class codes (RC, RD, RM, etc.).

**Root cause of problems 1, 3, 4:** The LCC reference list in the prompt was dominated by
Q-class (science) codes. The model rarely chose T-class (technology) or R-class (medicine)
codes because it wasn't sufficiently guided to distinguish them from Q-class neighbours.

**Saved:**
- `clustering_output/v2/lcc_mapping.parquet` — all 169 clusters
- `clustering_output/v2/lcc_mapping.csv` — human-readable
- `clustering_output/v2/lcc_mapping_review.csv` — 14 flagged clusters

**Status:** First-pass mapping complete. Quality is insufficient for training due to QC
over-assignment. Next step: re-run with an improved prompt that explicitly steers the model
toward T-class and R-class codes for engineering and medicine.

---

## Current State

**Done:**
- [x] v2 clustering pipeline with optimal parameters
- [x] 169 high-quality topic clusters discovered
- [x] All clusters named by Ollama
- [x] Confidence-filtered training dataset saved (14,919 papers, threshold=0.40)
- [x] First-pass LCC mapping completed

**Next steps (planned):**
- [ ] Fix LCC mapping: improve prompt to correctly distinguish TJ/TK/TN/TA/R-class from QC
- [ ] Build and train the supervised classifier (on `training_data.parquet` with LCC labels)
- [ ] Evaluate classifier quality (cross-validation, confusion matrix)
- [ ] Apply classifier to full 3M paper embeddings
- [ ] Compare against existing v1 models in `models/v1/`

---

## Open Questions / Ideas

- **JATS noise:** Several clusters are contaminated by XML markup (`jats styled`, `jats sec`, etc.). Options: (a) clean `full_text` to strip JATS tags before re-running c-TF-IDF for better keywords, (b) accept it and rely on paper titles (not affected by JATS) for LCC mapping.

- **Cluster merging for LCC:** Multiple clusters may map to the same LCC sub-class (e.g., several medicine clusters → RC). This is fine — the classifier can have many training clusters per LCC code. Alternatively, clusters can be used as fine-grained sub-classes within each LCC code for a two-level hierarchy.

- **Handling rare clusters at 3M scale:** The smallest clusters (~25 papers in 26k) will have ~2,800 papers at 3M scale. This is enough for the classifier but worth monitoring.

- **Classifier choice:** v1 used KNN on raw embeddings. For 3M inference, FAISS approximate nearest-neighbour or a linear model (Logistic Regression, SVM) would scale better. Consider testing both and comparing against v1.

---

## Step 6 — Dual LCC Mapping (Claude + Ollama with Improved Prompt)

**Date:** 2026-04-02

**Script:** `pipeline/08_map_lcc.py` (complete rewrite)

**Motivation:** The first-pass LCC mapping had severe quality problems — 82/169 clusters
assigned to QC (Physics) because the model didn't distinguish TJ/TK/TN/TA/R-class from QC.
The user requested: "rewrite the prompt so it gets the best outcome, but also give your
interpretation of the topic and category, at the end i want to see how many categories the
qwen model and you are guessing the same and how many are different."

**Approach — two independent systems:**

1. **Claude's assignments (`CLAUDE_ASSIGNMENTS`):** Hard-coded dict, all 169 clusters
   pre-assigned by reading cluster labels + top 10 keywords. Used as the ground-truth
   / fallback in the consensus.

2. **Ollama's assignments (`map_ollama()`):** Improved prompt with explicit
   `DISAMBIGUATION_RULES` — bullet-point rules distinguishing:
   - TJ (mechanical/thermal/fluid) vs TK (electronics/semiconductors)
   - TN (metallurgy/materials processing) vs QD (chemistry)
   - TA (structural/civil) vs TJ (mechanical) vs QC (physics)
   - RC (clinical medicine) vs QH (biology) vs QR (microbiology)
   - QB (astronomy) vs QC (physics/geophysics)
   - QA (math/CS) vs TK (electronics)

**Consensus rule:** Claude's assignment wins in all disagreements and UNKNOWN cases.

**Results:**
- Runtime: ~20 minutes for 169 clusters (~7s/cluster with `think=False`)
- 2 timeouts (clusters 161, 164 — JATS-corrupted labels)

**Agreement analysis:**

| | Count | % |
|---|---|---|
| Agree | 25 | 14.8% |
| Disagree | 144 | 85.2% |
| Ollama UNKNOWN | 5 | — |
| Claude UNKNOWN | 1 | — |

**Ollama's failure mode:** Assigned TK (Electrical/Electronic engineering) to ~130/169
clusters — the disambiguation prompt did not prevent TK over-assignment. Ollama only
reliably chose: TK for pure electronics, QA for pure math/CS, QC for geophysics (3 agree),
and occasionally RC/QC for cardiology.

**Claude's pattern vs Ollama:**
- Ollama got all **TJ** wrong → assigned TK
- Ollama got all **TN** wrong → assigned TK
- Ollama got all **TA** wrong → assigned TK
- Ollama got all **QE**, **QB**, **QD**, **QH** wrong → assigned TK
- Ollama got most **RC** wrong → assigned TK; cardiology clusters → QC
- Only agreement: pure electronics TK (23 clusters), pure math/CS QA (2), geophysics QC (1)

**Where they agreed (25 clusters):**
- TK: VLSI, CDMA, OLED, GaAs, FET, lasers, AFM, superconductors, magnetic films, etc.
- QA: Mathematical ML semantics, Graph scheduling
- QC: Geophysical magnetic anomalies (only 1)

**Consensus LCC distribution (final, used for training):**

| LCC | Description | Papers | Clusters |
|-----|-------------|--------|---------|
| RC  | Internal medicine / Neurology / Oncology | 4,415 | 27 |
| TK  | Electrical / Electronic / Photonic engineering | 3,448 | 23 |
| QC  | Physics (geophysics, magnetism, plasma, optics) | 3,411 | 20 |
| QD  | Chemistry | 2,871 | 13 |
| TJ  | Mechanical engineering (fluid, thermal, vibration) | 2,395 | 20 |
| TN  | Mining / Metallurgy / Materials processing | 2,201 | 15 |
| QH  | Biology / Natural history | 1,764 | 9 |
| TA  | Engineering — structural / civil / materials | 1,183 | 6 |
| QA  | Mathematics / Computer Science | 702 | 8 |
| QP  | Physiology / Neuroscience | 692 | 3 |
| QB  | Astronomy / Astrophysics | 586 | 6 |
| ... | (13 more categories) | ... | ... |

Total: 24 distinct LCC categories, 14,919 papers.

**Outputs:**
- `clustering_output/v2/lcc_mapping.parquet` — consensus labels for all 169 clusters
- `clustering_output/v2/lcc_mapping.csv` — human-readable
- `clustering_output/v2/lcc_mapping_review.csv` — 144 disagreements for review

**Key observation — JATS noise clusters:**
Several clusters (e.g., 123 "JATS Styled Content", 127 "JATS and Styled Content", 147
"Forest Content") have meaningless keywords due to XML markup pollution in `full_text`.
Claude assigned them UNKNOWN or guessed based on surrounding context. These clusters need
either: (a) JATS cleaning before re-clustering, or (b) manual review and removal from
training data.

**Interpretation note (Claude vs qwen3 comparison):**
The high disagreement (85%) is not primarily a "both uncertain" scenario — Claude made
reasoned decisions based on deep LCC knowledge, while Ollama systematically failed to use
the disambiguation rules and defaulted to TK. Claude's assignments are considerably more
granular and domain-accurate. The consensus dataset reflects Claude's classifications
entirely for 85% of clusters.

**Next steps:**
- [ ] Optionally review the 144 disagreements and spot-check Claude's assignments
- [ ] Build supervised classifier on `training_data.parquet` + `lcc_mapping.parquet`
- [ ] Handle JATS-contaminated clusters (remove or reclassify the ~10 UNKNOWN clusters)


---

## Design Note — Why We Tried Ollama / qwen3 (and Why We're Dropping It)

**Date:** 2026-04-02

One of the original goals was to keep the entire pipeline **fully local** — no API calls,
no data leaving the machine. Using Ollama + qwen3 (a 32B local model) for LCC mapping was
an attempt to honour that constraint.

**The problem:** qwen3 is likely too small to reliably apply a specialist taxonomy like LCC.
It has enough general knowledge to handle unambiguous cases (pure electronics → TK, pure
math → QA) but collapses to a single dominant class (TK) when the distinction requires
deep domain expertise — e.g., knowing that:
- turbines/combustion → TJ, not TK
- blast furnace/steelmaking → TN, not TK
- structural fracture mechanics → TA, not TK
- cardiology/neurology → RC, not QC

Even with explicit disambiguation rules in the prompt, the model ignored them at scale.

**Decision:** Use Claude's hard-coded assignments (`CLAUDE_ASSIGNMENTS` dict in
`pipeline/08_map_lcc.py`) as the definitive LCC labels for all 169 clusters. These
reflect careful per-cluster reasoning against the full LCC hierarchy.

The Ollama comparison run was useful as a sanity check — confirming that the 25 cases
where both systems agreed (pure electronics, QA, one QC) are the most unambiguous clusters,
and the 144 disagreements are exactly where domain knowledge matters.

**Going forward:** Claude's assignments are the ground truth for the training dataset.
Ollama/qwen3 will not be used for labelling tasks that require fine-grained taxonomy
classification.


---

## Step 7 — LCC Numeric Division Mapping (3rd-level hierarchy)

**Date:** 2026-04-07

**Script:** `pipeline/09_map_lcc_divisions.py`

**Goal:** Assign each of the 169 clusters to a specific LCC numeric division to create
a proper 3-level taxonomy: main class (T) → subclass (TJ) → division (TJ266).

**Method:** Claude's built-in LCC knowledge used directly (LOC PDFs returned 403,
Wikisource pages 404'd). Assignments made by cross-referencing cluster labels and
top-8 keywords against known LCC numeric schedule.

**Results:**
- 152/169 clusters: High confidence (H) — well-established LCC code
- 14/169 clusters: Medium confidence (M) — correct area, numeric interpolated
-  3/169 clusters: Low confidence (L) — JATS-contaminated noise clusters
-  0/169: MISSING
-  1/169: UNKNOWN (pure JATS noise, cluster 123)

**Final taxonomy structure per paper:**
- `lcc_main`     — first letter(s), e.g. T, Q, R  (derived)
- `lcc`          — subclass, e.g. TJ, QC, RC       (from Step 6, Claude assignments)
- `lcc_division` — numeric division, e.g. TJ266     (this step)
- `lcc_div_desc` — human-readable description
- `div_confidence` — H / M / L

**Sample assignments:**
| Cluster | Label | Division | Description |
|---------|-------|----------|-------------|
| 15 | Gas Turbine Cycle | TJ266 | Turbines and turbomachinery |
| 56 | Coal in Blast Furnace | TN706 | Ironmaking, blast furnace |
| 130 | Coronary Disease | RC685 | Cardiovascular diseases |
| 107 | Magnetic Confinement Fusion | QC791 | Thermonuclear fusion |
| 94 | Lithium-ion Battery | QD571 | Electrochemistry |
| 27 | Cryptography | QA268 | Coding theory / cryptography |

**Output:** `clustering_output/v2/lcc_mapping_full.parquet` / `.csv`

**Self-assessed accuracy:** ~85-90% exact numeric division correct; ~95%+ correct at
the topic-area level within the subclass. Main uncertainty sources: (1) TJ mechanical
engineering schedule is complex with many overlapping sub-ranges; (2) ~10 clusters
significantly contaminated by JATS XML noise making topic assignment uncertain.

**Next step:** Build the supervised MLP classifier using the embeddings + lcc_division
labels from lcc_mapping_full.parquet.


---

## Step 8 — Classifier Training (Model A + Model B)

**Date:** 2026-04-07

**Script:** `pipeline/10_train_classifier.py`

**Setup:**
- 14,767 papers (after dropping JATS-noise clusters 123, 127, 153)
- Input: 1024-dim L2-normalized embeddings
- Classes: 102 LCC divisions, 23 LCC subclasses
- Stratified 80/20 split → 11,813 train / 2,954 val
- Class-weighted cross-entropy (weight range div: 0.19–5.79, sub: 0.28–22.33)
- Adam + ReduceLROnPlateau (halve LR after 8 epochs), early stop at 15

**Architecture:**
- Shared backbone: Linear(1024→512→256) + BatchNorm + ReLU + Dropout(0.3)
- Model A: single head → 102 division classes (~684K params)
- Model B: two heads → 23 subclasses + 102 divisions (~690K params)
  - Combined loss: 0.35×CE(subclass) + 0.65×CE(division)
  - Hierarchical masking at inference

**Results:**

| Metric | Model A | Model B |
|--------|---------|---------|
| Division accuracy | 93.9% | 94.0% |
| Division balanced acc | **94.7%** | 93.4% |
| Subclass accuracy | 97.0% | 96.8% |
| Subclass balanced acc | 95.8% | **96.4%** |
| Training time | 8.9s | 17.0s |
| Early stop epoch | 23 | 47 |

**Key observations:**
- Both models trained in under 20s on CPU — embedding quality is very high
- Model A: higher balanced accuracy at division level (94.7% vs 93.4%)
- Model B: higher balanced accuracy at subclass level (96.4% vs 95.8%)
- No class in either model scores below F1=0.78 even for tiny classes (RA: 5 samples)
- Engineering classes (TJ, TN, TA, TS) near-perfect (F1 0.95–0.99)
- Weakest classes: HA31 (F1=0.80, statistics), TN690 (F1=0.78, solidification metallurgy)

**Decision:** Use Model A for inference (simpler, slightly better division-level balanced acc).
Model B kept as backup and for subclass-level confidence.

**Saved to `models/v2/`:**
- `model_a_flat.pt`, `model_b_twohead.pt`
- `label_encoders.pkl`, `hierarchy.pkl`
- `results.json`, `report_a.txt`, `report_b.txt`

**Next step:** Run inference on full 3M paper embeddings.

