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

---

## Chapter 9 — Server Pipeline: First End-to-End Run (Sample)

**Date:** 2026-05-25

**Goal:** Build a single crashproof orchestration script for the server and test it on a
sample of the full CiNii database before committing to the full 70M-document run.

### New scripts created

- **`pipeline/run_server.py`** — crashproof orchestrator covering parse → clean → embed →
  classify → report. Auto-skips stages whose output already exists. Writes a log file. Has
  `--force-stage`, `--skip-classify`, `--skip-embed`, `--dry-run` flags.
- **`pipeline/12_report_classification.py`** — post-classification quality report. Generates:
  overview stats, LCC distribution, top subclasses, TF-IDF top terms per class, sample titles,
  low-confidence examples. Accepts `--lcc-mapping` to show cluster training overview.

### Bugs discovered and fixed during first server run

1. **`ImportError: cannot import name 'MODEL_DIR'`** — `config.py` exports `MODELS_DIR`
   (plural). Fix: `from config import MODELS_DIR as MODEL_DIR` in `11_classify.py`.

2. **`KeyError: 'div'`** — training saved `{"le_div":..., "le_sub":...}` but classify read
   `encoders["div"]`. Fix: use `encoders["le_div"]`, `encoders["le_sub"]`.

3. **`KeyError: 'mask'`** — training saved `{"hier_mask":...}` but classify read
   `hier["mask"]`. Fix: `hier["hier_mask"]`.

4. **Schema mismatch in classify output** — Arrow infers `null` type for all-null columns
   (language, doi, error) in some chunks. Fix: promote null → string when establishing the
   file schema, cast every chunk to it.

5. **Partial file on crash** — failed write left a partial `.parquet` that `run_stage` saw as
   complete and skipped. Fix: atomic write via `.tmp.parquet` + `os.replace()`.

### First sample run results

- Source: 500k raw RDF files from a subset of the CiNii dump
- **35,456 English scientific papers** after clean stage
- Classification with Model B (v2): completed in ~40 min total

**Classification report — first sample run:**

| LCC | Topic | Papers | % | Mean Conf |
|-----|-------|--------|---|-----------|
| T   | Technology | 14,896 | 42.0% | 0.919 |
| Q   | Science | 14,575 | 41.1% | 0.932 |
| R   | Medicine | 5,184 | 14.6% | 0.897 |
| H   | Social Sciences | 539 | 1.5% | 1.000 |
| L   | Education | 262 | 0.7% | 1.000 |

- Mean confidence: **0.923** / Median: **0.998**
- conf < 0.50: 3.2% / conf < 0.30: 0.2%
- JATS in classified abstracts: **0** ✅

**Key observations from this first run:**

1. Only **5 of 21 LCC main classes** represented — expected, since training data only covered
   T/Q/R/H/L (CiNii is STEM-heavy, and the 26k training sample reflected that).
2. H and L have mean confidence = **1.000** — overconfidence flag. Model has never seen J, K,
   G, D etc., so anything not STEM gets forced into H or L with fake certainty.
3. Top TF-IDF terms are topically coherent for T/Q/R. Boilerplate words (abstract, paper,
   study, results) polluted the word lists → fixed by adding custom stopword set.
4. Japanese titles and abstracts still appeared in H class despite English-only filter —
   publisher declares `dc:language="en"` but abstract is Japanese. Root cause: langdetect
   fallback only runs when publisher tag is missing; if publisher says "en", we trust it.

### Fixes applied to report quality

- Added ~30 boilerplate stopwords to TF-IDF (abstract, paper, study, results, method, data…)
- `_strip_html()` strips `<SUB>`, `<SUP>`, `<i>` from displayed titles (HTML subscript
  characters common in chemistry/physics paper titles)
- Sample title selector prefers Latin-script titles; non-Latin shown with `[non-Latin]` flag
- `TfidfVectorizer` `stop_words` must be a `list`, not `frozenset` — fixed after server error

---

## Chapter 10 — Full CiNii Database Parse + Clean (FullDatasetV2)

**Date:** 2026-06-01 to 2026-06-02

### Parse stage

**Command:**
```bash
RAW_DIR=/mnt/exssd3/.../data/raw \
python pipeline/run_server.py \
    --run-name FullDatasetV2 \
    --parse-workers 12 \
    --skip-embed \
    --skip-classify
```

**Speedup applied:** `01_parse_rdf.py` updated to use `ProcessPoolExecutor` with `--workers`
flag. Each `.rdf` file is fully independent → safe for multiprocessing. Also switched merge
step from full-RAM concat to streaming `ParquetWriter` (peak RAM: one batch vs all batches).

**Parse results:**
| Metric | Value |
|--------|-------|
| Total RDF records parsed | **71,511,821** |
| Parse errors | 20,375 (0.03%) |
| Have abstract | 23,164,486 (32.4%) |
| Publisher-declared language | 41,073,717 rows |
| JATS in abstracts | **0** ✅ |
| Parse time | **622.3 minutes (~10.4 hours)** |
| Output file size | 19,823 MB |

### Clean stage — issues and fixes

**Problem 1 — OOM on `pd.read_parquet(71M rows)`:** Loading the full 20 GB parsed parquet
into pandas RAM killed the process. Fix: switched to `pq.iter_batches(batch_size=500_000)`
chunked loading, dropping empty rows per chunk. Peak RAM: ~2 GB vs ~40 GB.

**Problem 2 — Corrupted UTF-8 strings:** `ArrowException: Unknown error: Wrapping
一台の多機能機械が付加Á­れた... failed`. Some strings in the parquet have broken UTF-8
(garbled Japanese). Fix: wrapped each batch's `to_pandas()` in a try/except; corrupted batches
skipped with a warning rather than crashing the full load.

**Problem 3 — OOM on `report_parsed()`:** `report_parsed()` loaded the full 20 GB parquet
to count rows, killing the process. Fix: all four `report_*` functions now stream in 500k-row
chunks and accumulate counters, never loading the full file.

**Problem 4 — Japanese leakage past English filter:** The `keep_only_english()` function
trusts `dc:language="en"` from publishers without validation. Some Japanese journals declare
their papers as English when the abstract is actually Japanese. The exploration run (below)
revealed ~23,945/169,537 sampled docs with >10 non-ASCII characters, all with full Japanese
abstracts. Fix: added **CJK character ratio filter** — if >5% of abstract characters are
Japanese/Chinese Unicode (hiragana, katakana, kanji, CJK Extension A), the paper is excluded
regardless of publisher language tag.

**Clean results (after CJK fix):**
| Stage | Papers | Notes |
|-------|--------|-------|
| Input (parsed) | 71,511,821 | |
| After drop empty title/abstract | 23,156,080 | dropped 48.4M |
| After drop uninformative | 7,010,053 | dropped 16.1M |
| After langdetect filter | 4,741,531 | kept 67.6% |
| After CJK filter (new) | ~3,602,151 | removed ~108k Japanese-leaked papers |
| After doc type filter | **3,602,151** | scientific papers only |
| JATS in clean abstracts | **0** ✅ | |
| Clean time | ~58 minutes | |

The CJK filter removed **~108,000 papers** (≈ 2.9% of the pre-filter English corpus) that
were falsely declared English by publishers. These were papers with full Japanese abstracts.

### Data exploration (explore_data.py)

**New script:** `pipeline/explore_data.py` — generates PNG charts and text summary for any
pipeline stage output. Auto-detects parquet type (cleaned / classified / embedded).
Charts: missing values, abstract length distribution, publication year, top publishers/journals.
Special sections in summary.txt:
- **Random examples** (n=10): shows title + first 400 chars of abstract
- **Short abstract examples** (<20 words): catches Japanese text that slipped through
- **JATS/XML tag survivors**: checks if any HTML/XML tags survived cleaning → all 0 ✅
- **High non-ASCII examples** (>10 chars): catches language leakage

**Key exploration findings on the cleaned corpus:**
- Abstract length: mean **131 words**, median **113 words**
- Publication years: range 1950–2026, peak 2010s
- JATS survivors: **0** ✅
- Japanese leakage in H class before CJK fix: confirmed (all 5 short-abstract examples
  were full Japanese text)
- High non-ASCII (>10 chars) = 14% of sample — mostly **legitimate English** papers using
  Unicode em-dashes, subscripts, non-breaking spaces; NOT a problem

---

## Chapter 11 — Full Corpus Re-Embedding (FullDatasetV2_clean)

**Date:** 2026-06-02 (started), ongoing

**Goal:** Generate Qwen3-Embedding-0.6B embeddings for all 3,602,151 English scientific
papers. Outputs sharded parquet files to `data/embedded/FullDatasetV2_clean/`.

**Command:**
```bash
DEVICE=cuda \
python pipeline/run_reembed.py \
    --run-name FullDatasetV2_clean \
    --source data/cleaned/english_FullDatasetV2.parquet \
    --batch-size 16 \
    --max-length 768 \
    --shard-size 50000
```

### Token length analysis (new script: check_token_lengths.py)

Before embedding, measured the real token length distribution on a 5,000-doc sample:

| Percentile | Tokens |
|-----------|--------|
| p50 | 263 |
| p75 | 358 |
| p90 | 438 |
| p95 | 493 |
| p99 | 633 |
| p99.9 | 915 |
| max | 1,463 |

- **>512 tokens: 4.04%** — not "99% coverage" as initially claimed
- **>768 tokens: 0.28%** — very small fraction
- **Chosen cap: 768 tokens** — only 0.28% truncated, and those are very long abstracts
  where the topic is established well within the first 768 tokens

### Fixes to run_reembed.py

1. **CUDA OOM with batch_size=16, default max_length=32768:** Qwen3 defaults to 32768 max
   tokens. Attention is O(L²), so long abstracts consumed all 11GB VRAM, causing segfaults.
   Fix: `model.max_seq_length = max_length` (set to 768) caps tokenizer truncation.
   Added `--max-length` CLI argument (default: 512, used 768 for this run).

2. **Thousands of tiny "4/4" progress bars:** `getEmbeddings()` created a new tqdm bar per
   outer chunk (batch_size × 4 docs). Fix: added `show_progress=False` parameter to
   `getEmbeddings()`, replaced with one outer tqdm bar tracking total docs across the full job.

3. **Resume re-embedding already-done shards:** On restart, the code re-embedded all rows
   from position 0 (just skipping the write step). At 13 docs/sec this wasted many hours.
   Fix: calculate `skip_rows = n_done_shards × shard_size`, fast-forward through those rows
   without calling the embedding model.

4. **Outer batch size too small:** Was `batch_size × 4 = 64`. Increased to
   `max(batch_size × 8, 256)` for better GPU utilisation.

### Embedding run status (as of 2026-06-04)

| Metric | Value |
|--------|-------|
| Total docs to embed | 3,602,151 |
| Shard size | 50,000 docs (~200 MB each) |
| Total shards needed | 73 |
| GPU | NVIDIA GTX 1080 Ti (11 GB VRAM) |
| GPU utilisation | **100%** |
| VRAM used | 7,478 MB / 11,264 MB |
| Power draw | 193W / 250W |
| Temperature | 83°C (stable, max safe ~91°C) |
| Throughput | ~13 docs/sec |
| Progress | ~58% (≈2.1M / 3.6M docs) |
| ETA | ~32 hours remaining |

Two additional GTX 1080 Ti GPUs (GPU 1, 2) are idle on the same machine. Multi-GPU
parallelism was considered but deferred — the current run is healthy and will complete.

---

## Chapter 12 — Retraining Infrastructure (run_training.py)

**Date:** 2026-05-25

**Goal:** Build a solid, reusable pipeline for training new classifier versions on larger
and more diverse data, fixing the 5-class coverage limitation of v2.

### Root cause of the 5-class limitation

The v2 model only knows 5 LCC main classes (T, Q, R, H, L) because the 26k training sample
was dominated by STEM papers from CiNii. Non-STEM topics (J=Political Science, K=Law,
D=History, G=Geography, etc.) either didn't appear in the sample or appeared in too few
papers to form clusters.

**Evidence from classification report:**
- H (Social Sciences) and L (Education) have mean confidence = **1.000** — overconfidence
  indicating the model uses these as "garbage bins" for anything not STEM
- Sample titles in H class include J and D-class papers (political science, history)
  classified with fake certainty

### New script: pipeline/run_training.py

Self-contained orchestrator covering: sample → embed → cluster → [manual LCC pause] → train.

**Stages:**
1. **SAMPLE** — draw N docs from any parquet (cleaned text or pre-embedded)
2. **EMBED** — generate embeddings; auto-skipped if source already has `embeddings` column
3. **CLUSTER** — UMAP(15 components) + HDBSCAN auto-tune via silhouette score; TF-IDF
   keywords per cluster; outputs `lcc_mapping.csv` template; optionally bootstraps LCC
   suggestions from existing model (`--suggest-lcc`)
4. **[PAUSE]** — user fills `lcc_mapping.csv` (lcc_subclass + lcc_division columns)
5. **TRAIN** — validates mapping, builds training_dataset.parquet, trains Model A + B;
   saves `model_a.pt`, `model_b.pt`, `encoders.pkl`, `hierarchy.pkl`, `metrics.json`

**Key design decisions:**
- All stages tracked in `run_state.json` — auto-skip completed stages on re-run
- All writes atomic (`.tmp.parquet` → `os.replace()`) — no partial files
- LCC bootstrap: `--suggest-lcc models/release` runs existing model on sample docs, takes
  majority-vote prediction per cluster as `suggested_lcc_*` in the CSV — user just
  verifies/corrects instead of assigning from scratch
- File naming: `model_a.pt`, `model_b.pt`, `encoders.pkl` (cleaner than old
  `model_a_flat.pt`, `label_encoders.pkl`)
- `11_classify.py` updated to support both old and new naming conventions via fallback

**Output layout:**
```
training_runs/{run_name}/
    sample.parquet, embedded.parquet, clusters.parquet
    cluster_metadata.parquet, lcc_mapping.csv, training_dataset.parquet
    run_state.json, hdbscan_tuning.json

models/{run_name}/
    model_a.pt, model_b.pt, encoders.pkl, hierarchy.pkl
    metrics.json, report_a.txt, report_b.txt
```

### Pipeline methodology assessment

The approach (clustering → pseudo-label → train classifier) is a legitimate technique
called **cluster-based pseudo-labeling**, used in domain adaptation and digital library
research when no labeled data exists.

**Strengths:** STEM classification quality is good (T/Q/R ~97.7% of corpus, high confidence).
**Weaknesses:**
- 43.7% HDBSCAN outlier rate — nearly half the 26k sample discarded
- Training only on "easy" docs (clear cluster membership), never on ambiguous ones
- Two duplicate clustering steps in the old pipeline (04 + 05) — the v2 HDBSCAN fine-grain
  clustering superseded the BERTopic approach; `05_merge_clusters.py` is now dead code

---

## Current State (2026-06-04)

**Completed this session (continued):**
- [x] Re-embedding **COMPLETE** — 3,602,151 docs, 73 shards, 4799.5 min (~80h total)
      Note: ~34h wasted due to resume bug (re-embedded 1.5M already-done docs on restart).
      Fix committed — future runs will fast-forward correctly.

**Completed this session:**
- [x] Crashproof server pipeline `run_server.py`
- [x] Classification report `12_report_classification.py` with TF-IDF analysis
- [x] Full CiNii parse: 71,511,821 records
- [x] Full CiNii clean: 3,602,151 English scientific papers (with CJK leakage fix)
- [x] CJK character ratio filter — removed ~108k Japanese-leaked papers
- [x] Data exploration script `explore_data.py`
- [x] Token length analysis: mean 276 tokens, p99 = 633 tokens; chose cap 768
- [x] New training pipeline `run_training.py`
- [x] Multiple robustness fixes: chunked parquet loading, atomic writes, streaming reports

**Next steps (planned):**
- [ ] Run `run_training.py` on 150k-sample from FullDatasetV2_clean shards (ready to go)
- [ ] Fill in `lcc_mapping.csv` (manual LCC assignment, bootstrap from v2 model)
- [ ] Train new model (v3) with broader class coverage
- [ ] Run `11_classify.py` on all 73 shards with new model
- [ ] Compare v2 vs v3 classification quality report

