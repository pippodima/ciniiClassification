#!/usr/bin/env python3
"""
08_map_lcc.py — Dual LCC mapping: Claude's assignments vs Ollama's
===================================================================
Two independent sources assign an LCC code to each of the 169 clusters:

  1. CLAUDE  — hard-coded assignments based on reading every cluster's
               label + keywords directly (domain knowledge, no LLM bias)

  2. OLLAMA  — qwen3.5:0.8b with an improved prompt that explicitly
               steers toward T-class (engineering) and R-class (medicine)
               instead of defaulting to QC for everything

At the end, the two are compared: agreement rate, disagreement list,
and a final "consensus" column that uses Claude's assignment when they
disagree (more precise domain knowledge) or when Ollama returns UNKNOWN.

Outputs: clustering_output/v2/
  lcc_mapping.parquet / .csv   — full table with both assignments
  lcc_mapping_review.csv       — clusters where the two disagree
"""

import sys, re, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
import pandas as pd
from tqdm import tqdm
from config import CLUSTER_DIR

META_PATH = CLUSTER_DIR / "v2" / "cluster_metadata.parquet"
OUT_DIR   = CLUSTER_DIR / "v2"

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3.5:0.8b"


# ══════════════════════════════════════════════════════════════════════════════
# 1. CLAUDE'S ASSIGNMENTS
#    Read every cluster label + keywords above, assigned independently.
#    Reasoning is noted inline for the non-obvious ones.
# ══════════════════════════════════════════════════════════════════════════════

# Format: cluster_id → (lcc_code, brief_reason)
CLAUDE_ASSIGNMENTS = {
    # ── Geoscience / Solid Earth ──────────────────────────────────────────────
    0:  ("QE", "glaciology / sedimentology"),
    1:  ("QC", "seismic wave propagation / geophysics"),
    2:  ("QB", "planetary science / lunar mineralogy"),
    3:  ("QC", "geomagnetic dynamo / core physics"),
    4:  ("QC", "geomagnetic field modeling"),
    6:  ("QC", "paleomagnetism / geomagnetic reversals"),
    7:  ("QC", "paleomagnetism / rock magnetism"),
    8:  ("QC", "rock magnetism / titanomagnetite"),
    9:  ("QC", "geophysical EM / magnetotelluric"),
    12: ("QE", "volcanology"),
    13: ("QE", "volcanology / Japan"),
    17: ("QC", "seismology / earthquake mechanics"),
    18: ("QC", "seismology / crustal structure"),
    19: ("QC", "seismology / earthquake dynamics"),
    20: ("QC", "soil thermal/magnetic properties"),
    77: ("QE", "lithium isotope geochemistry"),
    84: ("QC", "geodesy / GPS / GNSS"),

    # ── Space physics / Astronomy ─────────────────────────────────────────────
    79:  ("QB", "asteroids, galaxies, dust — astrophysics"),
    108: ("QB", "cosmic rays / solar physics"),
    120: ("QB", "airglow / upper atmosphere / solar radiation"),
    121: ("QB", "magnetospheric VLF waves / space physics"),
    122: ("QC", "ionospheric electrojet / geomagnetism"),
    124: ("QB", "solar wind / magnetosphere"),
    125: ("QC", "ionospheric storms / space weather"),

    # ── Atmospheric / Climate science ─────────────────────────────────────────
    114: ("QC", "tropical meteorology / monsoon"),
    116: ("QC", "stratospheric aerosols / ozone"),
    117: ("QC", "atmospheric gravity waves"),

    # ── Plasma / Nuclear physics ─────────────────────────────────────────────
    105: ("QC", "magnetic reconnection / plasma physics"),
    106: ("QC", "plasma beam / ion beam physics"),
    107: ("QC", "magnetic confinement fusion / tokamak"),

    # ── Electrical engineering / Electronics ──────────────────────────────────
    10:  ("TK", "antenna / waveguide / microwave engineering"),
    11:  ("TK", "digital signal processing / adaptive filters"),
    27:  ("QA", "cryptography / network security"),   # QA covers computer science
    28:  ("QA", "computer vision / image recognition"),
    29:  ("QA", "image/video compression, watermarking"),
    30:  ("QA", "neural networks / machine learning"),
    32:  ("QA", "software / database / query languages"),
    33:  ("TK", "VLSI / CMOS / circuit design"),
    39:  ("TK", "CDMA / wireless communications"),
    40:  ("TK", "network QoS / ATM / wireless networking"),
    41:  ("TK", "error-correcting codes / convolutional codes"),
    42:  ("TK", "CDMA / OFDM / fading channels"),
    73:  ("TK", "optical disk recording / magneto-optical"),
    74:  ("TK", "AFM / scanning tunneling microscopy"),
    80:  ("TK", "electron beam lithography / nanofabrication"),
    82:  ("TK", "magnetic thin films / spintronics"),
    83:  ("TK", "liquid crystals / display technology"),
    86:  ("TK", "field effect transistors / TFTs"),
    89:  ("TK", "organic LEDs / phosphor emitters"),
    90:  ("TK", "solar cells / photovoltaics / perovskites"),
    98:  ("TK", "laser physics / photonics / waveguides"),
    100: ("TK", "ZnSe/GaAs epitaxial semiconductor growth"),
    110: ("TK", "MOSFETs / semiconductor gate"),
    115: ("TK", "thin film deposition / BST / PZT"),
    118: ("TK", "silicon film deposition / amorphous Si"),
    119: ("TK", "GaAs / InP / GaN epitaxy"),
    78:  ("TK", "superconductors / YBaCuO"),
    113: ("TK", "piezoelectric ceramics / ferroelectrics"),

    # ── Mechanical engineering ────────────────────────────────────────────────
    15:  ("TJ", "gas turbines / thermodynamic cycles"),
    16:  ("TJ", "diesel engines / combustion"),
    21:  ("TJ", "journal bearings / lubrication / tribology"),
    23:  ("TJ", "bolted joints / mechanical joints"),
    25:  ("TJ", "hydraulic systems / valves / piping"),
    26:  ("TJ", "shock waves / compressible flow / nozzles"),
    31:  ("TJ", "gas-liquid two-phase flow"),
    34:  ("TJ", "boiling heat transfer"),
    35:  ("TJ", "convective heat transfer / fins"),
    37:  ("TJ", "pipe flow / Reynolds / fluid dynamics"),
    43:  ("TJ", "jet flow / reattachment / turbulence"),
    44:  ("TJ", "turbomachinery / impeller / pump"),
    45:  ("TJ", "turbulent boundary layers / fluid dynamics"),
    47:  ("TJ", "gear fatigue / tribology"),
    48:  ("TS", "metal cutting / tool wear / machining"),
    49:  ("TS", "rolling / sheet metal / metal forming"),
    54:  ("TJ", "dynamic vibration absorbers"),
    55:  ("TJ", "robotic mechanisms / kinematic links"),
    61:  ("TJ", "wheel-rail vibration / friction-induced"),
    62:  ("TJ", "vibration analysis / nonlinear dynamics"),
    63:  ("TJ", "rotor dynamics / rotating machinery"),
    64:  ("TA", "composite material fracture / structural"),
    65:  ("TA", "fatigue crack growth / fracture mechanics"),
    66:  ("TA", "plasticity / deformation / constitutive"),
    67:  ("TA", "shell vibration / structural mechanics"),
    70:  ("TA", "stress intensity / fracture mechanics"),
    71:  ("TA", "elasticity / shell stress analysis"),
    14:  ("TJ", "ultrasonic testing / stress waves"),

    # ── Materials / Metallurgy ────────────────────────────────────────────────
    36:  ("TN", "Fe alloys / grain boundary / phase"),
    38:  ("TN", "hydrogen embrittlement in steel"),
    46:  ("TN", "corrosion / protective coatings / steel"),
    50:  ("TN", "continuous casting / solidification"),
    51:  ("TN", "steel inclusions / aluminosilicate"),
    52:  ("TN", "iron metallurgy / decarburization"),
    53:  ("TN", "Fe-Ti-Si alloy / deoxidation"),
    56:  ("TN", "blast furnace / coke / ironmaking"),
    57:  ("TN", "iron ore sintering / blast furnace"),
    58:  ("TN", "solidification / alloy dendrites"),
    59:  ("TN", "steelmaking slag chemistry"),
    60:  ("TN", "slag viscosity / silicate chemistry"),
    68:  ("TN", "steel microstructure / phase transformation"),
    69:  ("TN", "arc welding / heat-affected zone"),
    81:  ("TN", "carbon nanotubes / composites"),

    # ── Chemistry ─────────────────────────────────────────────────────────────
    85:  ("QD", "photocatalysis / TiO2 nanoparticles"),
    88:  ("QD", "polymer chemistry / copolymers"),
    91:  ("QD", "fuel cell catalysts / oxygen reduction"),
    92:  ("QD", "electrochemical biosensors"),
    93:  ("QD", "electrochemical capacitors / supercapacitors"),
    94:  ("QD", "lithium-ion batteries / electrolytes"),
    95:  ("QD", "analytical chemistry / LC-MS"),
    97:  ("QD", "structure-activity / medicinal chemistry"),
    99:  ("QD", "mass spectrometry / ion physics"),
    101: ("QD", "organic synthesis / reactions"),
    102: ("QD", "molecular chemistry"),
    103: ("QD", "mass spectrometry / ionization"),
    109: ("QD", "natural products / tyrosinase inhibitors"),

    # ── Biology / Molecular biology ───────────────────────────────────────────
    87:  ("QH", "protein folding / structural biology"),
    132: ("QH", "reproductive biology / embryology"),
    133: ("QH", "developmental biology / neural crest"),
    140: ("QH", "microbiology / 16S rRNA / taxonomy"),
    147: ("QL", "ecology / forest species"),       # zoology/ecology
    148: ("QK", "plant biology / Arabidopsis"),
    151: ("QH", "microbiology / enzyme / E.coli"),
    165: ("QH", "microbiology / bacterial strains"),
    166: ("QH", "cell biology / protein trafficking"),
    167: ("QH", "molecular biology / gene regulation"),
    168: ("QH", "molecular biology / gene clusters"),

    # ── Medicine ──────────────────────────────────────────────────────────────
    72:  ("RC", "clinical genetics / patient mutations"),
    75:  ("RK", "dental implants"),
    76:  ("RK", "periodontology / cartilage"),
    96:  ("RM", "drug delivery / pharmacokinetics"),
    104: ("RC", "radiation therapy / X-ray imaging"),
    111: ("RC", "diabetes / endocrinology"),
    112: ("RC", "obstetrics / fetal development"),
    126: ("RC", "epilepsy / seizure management"),
    127: ("RC", "clinical medicine — general JATS content"),
    128: ("RC", "angiotensin / hypertension / pharmacology"),
    129: ("RC", "clinical medicine — general"),
    130: ("RC", "coronary artery disease / cardiology"),
    131: ("RC", "cerebral ischemia / neurology"),
    134: ("RC", "Alzheimer's disease / neurology"),
    135: ("RA", "disaster risk / public health"),
    136: ("RC", "pain management / neurology"),
    137: ("QP", "calcium channels / physiology"),
    138: ("RC", "hepatitis / infectious disease"),
    139: ("RC", "cardiovascular / blood pressure"),
    141: ("RD", "endovascular surgery / aneurysm"),
    142: ("QP", "neuroscience / synaptic transmission"),
    143: ("RC", "hematology / leukemia / AML"),
    144: ("RC", "cardiology / myocardial infarction"),
    145: ("RC", "cardiac arrhythmia / electrophysiology"),
    146: ("RC", "heart failure / cardiology"),
    149: ("RC", "hepatitis B/C / infectious disease"),
    150: ("RC", "HIV / virology / infectious disease"),
    152: ("RC", "asthma / allergic disease / immunology"),
    153: ("RC", "dementia / geriatric medicine"),
    154: ("RC", "tumor immunology / cancer immunotherapy"),
    155: ("RC", "cancer cell biology / oncology"),
    158: ("QR", "infection immunology / macrophages"),
    159: ("QR", "T-cell immunology / immune response"),
    160: ("RC", "oncology / cancer survival"),
    161: ("RC", "oncology / clinical cancer"),
    162: ("QP", "cognitive neuroscience / emotion"),
    163: ("RC", "mental health / psychiatry"),
    164: ("LB", "nursing education / language learning"),

    # ── Social sciences / Other ───────────────────────────────────────────────
    157: ("HD", "corporate economics / market dynamics"),
    156: ("HA", "statistical inference / meta-analysis"),

    # ── Ambiguous / JATS-noise dominated ──────────────────────────────────────
    5:   ("QA", "MML/MathML annotation — math markup language"),
    22:  ("QA", "MML semantics — math markup for ML"),
    24:  ("QA", "graph algorithms / scheduling — computer science"),
}

LCC_DESCRIPTIONS = {
    "QA": "Mathematics / Computer Science",
    "QB": "Astronomy / Astrophysics",
    "QC": "Physics (geophysics, magnetism, plasma, optics, acoustics)",
    "QD": "Chemistry",
    "QE": "Geology / Earth sciences",
    "QH": "Biology / Natural history",
    "QK": "Botany",
    "QL": "Zoology / Ecology",
    "QM": "Human anatomy",
    "QP": "Physiology / Neuroscience",
    "QR": "Microbiology / Immunology",
    "R":  "Medicine — general",
    "RA": "Public health / Epidemiology",
    "RB": "Pathology",
    "RC": "Internal medicine / Neurology / Oncology / Cardiology",
    "RD": "Surgery",
    "RK": "Dentistry",
    "RM": "Pharmacology / Drug delivery",
    "RS": "Pharmacy",
    "TA": "Engineering — structural / civil / materials mechanics",
    "TC": "Hydraulic / Ocean engineering",
    "TD": "Environmental engineering",
    "TJ": "Mechanical engineering (fluid, thermal, vibration, robotics)",
    "TK": "Electrical / Electronic / Photonic engineering",
    "TL": "Aeronautics / Space engineering",
    "TN": "Mining / Metallurgy / Materials processing",
    "TP": "Chemical technology / Polymers",
    "TS": "Manufacturing / Machining / Precision engineering",
    "VM": "Naval / Marine engineering",
    "G":  "Geography / Geodesy / Remote sensing",
    "HA": "Statistics",
    "HB": "Economic theory",
    "HD": "Industries / Labour economics",
    "HF": "Commerce / Business",
    "HM": "Sociology",
    "LB": "Education",
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. IMPROVED OLLAMA PROMPT
# ══════════════════════════════════════════════════════════════════════════════

LCC_LIST_STR = "\n".join(f"  {k}: {v}" for k, v in LCC_DESCRIPTIONS.items())

DISAMBIGUATION_RULES = """
IMPORTANT DISAMBIGUATION RULES — read carefully before choosing:

• Use TJ  (not QC) for: engines, turbines, combustion, pumps, compressors, fluid machinery,
  heat exchangers, lubrication, bearings, vibration in machinery, robotics, piping systems.
• Use TK  (not QC) for: electronics, circuits, semiconductors, antennas, signal processing,
  lasers, photonics, optical fibers, solar cells, batteries, sensors, VLSI, CMOS, wireless
  communications, superconductors, magnetic films, liquid crystals, transistors.
• Use TN  (not QC) for: steelmaking, ironmaking, blast furnace, casting, welding, corrosion,
  metallurgy, metal processing, alloy design, slag chemistry.
• Use TA  (not QC) for: structural mechanics, fracture mechanics, composites, fatigue, stress
  analysis, finite element analysis of structures.
• Use TS  (not TJ) for: machining, cutting, tool wear, metal forming, rolling, stamping.
• Use QD  (not QC) for: chemistry — organic synthesis, electrochemistry, analytical chemistry,
  batteries/capacitors (the chemistry side), chromatography, spectroscopy.
• Use QC  ONLY for: physics — mechanics, thermodynamics, acoustics, optics, electromagnetism
  (as a science), plasma physics, condensed matter physics, geophysics, atmospheric physics.
• Use RC  (not QH) for: clinical medicine — cardiology, neurology, oncology, diabetes,
  infectious disease, psychiatry, any study involving patients/treatment/disease.
• Use QR  (not QH) for: microbiology, virology, immunology — especially when the focus is
  on pathogens, bacteria, immune cells, rather than clinical patients.
• Use QP  (not QH) for: physiology — calcium channels, neurotransmitters, synaptic biology,
  hormones, when studied at the cellular/molecular level without clinical patient context.
• Use QH  for: ecology, evolution, genetics (non-clinical), cell biology, developmental
  biology, natural history.
• Use QA  for: mathematics, algorithms, computer science theory, cryptography, machine learning,
  image processing, databases, software engineering.
"""


def ask_ollama(prompt: str, timeout: int = 30) -> str:
    for attempt in range(3):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "think": False,
                      "stream": False,
                      "options": {"temperature": 0}},
                timeout=timeout,
            )
            resp.raise_for_status()
            return re.sub(r"<think>.*?</think>", "", resp.json()["message"]["content"],
                          flags=re.DOTALL).strip()
        except requests.exceptions.Timeout:
            print(f"    timeout ({attempt+1}/3)…")
            time.sleep(3)
        except Exception as e:
            print(f"    error: {e}")
            break
    return "{}"


def map_ollama(row: dict) -> dict:
    kw_str = ", ".join(row["keywords"][:12])
    prompt = (
        f"You are a library classifier using Library of Congress Classification.\n\n"
        f"{DISAMBIGUATION_RULES}\n"
        f"Available LCC codes:\n{LCC_LIST_STR}\n\n"
        f"Cluster to classify:\n"
        f"  Label      : {row['label']}\n"
        f"  Description: {str(row.get('description',''))[:200]}\n"
        f"  Keywords   : {kw_str}\n\n"
        f"Apply the disambiguation rules strictly. "
        f"Reply JSON only: "
        f'{{ "lcc": "TK", "confidence": 4, "reason": "one sentence" }}'
    )
    raw = ask_ollama(prompt)
    try:
        m    = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
    except Exception:
        data = {}

    lcc = str(data.get("lcc", "")).strip().upper()
    if lcc not in LCC_DESCRIPTIONS:
        for code in LCC_DESCRIPTIONS:
            if lcc.startswith(code):
                lcc = code
                break
        else:
            lcc = "UNKNOWN"
    return {
        "lcc_ollama":      lcc,
        "conf_ollama":     int(data.get("confidence", 1)),
        "reason_ollama":   str(data.get("reason", "")),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. RUN BOTH AND COMPARE
# ══════════════════════════════════════════════════════════════════════════════
meta = pd.read_parquet(str(META_PATH))
print(f"Mapping {len(meta)} clusters — Claude's assignments + Ollama …\n")

rows = []
for _, row in tqdm(meta.iterrows(), total=len(meta), desc="LCC mapping"):
    cid  = int(row["cluster_id"])
    d    = row.to_dict()

    # Claude
    claude_entry = CLAUDE_ASSIGNMENTS.get(cid)
    if claude_entry:
        lcc_claude, reason_claude = claude_entry
    else:
        lcc_claude, reason_claude = "UNKNOWN", "not assigned"

    # Ollama
    ollama_res = map_ollama(d)

    # Consensus: prefer Claude when they disagree or Ollama is UNKNOWN
    agree = (lcc_claude == ollama_res["lcc_ollama"])
    if ollama_res["lcc_ollama"] == "UNKNOWN":
        consensus = lcc_claude
    elif not agree:
        consensus = lcc_claude   # Claude has explicit domain reasoning
    else:
        consensus = lcc_claude   # same anyway

    flag = "" if agree else " ≠"
    print(f"  [{cid:3d}] {row['label'][:50]:<50}  "
          f"Claude={lcc_claude:6s}  Ollama={ollama_res['lcc_ollama']:6s}"
          f"  conf={ollama_res['conf_ollama']}{flag}")

    rows.append({
        "cluster_id":    cid,
        "n_papers":      int(row["n_papers"]),
        "label":         row["label"],
        "keywords":      row["keywords"][:8],
        # Claude
        "lcc_claude":    lcc_claude,
        "reason_claude": reason_claude,
        # Ollama
        "lcc_ollama":    ollama_res["lcc_ollama"],
        "conf_ollama":   ollama_res["conf_ollama"],
        "reason_ollama": ollama_res["reason_ollama"],
        # Consensus
        "lcc":           consensus,
        "lcc_desc":      LCC_DESCRIPTIONS.get(consensus, ""),
        "agree":         agree,
    })

df = pd.DataFrame(rows)


# ── Save ──────────────────────────────────────────────────────────────────────
df.to_parquet(str(OUT_DIR / "lcc_mapping.parquet"), index=False)
df.drop(columns=["keywords"]).to_csv(str(OUT_DIR / "lcc_mapping.csv"), index=False)

disagree = df[~df["agree"]]
disagree.drop(columns=["keywords"]).to_csv(str(OUT_DIR / "lcc_mapping_review.csv"), index=False)


# ── Agreement report ──────────────────────────────────────────────────────────
n_total   = len(df)
n_agree   = df["agree"].sum()
n_disagree= n_total - n_agree
n_unknown_ollama = (df["lcc_ollama"] == "UNKNOWN").sum()
n_unknown_claude = (df["lcc_claude"] == "UNKNOWN").sum()

print("\n" + "=" * 65)
print("  AGREEMENT REPORT")
print("=" * 65)
print(f"  Total clusters     : {n_total}")
print(f"  Agree              : {n_agree}  ({n_agree/n_total:.1%})")
print(f"  Disagree           : {n_disagree}  ({n_disagree/n_total:.1%})")
print(f"  Ollama UNKNOWN     : {n_unknown_ollama}")
print(f"  Claude UNKNOWN     : {n_unknown_claude}")

print(f"\n  DISAGREEMENTS ({n_disagree} clusters):")
print(f"  {'ID':>4}  {'Claude':>8}  {'Ollama':>8}  Label")
print(f"  {'-'*60}")
for _, r in disagree.sort_values("cluster_id").iterrows():
    print(f"  [{int(r['cluster_id']):3d}]  {r['lcc_claude']:>8}  "
          f"{r['lcc_ollama']:>8}  {r['label'][:45]}")

print(f"\n  CONSENSUS LCC DISTRIBUTION (used for training):")
dist = (df.groupby(["lcc", "lcc_desc"])["n_papers"]
        .sum().reset_index().sort_values("n_papers", ascending=False))
for _, r in dist.iterrows():
    n_cl = (df["lcc"] == r["lcc"]).sum()
    bar  = "█" * min(int(r["n_papers"] / 300), 40)
    print(f"  {r['lcc']:6s}  {r['lcc_desc'][:45]:<45}  "
          f"{int(r['n_papers']):>5} papers  {n_cl:>3} clusters  {bar}")

print(f"\n  Saved:")
print(f"    {OUT_DIR / 'lcc_mapping.parquet'}")
print(f"    {OUT_DIR / 'lcc_mapping.csv'}")
print(f"    {OUT_DIR / 'lcc_mapping_review.csv'}  ← {n_disagree} disagreements to review")
