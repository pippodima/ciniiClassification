"""
pipeline/09_map_lcc_divisions.py
================================
Maps each of the 169 clusters to an LCC numeric division (3rd-level classification).

LCC hierarchy used:
  Level 1  –  main class        e.g. T
  Level 2  –  subclass          e.g. TJ
  Level 3  –  numeric division  e.g. TJ266  ← this script assigns this

Assignments are based on Claude's knowledge of the LCC schedule.
Each entry includes:
  lcc_division  – the numeric division code
  lcc_div_desc  – short description of that division
  div_confidence – H / M / L  (High / Medium / Low confidence in numeric precision)
"""

import pandas as pd
from pathlib import Path

OUT_DIR = Path("clustering_output/v2")

# ---------------------------------------------------------------------------
# FULL DIVISION MAPPING
# Format: cluster_id → (lcc_division, description, confidence)
# Confidence: H = authoritative / well-known range
#             M = correct sub-area, numeric interpolated
#             L = best-guess given subclass, topic uncertain from JATS noise
# ---------------------------------------------------------------------------
DIVISION_MAP = {
    # ── QE  Geology ──────────────────────────────────────────────────────────
    0:  ("QE576",   "Glaciology – glacial sediments, ice sheets",                    "H"),
    12: ("QE522",   "Volcanology – volcanic phenomena, magma, Iwo Jima",             "H"),
    13: ("QE522",   "Volcanology – eruptions, magmatic activity, Japan",             "H"),
    77: ("QE515",   "Geochemistry – stable/radiogenic isotopes (Li, Sr, Ce)",        "H"),

    # ── QC  Physics ───────────────────────────────────────────────────────────
    1:  ("QC225",   "Wave motion / elastic waves – Rayleigh, scattering",            "H"),
    3:  ("QC815",   "Geomagnetism – core-mantle dynamo, internal field",             "H"),
    4:  ("QC815",   "Geomagnetism – main field modelling (IGRF, secular variation)", "H"),
    6:  ("QC849",   "Paleomagnetism – dipole, reversals, VGP",                       "H"),
    7:  ("QC849",   "Paleomagnetism – remanent magnetization, paleointensity",       "H"),
    8:  ("QC849",   "Rock magnetism – titanomagnetite, NRM, coercivity",             "H"),
    9:  ("QC809",   "Geophysics – magnetotellurics, resistivity, anomalies",         "H"),
    17: ("QC809",   "Geophysics / seismology – fault slip, stress drop, rupture",    "H"),
    18: ("QC809",   "Geophysics / seismology – crustal structure, seismic velocity", "H"),
    19: ("QC809",   "Geophysics / seismology – aftershocks, earthquake statistics",  "H"),
    20: ("QC809",   "Geophysics – thermal/magnetic properties of Earth materials",   "M"),
    84: ("QC809",   "Geophysics – GPS/GNSS geodesy, precipitable water vapour",      "H"),
    105:("QC718",   "Plasma physics – magnetic reconnection, current sheets",        "H"),
    106:("QC718",   "Plasma physics – electron/ion beam discharges",                 "H"),
    107:("QC791",   "Thermonuclear fusion – magnetic confinement (tokamak)",         "H"),
    108:("QB464",   "Solar cosmic rays – galactic and solar energetic particles",    "H"),  # QB subclass override noted
    114:("QC993",   "Tropical meteorology – monsoon, rainfall, convection",          "H"),
    116:("QC975",   "Atmospheric chemistry – stratospheric ozone, aerosols",         "H"),
    117:("QC880",   "Dynamic meteorology – atmospheric gravity waves",               "H"),
    122:("QC881",   "Ionosphere – equatorial electrojet, solar-ionospheric coupling","H"),
    125:("QC881",   "Ionosphere – geomagnetic storms, auroral ionosphere",           "H"),

    # ── QB  Astronomy ─────────────────────────────────────────────────────────
    2:  ("QB592",   "Moon – crater formation, surface mineralogy, ejecta",           "H"),
    79: ("QB651",   "Minor planets / asteroids – dust, solar system debris",         "H"),
    120:("QB521",   "Solar physics – solar atmosphere, airglow, irradiance",         "H"),
    121:("QB529",   "Solar wind – magnetospheric waves, VLF, pulsations",            "H"),
    124:("QB529",   "Solar wind – magnetosphere dynamics, substorms, IMF",           "H"),

    # ── QA  Mathematics / Computer Science ───────────────────────────────────
    5:  ("QA76.73", "Programming / markup languages – MathML annotation semantics",  "M"),
    22: ("QA76.87", "Machine learning – mathematical semantics, neural networks",    "H"),
    24: ("QA164",   "Combinatorics / graph theory – scheduling, NP problems",        "H"),
    27: ("QA268",   "Coding theory – cryptography, authentication, encryption",      "H"),
    28: ("QA76.9",  "Computer science – computer vision, image recognition",         "H"),
    29: ("QA268",   "Coding theory – image/video compression, watermarking",         "H"),
    30: ("QA76.87", "Machine learning – neural networks, fuzzy logic",               "H"),
    32: ("QA76.9",  "Computer science – database systems, query languages",          "H"),

    # ── TK  Electrical / Electronic Engineering ───────────────────────────────
    10: ("TK7871",  "Antennas and waveguides – microstrip, slot, resonators",        "H"),
    11: ("TK5102",  "Signal processing – adaptive filters, speech, noise reduction", "H"),
    33: ("TK7874",  "Integrated circuits – VLSI, CMOS, chip architecture",           "H"),
    39: ("TK5103",  "Wireless communications – CDMA, channel, handoff protocols",   "H"),
    40: ("TK5105",  "Computer networks – QoS, ATM, packet routing, wireless",        "H"),
    41: ("TK5102",  "Communications – error-correcting codes, convolutional codes",  "H"),
    42: ("TK5103",  "Wireless communications – CDMA, OFDM, fading channels",        "H"),
    73: ("TK7887",  "Optical storage – magneto-optical recording, disk technology",  "H"),
    74: ("TK7878",  "Scanning probe microscopy – AFM, STM, cantilevers",             "H"),
    78: ("TK7872",  "Electronic materials – high-Tc superconductors, Cu-oxide",      "H"),
    80: ("TK7874",  "Semiconductor fabrication – e-beam / optical lithography",      "H"),
    82: ("TK7872",  "Magnetic thin films – anisotropy, coercivity, spintronics",     "H"),
    83: ("TK7872",  "Liquid crystal materials – nematic, ferroelectric LCs",         "H"),
    86: ("TK7874",  "Semiconductor devices – FET, TFT, MOSFET, CMOS",               "H"),
    89: ("TK7872",  "Organic optoelectronics – OLEDs, light-emitting substrates",    "H"),
    90: ("TK2960",  "Photovoltaics – dye-sensitised and perovskite solar cells",     "H"),
    98: ("TK8315",  "Lasers and fibre optics – semiconductor lasers, waveguides",    "H"),
    100:("TK7874",  "Semiconductor epitaxy – GaAs/ZnSe/ZnO compound films",         "H"),
    110:("TK7874",  "Semiconductor devices – MOSFET gate control, tunnelling",       "H"),
    113:("TK7872",  "Electronic ceramics – piezoelectric, ferroelectric ceramics",   "H"),
    115:("TK7874",  "Thin film deposition – sputtering, CVD, dielectric films",      "H"),
    118:("TK7874",  "Silicon thin films – SiH4 deposition, amorphous silicon",       "H"),
    119:("TK7874",  "Compound semiconductor epitaxy – GaAs, GaN, InP growth",       "H"),

    # ── TJ  Mechanical Engineering ────────────────────────────────────────────
    14: ("TJ170",   "Applied mechanics – ultrasonic stress analysis, NDE",           "M"),
    15: ("TJ266",   "Turbines / turbomachinery – gas turbine thermodynamic cycles",  "H"),
    16: ("TJ790",   "Internal combustion engines – diesel combustion, flame",        "H"),
    21: ("TJ1075",  "Lubrication – oil film, journal bearings, tribology",           "H"),
    23: ("TJ1338",  "Machine elements – bolted joints, flanges, fasteners",          "M"),
    25: ("TJ900",   "Hydraulic machinery – valves, pipelines, hydraulic systems",    "H"),
    26: ("TJ267",   "Gas dynamics – shock waves, supersonic nozzle flow",            "H"),
    31: ("TJ267",   "Fluid mechanics – gas-liquid two-phase flow, void fraction",    "H"),
    34: ("TJ260",   "Heat transfer engineering – boiling, heat flux, film boiling",  "H"),
    35: ("TJ260",   "Heat transfer engineering – convection, fins, cylinder",        "H"),
    37: ("TJ267",   "Fluid mechanics – pipe flow, Reynolds number, laminar/turbulent","H"),
    43: ("TJ267",   "Fluid mechanics – jet flow, reattachment, nozzle turbulence",   "H"),
    44: ("TJ266",   "Turbomachinery – impellers, blades, diffusers, stall",          "H"),
    45: ("TJ267",   "Fluid mechanics – turbulent boundary layers, vortices",         "H"),
    47: ("TJ184",   "Gears – spur gear fatigue, tooth bending, rolling contact",     "H"),
    54: ("TJ175",   "Dynamics of machinery – vibration dampers, absorbers",          "H"),
    55: ("TJ211",   "Mechanisms / robotics – linkages, manipulators, robot arms",    "H"),
    61: ("TJ175",   "Dynamics of machinery – self-excited vibration, wheel-rail",    "H"),
    62: ("TJ175",   "Dynamics of machinery – modal analysis, natural frequencies",   "H"),
    63: ("TJ175",   "Dynamics of machinery – rotating shaft, critical speed, rotor", "H"),

    # ── TN  Mining / Metallurgy ───────────────────────────────────────────────
    36: ("TN693",   "Physical metallurgy – Fe-based alloy substitution, microstructure","H"),
    38: ("TN693",   "Physical metallurgy – hydrogen embrittlement, steel fracture",  "H"),
    46: ("TN752",   "Surface treatment / corrosion – Zn-Cr coatings, steel",        "H"),
    50: ("TN730",   "Steelmaking – continuous casting, mould, tundish, solidification","H"),
    51: ("TN730",   "Steelmaking – inclusions, slag, Al2O3-MgO-CaO chemistry",      "H"),
    52: ("TN706",   "Ironmaking – decarburisation, desulfurisation, hot metal",      "H"),
    53: ("TN693",   "Physical metallurgy – deoxidation, Fe-Ti-Mn alloy chemistry",  "H"),
    56: ("TN706",   "Blast furnace – coke combustion, raceway, burden materials",    "H"),
    57: ("TN706",   "Ironmaking – iron ore sintering, blast furnace burden",         "H"),
    58: ("TN690",   "Solidification metallurgy – dendrites, peritectic reactions",   "H"),
    59: ("TN677",   "Slag – steelmaking slag chemistry, CaO-SiO2-Al2O3 systems",    "H"),
    60: ("TN677",   "Slag – viscosity, alumina-silica-calcia flux systems",          "H"),
    68: ("TN693",   "Physical metallurgy – austenite, martensite, microstructure",   "H"),
    69: ("TN752",   "Welding metallurgy – arc welding, heat input, weld properties", "H"),
    81: ("TN693",   "Advanced materials – carbon nanotubes, graphene composites",    "M"),

    # ── TA  Civil / Structural Engineering ───────────────────────────────────
    64: ("TA418",   "Composite materials – fibre-reinforced, failure analysis",      "H"),
    65: ("TA409",   "Fracture mechanics – fatigue crack growth, stress intensity",   "H"),
    66: ("TA405",   "Mechanics of materials – stress-strain, plasticity, hardening", "H"),
    67: ("TA660",   "Structural analysis – shell and plate vibration, frequencies",  "H"),
    70: ("TA418",   "Materials testing – piezoelectric stress analysis, fracture",   "H"),
    71: ("TA660",   "Structural analysis – cylindrical shells, elastic stress",      "H"),

    # ── TS  Manufactures / Manufacturing Engineering ──────────────────────────
    48: ("TS560",   "Metal cutting – tool wear, grinding, carbide tools",            "H"),
    49: ("TS320",   "Sheet metalwork – rolling, deep drawing, sheet forming",        "H"),

    # ── QD  Chemistry ─────────────────────────────────────────────────────────
    85: ("QD701",   "Photochemistry – photocatalysis, TiO2 nanoparticles",           "H"),
    88: ("QD381",   "Polymer chemistry – polymer synthesis, substitution",           "H"),
    91: ("QD571",   "Electrochemistry – electrocatalysis, oxygen reduction, Pt",     "H"),
    92: ("QD571",   "Electrochemistry – biosensors, glucose detection",              "H"),
    93: ("QD571",   "Electrochemistry – supercapacitors, EDLC, electrolyte",        "H"),
    94: ("QD571",   "Electrochemistry – lithium-ion batteries, electrode materials", "H"),
    95: ("QD79",    "Analytical chemistry – LC-MS, chromatography, toxins",          "H"),
    97: ("QD415",   "Biochemistry – structure-activity relationships, drug design",  "H"),
    99: ("QD96",    "Mass spectrometry – ion sources, drift tube, instrumentation",  "H"),
    101:("QD476",   "Physical organic chemistry – synthesis, reaction mechanisms",   "M"),
    102:("QD415",   "Biochemistry – molecular chemistry, biomolecule synthesis",     "M"),
    103:("QD96",    "Mass spectrometry – ionisation, mass spectra methods",          "H"),
    109:("QD415",   "Biochemistry – enzyme activity, glycosides, antioxidants",      "H"),

    # ── QH  Biology / Natural History ─────────────────────────────────────────
    87: ("QH506",   "Molecular biology – protein structure, DNA binding, folding",   "H"),
    132:("QH471",   "Reproductive biology – sperm, oocyte, gene expression",        "H"),
    133:("QH491",   "Developmental biology – neural crest, embryogenesis",           "H"),
    140:("QH301",   "Systematic biology – 16S rRNA, microbial taxonomy, phylogeny",  "H"),
    151:("QH506",   "Molecular biology – enzyme activity, E. coli proteins",        "H"),
    165:("QH301",   "Biology – microbial strains, E. coli isolates",                "M"),
    166:("QH634",   "Cell biology – endocytosis, actin, protein trafficking",       "H"),
    167:("QH450",   "Molecular genetics – DNA-protein binding, gene regulation",     "H"),
    168:("QH506",   "Molecular biology – protein-gene cluster analysis",             "H"),

    # ── QK  Botany ────────────────────────────────────────────────────────────
    148:("QK495",   "Systematic botany – Arabidopsis, angiosperms, plant taxonomy",  "H"),

    # ── QL  Zoology ───────────────────────────────────────────────────────────
    147:("QL112",   "Zoogeography / ecology – species distribution, forest fauna",  "M"),

    # ── QP  Physiology ────────────────────────────────────────────────────────
    137:("QP363",   "Neurophysiology – Ca2+ channels, ion channels, electrophysiology","H"),
    142:("QP364",   "Neurochemistry – neurotransmitters, synaptic communication",    "H"),
    162:("QP401",   "Neuropsychology – emotion processing, cortical stimuli",        "H"),

    # ── QR  Microbiology ──────────────────────────────────────────────────────
    158:("QR180",   "Immunology – host-pathogen interactions, macrophage infection", "H"),
    159:("QR185",   "Immunology – T lymphocytes, Treg, immune regulation",           "H"),

    # ── RC  Internal Medicine ─────────────────────────────────────────────────
    72: ("RC271",   "Oncology – cancer genetics, mutations in patients",             "M"),
    104:("RC78",    "Diagnostic radiology – X-ray beam imaging, dose, MLC",          "H"),
    111:("RC660",   "Endocrinology – diabetes mellitus, insulin, glucose",           "H"),
    112:("RC648",   "Endocrine diseases – hormonal aspects of pregnancy/fetal dev",  "M"),
    126:("RC372",   "Neurology – epilepsy, seizure management",                      "H"),
    127:("RC31",    "Internal medicine – general (JATS-contaminated cluster)",       "L"),
    128:("RC685",   "Cardiovascular – renin-angiotensin system, hypertension",       "H"),
    129:("RC924",   "Connective tissue diseases – SLE, autoimmune (JATS noise)",    "M"),
    130:("RC685",   "Cardiovascular – coronary artery disease, risk assessment",     "H"),
    131:("RC388",   "Cerebrovascular disease – cerebral ischaemia, stroke, BBB",    "H"),
    134:("RC523",   "Alzheimer's disease – amyloid, tau, dementia",                  "H"),
    135:("RA645",   "Disaster medicine – tsunami, flood, emergency preparedness",    "H"),
    136:("RC927",   "Musculoskeletal – muscle pain, myalgia, pain management",       "H"),
    138:("RC848",   "Liver diseases – hepatitis B/C, H. pylori, viral hepatitis",   "H"),
    139:("RC685",   "Cardiovascular – blood pressure, hypertension monitoring",      "H"),
    141:("RD598",   "Vascular surgery – endovascular aneurysm repair, stent",        "H"),
    143:("RC643",   "Blood diseases – leukaemia, AML (JATS-contaminated)",           "M"),
    144:("RC685",   "Cardiovascular – myocardial infarction, valve disease",         "H"),
    145:("RC685",   "Cardiovascular – arrhythmia, tachycardia, pacing",              "H"),
    146:("RC685",   "Cardiovascular – heart failure diagnosis, left ventricle",      "H"),
    149:("RC848",   "Liver diseases – HBV, HCV, HIV viral hepatitis",               "H"),
    150:("RC607",   "Infectious diseases – HIV/AIDS, viral pathogenesis, influenza", "H"),
    152:("RC591",   "Respiratory diseases – asthma, allergic airway, IL cytokines",  "H"),
    153:("RC521",   "Geriatrics / dementia care (JATS-contaminated cluster)",        "L"),
    154:("RC268",   "Oncology – tumour immunology, CD8/CD4 immune response",         "H"),
    155:("RC254",   "Neoplasms – cancer cell biology, tumour proliferation",         "H"),
    160:("RC262",   "Oncology – cancer survival analysis, carcinoma outcomes",       "H"),
    161:("RC254",   "Neoplasms – cancer management (JATS-contaminated)",             "M"),
    163:("RC569",   "Psychiatry – suicide risk, anxiety, depression",                "H"),

    # ── RK  Dentistry ────────────────────────────────────────────────────────
    75: ("RK667",   "Dental implants – osseointegration, implant design",            "H"),
    76: ("RK361",   "Periodontics – periodontal staging, bone health",               "H"),

    # ── RM  Pharmacology ─────────────────────────────────────────────────────
    96: ("RM147",   "Drug delivery – liposomes, controlled release systems",         "H"),

    # ── RA  Public Health ─────────────────────────────────────────────────────
    # (cluster 135 remapped to RA645 above)

    # ── HA  Statistics ────────────────────────────────────────────────────────
    156:("HA31",    "Statistical theory – inference, meta-analysis, regression",     "H"),

    # ── HD  Industries / Economics ────────────────────────────────────────────
    157:("HD2741",  "Corporate organisation – market dynamics, governance",          "H"),

    # ── LB  Education ────────────────────────────────────────────────────────
    164:("LB1049",  "Teaching methods – language learning strategies, nursing students","H"),

    # ── UNKNOWN ───────────────────────────────────────────────────────────────
    123:("UNKNOWN", "JATS-noise cluster – no assignable topic",                      "L"),
}

# ---------------------------------------------------------------------------
# Merge into lcc_mapping and save
# ---------------------------------------------------------------------------
df = pd.read_parquet(OUT_DIR / "lcc_mapping.parquet")

# Build division columns
div_codes  = []
div_descs  = []
div_confs  = []

for cid in df["cluster_id"]:
    if cid in DIVISION_MAP:
        code, desc, conf = DIVISION_MAP[cid]
    else:
        code, desc, conf = ("MISSING", "Not mapped", "L")
    div_codes.append(code)
    div_descs.append(desc)
    div_confs.append(conf)

df["lcc_division"]      = div_codes
df["lcc_div_desc"]      = div_descs
df["div_confidence"]    = div_confs

# Derive lcc_main from subclass (first letter(s) before the number)
def main_class(code):
    import re
    m = re.match(r"([A-Z]+)", str(code))
    return m.group(1) if m else "?"

df["lcc_main"] = df["lcc"].apply(main_class)

# Re-order columns for readability
cols = ["cluster_id", "n_papers", "label", "keywords",
        "lcc_main", "lcc", "lcc_division", "lcc_div_desc", "div_confidence",
        "lcc_claude", "lcc_ollama", "agree"]
existing = [c for c in cols if c in df.columns]
df = df[existing + [c for c in df.columns if c not in existing]]

df.to_parquet(OUT_DIR / "lcc_mapping_full.parquet", index=False)
df.to_csv(OUT_DIR / "lcc_mapping_full.csv", index=False)

# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
print("=" * 65)
print("  LCC DIVISION ASSIGNMENT SUMMARY")
print("=" * 65)
total = len(df)
conf_counts = df["div_confidence"].value_counts()
missing = (df["lcc_division"] == "MISSING").sum()
unknown = (df["lcc_division"] == "UNKNOWN").sum()

print(f"  Total clusters      : {total}")
print(f"  High confidence (H) : {conf_counts.get('H', 0)}")
print(f"  Medium conf.  (M)   : {conf_counts.get('M', 0)}")
print(f"  Low conf.     (L)   : {conf_counts.get('L', 0)}")
print(f"  MISSING / not found : {missing}")
print(f"  UNKNOWN (JATS noise): {unknown}")
print()

# Division distribution
print("  DIVISION DISTRIBUTION")
print(f"  {'Division':<12} {'Desc':<52} {'N':>4}  {'C'}")
print("  " + "-" * 72)
div_grp = df[~df["lcc_division"].isin(["MISSING", "UNKNOWN"])].groupby(
    ["lcc_division", "lcc_div_desc", "div_confidence"]
)["n_papers"].sum().reset_index().sort_values("n_papers", ascending=False)

for _, row in div_grp.iterrows():
    desc = row["lcc_div_desc"][:50]
    print(f"  {row['lcc_division']:<12} {desc:<52} {row['n_papers']:>4}  {row['div_confidence']}")

print()
print("  Saved:")
print(f"    {OUT_DIR / 'lcc_mapping_full.parquet'}")
print(f"    {OUT_DIR / 'lcc_mapping_full.csv'}")
