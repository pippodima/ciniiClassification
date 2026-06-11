"""
pipeline/label_journal_gold_firstpass.py
=========================================
Apply the first-pass (Claude-assigned) journal→LCC gold labels to a
journal_gold_template.csv produced by 16_build_journal_gold.py, and emit TWO
convention variants so we can measure how the applied-physics QC↔TK choice moves
the accuracy:

  <out-dir>/journal_gold_QC.csv   applied-physics journals labeled QC (physics)
  <out-dir>/journal_gold_TK.csv   same, but JJAP / APL / J.Appl.Phys → TK

Labels are assigned from journal SCOPE (independent of the model), keyed by
journal name so row order is irrelevant. 187/200 labeled; the 13 genuinely
multidisciplinary journals are left blank (excluded from scoring). This script is
the reproducible record of the labeling — review/adjust GOLD_SUB then re-run.

Usage (server):
    python pipeline/label_journal_gold_firstpass.py \
        --template reports/journal_validation_v3_300k/journal_gold_template.csv \
        --out-dir  reports/journal_validation_v3_300k
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Journals where the QC (physics) vs TK (applied/electronics) call is genuine and
# high-volume (~180k papers). QC variant keeps them QC; TK variant flips to TK.
APPLIED_PHYSICS = [
    "Japanese Journal of Applied Physics",
    "Applied Physics Letters",
    "Journal of Applied Physics",
]

# journal name → gold LCC subclass (first-pass, from journal scope)
GOLD_SUB = {
    "Japanese Journal of Applied Physics": "QC",
    "Applied Physics Letters": "QC",
    "Journal of Applied Physics": "QC",
    "The Journal of Chemical Physics": "QD",
    "Bulletin of the Chemical Society of Japan": "QD",
    "Journal of the Physical Society of Japan": "QC",
    "CHEMICAL & PHARMACEUTICAL BULLETIN": "QD",
    "The Journal of Immunology": "QR",
    "Chemistry Letters": "QD",
    "The Journal of Biochemistry": "QP",
    "Blood": "RC",
    "The Journal of Neuroscience": "QP",
    "Journal of Virology": "QR",
    "Circulation": "RC",
    "Applied and Environmental Microbiology": "QR",
    "Journal of Bacteriology": "QR",
    "Angewandte Chemie International Edition": "QD",
    "YAKUGAKU ZASSHI": "QD",
    "FEBS Letters": "QP",
    "Chemistry – A European Journal": "QD",
    "Internal Medicine": "RC",
    "IEICE Proceeding Series": "TK",
    "Bioscience, Biotechnology, and Biochemistry": "QP",
    "Biochemical Journal": "QP",
    "Biological & Pharmaceutical Bulletin": "RM",
    "Cancer Research": "RC",
    "Journal of Clinical Oncology": "RC",
    "日本金属学会誌": "TN",
    "Development": "QH",
    "The Tohoku Journal of Experimental Medicine": "RC",
    "Journal of Geophysical Research: Atmospheres": "QC",
    "MATERIALS TRANSACTIONS": "TN",
    "Infection and Immunity": "QR",
    "The Journal of Physiology": "QP",
    "Review of Scientific Instruments": "QC",
    "Journal of Geophysical Research: Space Physics": "QC",
    "Agricultural and Biological Chemistry": "QP",
    "ISIJ International": "TN",
    "Journal of the American Ceramic Society": "TP",
    "Stroke": "RC",
    "Journal of Applied Physiology": "QP",
    "Journal of Clinical Microbiology": "QR",
    "Journal of Neurochemistry": "QP",
    "Journal of Neurophysiology": "QP",
    "Tetsu-to-Hagane": "TN",
    "Okayama Igakkai Zasshi (Journal of Okayama Medical Association)": "RC",
    "Journal of Cell Science": "QH",
    "Chemical & pharmaceutical bulletin": "QD",
    "Antimicrobial Agents and Chemotherapy": "QR",
    "Clinical Cancer Research": "RC",
    "Journal of Geophysical Research: Solid Earth": "QE",
    "Circulation Research": "RC",
    "British Journal of Pharmacology": "RM",
    "The Astrophysical Journal": "QB",
    "Journal of Neurosurgery": "RD",
    "Journal of Veterinary Medical Science": "SF",
    "Analytical Sciences": "QD",
    "Transactions of the Society of Instrument and Control Engineers": "TJ",
    "RSC Advances": "QD",
    "Diabetes": "RC",
    "Advanced Functional Materials": "TK",
    "Journal of Applied Polymer Science": "QD",
    "Journal of the Ceramic Society of Japan": "TP",
    "American Journal of Physiology-Heart and Circulatory Physiology": "QP",
    "International Journal of Cancer": "RC",
    "Chemical Communications": "QD",
    "Journal of Comparative Neurology": "QP",
    "Circulation Journal": "RC",
    "JOURNAL OF CHEMICAL ENGINEERING OF JAPAN": "TP",
    "European Journal of Biochemistry": "QP",
    "Journal of Fluid Mechanics": "TA",
    "Plant Physiology": "QK",
    "Hypertension": "RC",
    "Physics of Plasmas": "QC",
    "Molecular Biology of the Cell": "QH",
    "Neurologia medico-chirurgica": "RD",
    "Genetics": "QH",
    "Bulletin of JSME": "TJ",
    "Molecular Microbiology": "QR",
    "IEICE Transactions on Fundamentals of Electronics, Communications and Computer Sciences": "TK",
    "IEICE Transactions on Information and Systems": "QA",
    "IEICE Transactions on Communications": "TK",
    "The Journal of Antibiotics": "QR",
    "The Plant Journal": "QK",
    "Advanced Materials": "TA",
    "The Japanese Journal of Pharmacology": "RM",
    "The Journal of Cell Biology": "QH",
    "Genes & Development": "QH",
    "IEICE transactions on fundamentals of electronics, communications and computer sciences": "TK",
    "European Journal of Neuroscience": "QP",
    "Pediatrics": "RJ",
    "CYTOLOGIA": "QH",
    "Journal of Experimental Biology": "QP",
    "Atmospheric Chemistry and Physics": "QC",
    "Diabetes Care": "RC",
    "IEICE transactions on communications": "TK",
    "Arteriosclerosis, Thrombosis, and Vascular Biology": "RC",
    "Journal of Materials Chemistry A": "TK",
    "Journal of Physical Therapy Science": "RM",
    "Plant and cell physiology": "QK",
    "Angewandte Chemie": "QD",
    "American Journal of Physiology-Cell Physiology": "QP",
    "Japanese Circulation Journal": "RC",
    "Canadian Journal of Chemistry": "QD",
    "Journal of Geophysical Research: Oceans": "GC",
    "European Journal of Organic Chemistry": "QD",
    "Cancer": "RC",
    "IEICE Electronics Express": "TK",
    "Bioinformatics": "QH",
    "Journal of Vacuum Science & Technology A: Vacuum, Surfaces, and Films": "TK",
    "The Journal of cell biology": "QH",
    "Polymer Journal": "QD",
    "Physical Chemistry Chemical Physics": "QD",
    "The Laryngoscope": "RF",
    "Journal of Vacuum Science & Technology B: Microelectronics and Nanometer Structures Processing, Measurement, and Phenomena": "TK",
    "Journal of Dental Research": "RK",
    "The Journal of Experimental Medicine": "QR",
    "Arthritis & Rheumatism": "RC",
    "動脈硬化": "RC",
    "Biotechnology and Bioengineering": "TP",
    "IEICE transactions on electronics": "TK",
    "New Phytologist": "QK",
    "Journal of Cellular Physiology": "QH",
    "American Journal of Physiology-Regulatory, Integrative and Comparative Physiology": "QP",
    "Journal of Food Science": "TP",
    "Journal of Bone and Mineral Research": "RC",
    "British Journal of Haematology": "RC",
    "Annals of Neurology": "RC",
    "Physics of Fluids": "TA",
    "Endocrine Journal": "RC",
    "Water Resources Research": "GB",
    "European Journal of Immunology": "QR",
    "American Journal of Physiology-Renal Physiology": "QP",
    "PROCEEDINGS OF HYDRAULIC ENGINEERING": "TC",
    "Journal of Nutritional Science and Vitaminology": "QP",
    "American Journal of Physiology-Endocrinology and Metabolism": "QP",
    "Journal of Polymer Science Part A: Polymer Chemistry": "QD",
    "physica status solidi (b)": "QC",
    "MICROBIOLOGY and IMMUNOLOGY": "QR",
    "Journal of Climate": "QC",
    "Endocrinology": "QP",
    "Journal of the Society of Naval Architects of Japan": "VM",
    "Japanese Heart Journal": "RC",
    "Clinical Chemistry": "RB",
    "Journal of High Energy Physics": "QC",
    "Helvetica Chimica Acta": "QD",
    "Journal of Periodontology": "RK",
    "MRS Proceedings": "TA",
    "Advanced Synthesis & Catalysis": "QD",
    "American Journal of Physiology-Gastrointestinal and Liver Physiology": "QP",
    "Sen'i Gakkaishi": "TS",
    "Medical Physics": "R",
    "Astronomy & Astrophysics": "QB",
    "IEICE transactions on information and systems": "QA",
    "Journal of Nuclear Science and Technology": "TK",
    "Molecular Ecology": "QH",
    "British Journal of Surgery": "RD",
    "Chemical Science": "QD",
    "The Physics of Fluids": "QC",
    "Limnology and Oceanography": "QH",
    "日本鉱業会誌": "TN",
    "Monthly Notices of the Royal Astronomical Society": "QB",
    "Journal of Materials Research": "TA",
    "Journal of Endocrinology": "QP",
    "Epilepsia": "RC",
    "Proceedings of the Royal Society B: Biological Sciences": "QH",
    "British Journal of Nutrition": "QP",
    "Journal of Leukocyte Biology": "QR",
    "Magnetic Resonance in Medicine": "R",
    "Journal of Experimental Medicine": "QR",
    "Journal of Photopolymer Science and Technology": "TK",
    "Thrombosis and Haemostasis": "RC",
    "IEICE Transactions on Electronics": "TK",
    "Journal of Cellular Biochemistry": "QP",
    "Nanoscale": "QD",
    "Journal of Cerebral Blood Flow & Metabolism": "RC",
    "CHEMOTHERAPY": "RM",
    "Journal of the American Oil Chemists' Society": "TP",
    "The American Journal of Sports Medicine": "RD",
    "気象集誌. 第2輯": "QC",
    "The FEBS Journal": "QP",
    "AIChE Journal": "TP",
    "Electrical Engineering in Japan": "TK",
    "European Journal of Inorganic Chemistry": "QD",
    "Genome Research": "QH",
    "Journal of the American Geriatrics Society": "RC",
    "Endocrinologia Japonica": "QP",
}


def write_variant(template: pd.DataFrame, applied_physics_sub: str, out: Path):
    df = template.copy()
    sub_map = dict(GOLD_SUB)
    for j in APPLIED_PHYSICS:
        sub_map[j] = applied_physics_sub
    df["gold_lcc_sub"] = df["journal"].map(lambda j: sub_map.get(j, ""))
    df["gold_lcc_main"] = df["gold_lcc_sub"].map(lambda s: s[0] if s else "")
    df["gold_lcc_div"] = ""
    df.to_csv(out, index=False)
    labeled = (df["gold_lcc_sub"] != "").sum()
    missed = [j for j in GOLD_SUB if j not in set(template["journal"])]
    print(f"  {out.name}: {labeled}/{len(df)} labeled  (applied-physics→{applied_physics_sub})")
    if missed:
        print(f"    ⚠ {len(missed)} label keys not found in template (name mismatch): "
              f"{missed[:3]}{'…' if len(missed) > 3 else ''}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", required=True, help="journal_gold_template.csv")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tmpl = pd.read_csv(args.template)
    print(f"  template: {len(tmpl)} journals")
    write_variant(tmpl, "QC", out_dir / "journal_gold_QC.csv")
    write_variant(tmpl, "TK", out_dir / "journal_gold_TK.csv")
    print("  → score each with 17_score_vs_journal_gold.py and compare")


if __name__ == "__main__":
    main()
