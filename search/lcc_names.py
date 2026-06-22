"""
search/lcc_names.py
===================
Human-readable Library of Congress Classification names, so search facets show
"Q — Science > QD — Chemistry" instead of bare codes. Covers all main classes
and the subclasses that occur in this corpus (model vocab + journal-gold set).
Extend SUB as needed; unknown codes fall back to the bare code.
"""
from __future__ import annotations

MAIN: dict[str, str] = {
    "A": "General Works",
    "B": "Philosophy, Psychology, Religion",
    "C": "Auxiliary Sciences of History",
    "D": "World History",
    "E": "American History",
    "F": "Local History of the Americas",
    "G": "Geography, Anthropology, Recreation",
    "H": "Social Sciences",
    "J": "Political Science",
    "K": "Law",
    "L": "Education",
    "M": "Music",
    "N": "Fine Arts",
    "P": "Language and Literature",
    "Q": "Science",
    "R": "Medicine",
    "S": "Agriculture",
    "T": "Technology",
    "U": "Military Science",
    "V": "Naval Science",
    "Z": "Bibliography, Library Science",
}

SUB: dict[str, str] = {
    # B — Philosophy / Psychology
    "BF": "Psychology",
    # G — Geography / Anthropology
    "GB": "Physical Geography",
    "GC": "Oceanography",
    "GN": "Anthropology",
    # H — Social Sciences
    "HD": "Industries, Land Use, Labor",
    "HE": "Transportation and Communications",
    "HT": "Communities, Classes, Races",
    # J — Political Science
    "JZ": "International Relations",
    # L — Education
    "LB": "Theory and Practice of Education",
    # P — Language and Literature
    "P":  "Philology and Linguistics",
    "PE": "English Language",
    # Q — Science
    "QA": "Mathematics and Computer Science",
    "QB": "Astronomy",
    "QC": "Physics",
    "QD": "Chemistry",
    "QE": "Geology and Earth Sciences",
    "QH": "Biology",
    "QK": "Botany",
    "QL": "Zoology",
    "QP": "Physiology",
    "QR": "Microbiology",
    # R — Medicine
    "R":  "Medicine (General)",
    "RA": "Public Health",
    "RB": "Pathology",
    "RC": "Internal Medicine",
    "RD": "Surgery",
    "RE": "Ophthalmology",
    "RF": "Otorhinolaryngology",
    "RG": "Gynecology and Obstetrics",
    "RJ": "Pediatrics",
    "RK": "Dentistry",
    "RM": "Therapeutics and Pharmacology",
    "RS": "Pharmacy",
    "RT": "Nursing",
    # S — Agriculture
    "S":  "Agriculture (General)",
    "SF": "Animal Culture and Veterinary",
    # T — Technology
    "TA": "Civil and General Engineering",
    "TC": "Hydraulic Engineering",
    "TJ": "Mechanical Engineering",
    "TK": "Electrical Engineering and Electronics",
    "TN": "Mining and Metallurgy",
    "TP": "Chemical Technology",
    "TS": "Manufactures",
    "TX": "Home Economics",
    # V — Naval Science
    "VM": "Naval Architecture and Marine Engineering",
}


def main_label(code: str) -> str:
    code = (code or "").strip()
    return f"{code} — {MAIN[code]}" if code in MAIN else code


def sub_label(code: str) -> str:
    code = (code or "").strip()
    return f"{code} — {SUB[code]}" if code in SUB else code


def levels(main: str, sub: str, div: str) -> dict | None:
    """Build the hierarchical-facet object InstantSearch expects (lvl0/1/2)."""
    main = (main or "").strip()
    sub = (sub or "").strip()
    div = (div or "").strip()
    if not main:
        return None
    lvl0 = main_label(main)
    out = {"lvl0": lvl0}
    if sub:
        out["lvl1"] = f"{lvl0} > {sub_label(sub)}"
        if div:
            out["lvl2"] = f"{out['lvl1']} > {div}"
    return out
