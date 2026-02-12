"""Medical abbreviation dictionaries for normalization.

This module contains comprehensive medical abbreviation mappings used by
the MedicalAbbreviationNormalizer. Dictionaries are organized by category
and pattern type (standard uppercase, dotted forms, slash forms, etc.).

The abbreviation list is intentionally large. For RAG evaluation, the cost
of a missed normalization (false negative → LLM judge marks a correct claim
as unsupported) is higher than the cost of an incorrect expansion, so we
err on the side of coverage.
"""

from __future__ import annotations

from dataclasses import dataclass

# ========================== Data Structures =================================


@dataclass
class AmbiguousEntry:
    """One possible meaning of an ambiguous abbreviation."""

    full_form: str
    context_keywords: list[str]
    priority: int = 0  # Higher = more common meaning


# ========================== Abbreviation Dictionary =========================
#
# Each key maps to either:
#   - A plain string (unambiguous abbreviation), or
#   - A list of AmbiguousEntry (context-dependent disambiguation).
# ---------------------------------------------------------------------------

# fmt: off
ABBREVIATIONS: dict[str, str | list[AmbiguousEntry]] = {
    # ── Cardiology ──────────────────────────────────────────────────────
    "CHF":   "congestive heart failure",
    "MI":    "myocardial infarction",
    "CABG":  "coronary artery bypass graft",
    "STEMI": "ST-elevation myocardial infarction",
    "NSTEMI": "non-ST-elevation myocardial infarction",
    "PCI":   "percutaneous coronary intervention",
    "EKG":   "electrocardiogram",
    "ECG":   "electrocardiogram",
    "EF":    "ejection fraction",
    "LVEF":  "left ventricular ejection fraction",
    "AFIB":  "atrial fibrillation",
    "SVT":   "supraventricular tachycardia",
    "VT":    "ventricular tachycardia",
    "VF":    "ventricular fibrillation",
    "ACS":   "acute coronary syndrome",
    "HTN":   "hypertension",
    "BP":    "blood pressure",
    "SBP":   "systolic blood pressure",
    "DBP":   "diastolic blood pressure",
    "HR":    "heart rate",
    "PAD":   "peripheral artery disease",
    "DVT":   "deep vein thrombosis",
    "PVD":   "peripheral vascular disease",
    "MVP":   "mitral valve prolapse",
    "AS": [
        AmbiguousEntry("aortic stenosis", ["valve", "heart", "murmur", "echo", "cardiac"], 1),
        AmbiguousEntry("ankylosing spondylitis", ["spine", "back", "joint", "rheum", "hla"], 0),
    ],

    # ── Pulmonology ─────────────────────────────────────────────────────
    "COPD":  "chronic obstructive pulmonary disease",
    "SOB":   "shortness of breath",
    "DOE":   "dyspnea on exertion",
    "ABG":   "arterial blood gas",
    "CXR":   "chest X-ray",
    "CT":    "computed tomography",
    "PFT":   "pulmonary function test",
    "FEV":   "forced expiratory volume",
    "FVC":   "forced vital capacity",
    "OSA":   "obstructive sleep apnea",
    "ARDS":  "acute respiratory distress syndrome",
    "TB":    "tuberculosis",
    "PE": [
        AmbiguousEntry("pulmonary embolism", ["clot", "dvt", "anticoag", "lung", "d-dimer", "ctpa", "embol"], 1),
        AmbiguousEntry("physical exam", ["exam", "assessment", "evaluation", "noted on", "unremarkable"], 0),
        AmbiguousEntry("pleural effusion", ["effusion", "fluid", "tap", "thorecent", "chest tube"], 0),
    ],
    "CP": [
        AmbiguousEntry("chest pain", ["heart", "cardiac", "acs", "troponin", "ekg", "substernal", "angina", "exertion", "sob", "shortness", "dyspnea", "chest", "stemi", "nstemi", "nitro", "mi", "cath", "stress test", "pressure"], 1),
        AmbiguousEntry("cerebral palsy", ["neuro", "motor", "spastic", "developmental", "pediatric", "gait", "tone"], 0),
    ],

    # ── Neurology ───────────────────────────────────────────────────────
    "CVA":   "cerebrovascular accident",
    "TIA":   "transient ischemic attack",
    "TBI":   "traumatic brain injury",
    "SAH":   "subarachnoid hemorrhage",
    "SDH":   "subdural hematoma",
    "EDH":   "epidural hematoma",
    "ICH":   "intracerebral hemorrhage",
    "LOC":   "loss of consciousness",
    "GCS":   "Glasgow Coma Scale",
    "ICP":   "intracranial pressure",
    "EEG":   "electroencephalogram",
    "LP":    "lumbar puncture",
    "CSF":   "cerebrospinal fluid",
    "ALS":   "amyotrophic lateral sclerosis",
    "CNS":   "central nervous system",
    "PNS":   "peripheral nervous system",
    "MRI":   "magnetic resonance imaging",
    "MS": [
        AmbiguousEntry("multiple sclerosis", ["neuro", "brain", "lesion", "demyelinat", "relaps", "white matter"], 1),
        AmbiguousEntry("mitral stenosis", ["valve", "heart", "murmur", "echo", "rheumatic", "cardiac"], 0),
        AmbiguousEntry("morphine sulfate", ["pain", "mg", "dose", "iv", "prn", "opioid", "narcotic"], 0),
        AmbiguousEntry("mental status", ["exam", "oriented", "alert", "confused", "ams"], 0),
    ],

    # ── Endocrinology ───────────────────────────────────────────────────
    "DM":    "diabetes mellitus",
    "IDDM":  "insulin-dependent diabetes mellitus",
    "NIDDM": "non-insulin-dependent diabetes mellitus",
    "DKA":   "diabetic ketoacidosis",
    "HBA1C": "hemoglobin A1c",
    "A1C":   "hemoglobin A1c",
    "TSH":   "thyroid-stimulating hormone",
    "FBS":   "fasting blood sugar",
    "BG":    "blood glucose",
    "FSH":   "follicle-stimulating hormone",
    "LH":    "luteinizing hormone",

    # ── Gastroenterology ────────────────────────────────────────────────
    "GERD":  "gastroesophageal reflux disease",
    "GI":    "gastrointestinal",
    "IBD":   "inflammatory bowel disease",
    "IBS":   "irritable bowel syndrome",
    "LFT":   "liver function test",
    "UGI":   "upper gastrointestinal",
    "EGD":   "esophagogastroduodenoscopy",
    "ERCP":  "endoscopic retrograde cholangiopancreatography",
    "NPO":   "nothing by mouth",
    "NGT":   "nasogastric tube",
    "BM":    "bowel movement",

    # ── Nephrology / Urology ────────────────────────────────────────────
    "CKD":   "chronic kidney disease",
    "ESRD":  "end-stage renal disease",
    "GFR":   "glomerular filtration rate",
    "BUN":   "blood urea nitrogen",
    "UTI":   "urinary tract infection",
    "UA":    "urinalysis",
    "HD":    "hemodialysis",
    "AKI":   "acute kidney injury",
    "ARF":   "acute renal failure",

    # ── Hematology / Oncology ───────────────────────────────────────────
    "CBC":   "complete blood count",
    "WBC":   "white blood cell",
    "RBC":   "red blood cell",
    "HGB":   "hemoglobin",
    "HCT":   "hematocrit",
    "PLT":   "platelet",
    "PT":    "prothrombin time",
    "PTT":   "partial thromboplastin time",
    "INR":   "international normalized ratio",
    "ESR":   "erythrocyte sedimentation rate",
    "CRP":   "C-reactive protein",
    "BMP":   "basic metabolic panel",
    "CMP":   "comprehensive metabolic panel",
    "LDH":   "lactate dehydrogenase",
    "DIC":   "disseminated intravascular coagulation",
    "AML":   "acute myeloid leukemia",
    "CLL":   "chronic lymphocytic leukemia",
    "CML":   "chronic myeloid leukemia",
    "NHL":   "non-Hodgkin lymphoma",

    # ── Infectious Disease ──────────────────────────────────────────────
    "HIV":   "human immunodeficiency virus",
    "AIDS":  "acquired immunodeficiency syndrome",
    "MRSA":  "methicillin-resistant Staphylococcus aureus",
    "VRE":   "vancomycin-resistant Enterococcus",
    "CDI":   "Clostridioides difficile infection",
    "URI":   "upper respiratory infection",
    "STI":   "sexually transmitted infection",
    "STD":   "sexually transmitted disease",
    "HSV":   "herpes simplex virus",
    "CMV":   "cytomegalovirus",
    "EBV":   "Epstein-Barr virus",
    "RSV":   "respiratory syncytial virus",
    "HCV":   "hepatitis C virus",
    "HBV":   "hepatitis B virus",
    "HAV":   "hepatitis A virus",

    # ── Pharmacology / Orders ───────────────────────────────────────────
    "PRN":   "as needed",
    "BID":   "twice a day",
    "TID":   "three times a day",
    "QID":   "four times a day",
    "QD":    "once a day",
    "QHS":   "every night at bedtime",
    "STAT":  "immediately",
    "PO":    "by mouth",
    "IV":    "intravenous",
    "IM":    "intramuscular",
    "SQ":    "subcutaneous",
    "SL":    "sublingual",
    "DC":    "discontinue",
    "RX":    "prescription",
    "OTC":   "over the counter",
    "NSAID": "nonsteroidal anti-inflammatory drug",
    "SSRI":  "selective serotonin reuptake inhibitor",
    "SNRI":  "serotonin-norepinephrine reuptake inhibitor",
    "MAOI":  "monoamine oxidase inhibitor",
    "TCA":   "tricyclic antidepressant",
    "ACE":   "angiotensin-converting enzyme",
    "ARB":   "angiotensin receptor blocker",
    "CCB":   "calcium channel blocker",
    "PPI":   "proton pump inhibitor",

    # ── Surgery / Procedures ────────────────────────────────────────────
    "TAH":   "total abdominal hysterectomy",
    "TKR":   "total knee replacement",
    "THR":   "total hip replacement",
    "ORIF":  "open reduction internal fixation",
    "TURP":  "transurethral resection of prostate",

    # ── Emergency / Critical Care ───────────────────────────────────────
    "CPR":   "cardiopulmonary resuscitation",
    "AED":   "automated external defibrillator",
    "ACLS":  "advanced cardiac life support",
    "BLS":   "basic life support",
    "ROSC":  "return of spontaneous circulation",
    "ICU":   "intensive care unit",
    "CCU":   "coronary care unit",
    "MICU":  "medical intensive care unit",
    "SICU":  "surgical intensive care unit",
    "NICU":  "neonatal intensive care unit",
    "PICU":  "pediatric intensive care unit",
    "ED":    "emergency department",
    "ER":    "emergency room",
    "EMS":   "emergency medical services",
    "MVA":   "motor vehicle accident",
    "MVC":   "motor vehicle collision",
    "GSW":   "gunshot wound",

    # ── Anatomical Locations ────────────────────────────────────────────
    "RUQ":   "right upper quadrant",
    "LUQ":   "left upper quadrant",
    "RLQ":   "right lower quadrant",
    "LLQ":   "left lower quadrant",
    "RUE":   "right upper extremity",
    "LUE":   "left upper extremity",
    "RLE":   "right lower extremity",
    "LLE":   "left lower extremity",
    "LUL":   "left upper lobe",
    "RUL":   "right upper lobe",
    "LLL":   "left lower lobe",
    "RLL":   "right lower lobe",
    "RML":   "right middle lobe",

    # ── Psychiatry ──────────────────────────────────────────────────────
    "MDD":   "major depressive disorder",
    "GAD":   "generalized anxiety disorder",
    "PTSD":  "post-traumatic stress disorder",
    "OCD":   "obsessive-compulsive disorder",
    "BPD":   "borderline personality disorder",
    "ADHD":  "attention deficit hyperactivity disorder",
    "SI":    "suicidal ideation",
    "HI":    "homicidal ideation",
    "AVH":   "auditory visual hallucinations",
    "ASD":   "autism spectrum disorder",
    "ECT":   "electroconvulsive therapy",

    # ── Orthopedics ─────────────────────────────────────────────────────
    "ACL":   "anterior cruciate ligament",
    "PCL":   "posterior cruciate ligament",
    "MCL":   "medial collateral ligament",
    "MCP":   "metacarpophalangeal",
    "PIP":   "proximal interphalangeal",
    "DIP":   "distal interphalangeal",
    "CMC":   "carpometacarpal",
    "OA":    "osteoarthritis",
    "RA":    "rheumatoid arthritis",

    # ── OB/GYN ──────────────────────────────────────────────────────────
    "LMP":   "last menstrual period",
    "EDD":   "estimated date of delivery",
    "EDC":   "estimated date of confinement",
    "GDM":   "gestational diabetes mellitus",
    "PROM":  "premature rupture of membranes",
    "PPROM": "preterm premature rupture of membranes",
    "IUGR":  "intrauterine growth restriction",
    "IVF":   "in vitro fertilization",
    "IUI":   "intrauterine insemination",
    "NSVD":  "normal spontaneous vaginal delivery",
    "LTCS":  "low transverse cesarean section",

    # ── Radiology / Imaging ─────────────────────────────────────────────
    "US":    [
        AmbiguousEntry("ultrasound", ["imaging", "scan", "sonogr", "doppler", "echo", "radiol", "abdomen", "pelvi"], 1),
    ],
    "XR":    "X-ray",
    "KUB":   "kidneys, ureters, and bladder",
    "CTPA":  "CT pulmonary angiography",
    "CTA":   [
        AmbiguousEntry("clear to auscultation", ["lung", "breath", "chest", "bilat", "wheez", "pulm"], 1),
        AmbiguousEntry("CT angiography", ["scan", "imaging", "radiol", "vessel", "steno", "aneurysm"], 0),
    ],
    "MRA":   "magnetic resonance angiography",
    "PET":   "positron emission tomography",
    "DEXA":  "dual-energy X-ray absorptiometry",

    # ── Additional Labs / Vitals ────────────────────────────────────────
    "ABX":   "antibiotics",
    "AST":   "aspartate aminotransferase",
    "ALT":   "alanine aminotransferase",
    "ALP":   "alkaline phosphatase",
    "GGT":   "gamma-glutamyl transferase",
    "BNP":   "B-type natriuretic peptide",
    "PSA":   "prostate-specific antigen",
    "CEA":   "carcinoembryonic antigen",
    "AFP":   "alpha-fetoprotein",
    "HCG":   "human chorionic gonadotropin",
    "TIBC":  "total iron-binding capacity",
    "UIBC":  "unsaturated iron-binding capacity",
    "RDW":   "red cell distribution width",
    "MPV":   "mean platelet volume",
    "MCV":   "mean corpuscular volume",
    "MCH":   "mean corpuscular hemoglobin",
    "MCHC":  "mean corpuscular hemoglobin concentration",
    "ANA":   "antinuclear antibody",
    "ANCA":  "antineutrophil cytoplasmic antibody",
    "RF":    "rheumatoid factor",
    "CPK":   "creatine phosphokinase",
    "CK":    "creatine kinase",
    "MAP":   "mean arterial pressure",
    "CVP":   "central venous pressure",
    "SVR":   "systemic vascular resistance",

    # ── Diabetes subtypes ───────────────────────────────────────────────
    "T1DM":  "type 1 diabetes mellitus",
    "T2DM":  "type 2 diabetes mellitus",

    # ── General / Documentation ─────────────────────────────────────────
    "CC":    "chief complaint",
    "HPI":   "history of present illness",
    "PMH":   "past medical history",
    "PSH":   "past surgical history",
    "FH":    "family history",
    "SH":    "social history",
    "ROS":   "review of systems",
    "WNL":   "within normal limits",
    "NAD":   "no acute distress",
    "AAO":   "awake, alert, and oriented",
    "AMA":   "against medical advice",
    "DNR":   "do not resuscitate",
    "DNI":   "do not intubate",
    "ADL":   "activities of daily living",
    "BMI":   "body mass index",
    "ROM":   "range of motion",
    "VSS":   "vital signs stable",
    "HEENT": "head, eyes, ears, nose, and throat",
    "PERRLA": "pupils equal, round, reactive to light and accommodation",
    "RRR":   "regular rate and rhythm",
    "CTAB":  "clear to auscultation bilaterally",
    "NKA":   "no known allergies",
    "NKDA":  "no known drug allergies",
}
# fmt: on


# ── Dotted / lowercase dosing & route abbreviations ────────────────────────
# These are commonly written as "q.d.", "b.i.d.", "p.r.n.", "p.o.", etc.
# We normalize all surface variants (q.d, q.d., QD, Q.D.) to the same
# canonical key, then expand from here.
#
# Key = canonical dotted form (lowercase, with periods between each letter,
# no trailing period).  The matching regex handles all variants.
# ---------------------------------------------------------------------------
DOTTED_ABBREVIATIONS: dict[str, str] = {
    # ── Frequency ───────────────────────────────────────────────────────
    "q.d": "once a day",
    "q.o.d": "every other day",
    "b.i.d": "twice a day",
    "t.i.d": "three times a day",
    "q.i.d": "four times a day",
    "q.h.s": "every night at bedtime",
    "q.h": "every hour",
    "q.2.h": "every 2 hours",
    "q.4.h": "every 4 hours",
    "q.6.h": "every 6 hours",
    "q.8.h": "every 8 hours",
    "q.12.h": "every 12 hours",
    "q.a.m": "every morning",
    "q.p.m": "every evening",
    # ── Timing ──────────────────────────────────────────────────────────
    "a.c": "before meals",
    "p.c": "after meals",
    "h.s": "at bedtime",
    "p.r.n": "as needed",
    "s.o.s": "if needed",
    # ── Route ───────────────────────────────────────────────────────────
    "p.o": "by mouth",
    "s.l": "sublingual",
    "s.q": "subcutaneous",
    "s.c": "subcutaneous",
    "i.v": "intravenous",
    "i.m": "intramuscular",
    "i.d": "intradermal",
    "p.r": "per rectum",
    "p.v": "per vagina",
    "i.t": "intrathecal",
    "i.n": "intranasal",
    # ── Eye ─────────────────────────────────────────────────────────────
    "o.d": "right eye",
    "o.s": "left eye",
    "o.u": "both eyes",
    # ── Other common dotted forms ───────────────────────────────────────
    "n.p.o": "nothing by mouth",
    "d.c": "discontinue",
    "a.d.l": "activities of daily living",
    "w.n.l": "within normal limits",
    "d.t": "delirium tremens",
}


# ── Slash abbreviations (s/p, c/o, w/, w/o, n/v, etc.) ────────────────────
# Very common in clinical notes.  Matched by a dedicated regex pass.
# Key = canonical lowercase form with slash.
# ---------------------------------------------------------------------------
SLASH_ABBREVIATIONS: dict[str, str] = {
    "s/p": "status post",
    "c/o": "complains of",
    "w/": "with",
    "w/o": "without",
    "b/l": "bilateral",
    "n/v": "nausea and vomiting",
    "n/v/d": "nausea, vomiting, and diarrhea",
    "d/c": "discharge",
    "f/u": "follow up",
    "h/o": "history of",
    "r/o": "rule out",
    "d/t": "due to",
    "a/w": "associated with",
    "p/w": "presents with",
    "c/b": "complicated by",
    "c/w": "consistent with",
    "e/o": "evidence of",
    "s/s": "signs and symptoms",
    "l/s": "lumbosacral",
    "c/s": "cesarean section",
    "i/o": "intake and output",
}

# ── Ampersand abbreviations (I&D, D&C, etc.) ──────────────────────────────
AMPERSAND_ABBREVIATIONS: dict[str, str] = {
    "I&D": "incision and drainage",
    "D&C": "dilation and curettage",
    "T&A": "tonsillectomy and adenoidectomy",
    "S&S": "signs and symptoms",
    "C&S": "culture and sensitivity",
    "A&O": "alert and oriented",
    "A&Ox3": "alert and oriented times three",
    "A&Ox4": "alert and oriented times four",
    "D&E": "dilation and evacuation",
    "O&P": "ova and parasites",
    "R&R": "rate and rhythm",
    "S&A": "sugar and acetone",
    "T&C": "type and crossmatch",
    "T&S": "type and screen",
}

# ── Mixed-case / shorthand abbreviations ───────────────────────────────────
# These don't follow the 2+ UPPERCASE pattern: Dx, Tx, SpO2, FiO2, etc.
# Matched case-insensitively by a dedicated pass.
# ---------------------------------------------------------------------------
MIXEDCASE_ABBREVIATIONS: dict[str, str] = {
    # Clinical shorthand (Xx pattern)
    "Dx": "diagnosis",
    "Tx": "treatment",
    "Rx": "prescription",
    "Hx": "history",
    "Sx": "symptoms",
    "Fx": "fracture",
    "Bx": "biopsy",
    "Cx": "culture",
    "Ax": "assessment",
    "Mx": "management",
    "Px": "prognosis",
    "Ix": "investigations",
    # Physiological values with numbers/mixed case
    "SpO2": "oxygen saturation",
    "SaO2": "arterial oxygen saturation",
    "PaO2": "partial pressure of oxygen",
    "PaCO2": "partial pressure of carbon dioxide",
    "pO2": "partial pressure of oxygen",
    "pCO2": "partial pressure of carbon dioxide",
    "FiO2": "fraction of inspired oxygen",
    "EtCO2": "end-tidal carbon dioxide",
    "HbA1c": "hemoglobin A1c",
    "Hb": "hemoglobin",
    "Hgb": "hemoglobin",
    "Plt": "platelet",
    "Cr": "creatinine",
    "BNP": "B-type natriuretic peptide",
    "tPA": "tissue plasminogen activator",
    "pH": "pH",  # Keep as-is, but include so it's recognized
    # Temperature / vitals
    "Tmax": "maximum temperature",
    "O2sat": "oxygen saturation",
}

# ── Degree / severity abbreviations (1°, 2°, 3°) ─────────────────────────
DEGREE_ABBREVIATIONS: dict[str, str] = {
    "1°": "primary",
    "2°": "secondary",
    "3°": "tertiary",
    "4°": "quaternary",
}

# ── Bar-notation (old-school Latin shorthand) ──────────────────────────────
# c̄ = cum (with), s̄ = sine (without), p̄ = post (after), ā = ante (before)
# These use Unicode combining overline (U+0304) or macron characters.
BAR_ABBREVIATIONS: dict[str, str] = {
    "c\u0304": "with",  # c̄
    "s\u0304": "without",  # s̄
    "p\u0304": "after",  # p̄
    "a\u0304": "before",  # ā
    "\u0101": "before",  # ā (precomposed)
}


# ── Dotted / lowercase forms common in clinical notes ───────────────────────
# Maps dotted variants to their canonical uppercase key in ABBREVIATIONS.
# Order matters: longer patterns are checked first to avoid partial matches
# (e.g. "q.i.d." must match before "q.d." could partially match "q." in it).
DOTTED_FORMS: dict[str, str] = {
    # Frequency / dosing
    "q.d.": "QD",
    "qd": "QD",
    "b.i.d.": "BID",
    "bid": "BID",
    "t.i.d.": "TID",
    "tid": "TID",
    "q.i.d.": "QID",
    "qid": "QID",
    "q.h.s.": "QHS",
    "qhs": "QHS",
    "q.o.d.": "QOD",
    "qod": "QOD",
    "p.r.n.": "PRN",
    "prn": "PRN",
    # Route
    "p.o.": "PO",
    "po": "PO",
    "i.v.": "IV",
    "iv": "IV",
    "i.m.": "IM",
    "im": "IM",
    "s.q.": "SQ",
    "sq": "SQ",
    "s.c.": "SQ",
    "sc": "SQ",
    "s.l.": "SL",
    "sl": "SL",
    "p.r.": "PR",
    "pr": "PR",
    # Timing
    "a.c.": "AC",
    "ac": "AC",
    "p.c.": "PC",
    "pc": "PC",
    "h.s.": "QHS",
    "hs": "QHS",
    # Eye
    "o.d.": "OD",
    "od": "OD",
    "o.s.": "OS",
    "os": "OS",
    "o.u.": "OU",
    "ou": "OU",
    # Other common dotted forms
    "d.c.": "DC",
    "dc": "DC",
    "r.x.": "RX",
    "rx": "RX",
    "c.n.s.": "CNS",
    "g.i.": "GI",
    "n.p.o.": "NPO",
    "npo": "NPO",
    "d.v.t.": "DVT",
    "c.h.f.": "CHF",
    "a.b.g.": "ABG",
    "e.k.g.": "EKG",
    "e.c.g.": "ECG",
}


# ── False positives: common English words that are 2+ uppercase letters ─────
FALSE_POSITIVES: set[str] = {
    "AM",
    "PM",
    "OR",
    "US",
    "IT",
    "IS",
    "IF",
    "IN",
    "ON",
    "AN",
    "AT",
    "BE",
    "BY",
    "DO",
    "GO",
    "HE",
    "ME",
    "MY",
    "NO",
    "OF",
    "SO",
    "TO",
    "UP",
    "WE",
    "ID",
    "OK",
    "TV",
    "UK",
    "EU",
    "UN",
    "CEO",
    "CFO",
    "CTO",
    "VP",
    "HR",
    "PR",
    "PC",
    "LLC",
    "INC",
    "LTD",
    "MR",
    "MRS",
    "DR",
    "JR",
    "SR",
    "MD",
    "RN",
    "PA",
    "NP",
    "USA",
    "FBI",
    "CIA",
    "NASA",
    "IRS",
    "PDF",
    "URL",
    "API",
    "SQL",
    "HTML",
    "CSS",
    "FYI",
    "ASAP",
    "FAQ",
    "ETA",
    "TBD",
    "TBA",
    "ALL",  # "acute lymphoblastic leukemia" — too risky as false positive
}
