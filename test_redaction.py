"""
Redaction test suite.

Creates a synthetic medical PDF, runs the full redaction pipeline, then
verifies that PII is gone and non-PII is preserved.

Run:
    conda run -n redact pytest test_redaction.py -v
"""
from __future__ import annotations

import re
from pathlib import Path

import fitz
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pdf(lines: list[str], path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 60
    for line in lines:
        if y > 740:
            page = doc.new_page(width=612, height=792)
            y = 60
        page.insert_text((72, y), line, fontsize=11)
        y += 18
    doc.save(str(path))
    doc.close()


def _pdf_text(path: Path) -> str:
    doc = fitz.open(str(path))
    text = "\n".join(page.get_text("text") for page in doc)
    doc.close()
    return text


# ---------------------------------------------------------------------------
# Pipeline fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pipeline():
    from redact import extract_pii_terms_from_pdf, redact_pdf_names, DEFAULT_ENTITIES
    return extract_pii_terms_from_pdf, redact_pdf_names, DEFAULT_ENTITIES


# ---------------------------------------------------------------------------
# Test data — (id, pdf_line, term_to_check)
# ---------------------------------------------------------------------------

SHOULD_REDACT = [
    # ── Names ────────────────────────────────────────────────────────────────
    ("patient name",              "Patient: Jonathan Harker",                          "Jonathan Harker"),
    ("patient name spaced",       "Patient Name: Emily Clarke",                        "Emily Clarke"),
    ("dr prefix",                 "Attending: Dr. Marcus Webb",                        "Marcus Webb"),
    ("name with middle initial",  "Patient: Sarah J. Thompson",                        "Sarah"),
    ("two-word name inline",      "Referred by James Morrison MD",                     "James Morrison"),
    ("name after colon no space", "Provider:Rachel Green",                             "Rachel Green"),
    ("continuously missed name",              "April Murphy",                                     "April Murphy"),
    ("april in table",             "Name Date Changes April Murphy Aug. 2019 Extensive",        "April Murphy"),


    # ── Emergency / next of kin ───────────────────────────────────────────────
    ("emergency contact nospace", "Emergency Contact:Mina Murray",                     "Mina Murray"),
    ("emergency contact space",   "Emergency Contact: Arthur Holmwood",                "Arthur Holmwood"),
    ("next of kin",               "Next of Kin: Lucy Westenra",                        "Lucy Westenra"),
    ("guarantor",                 "Guarantor: Renfield Smith",                         "Renfield Smith"),

    # ── SSN ──────────────────────────────────────────────────────────────────
    ("ssn dashes",                "SSN: 123-45-6789",                                  "123-45-6789"),
    ("ssn spaces",                "Social Security: 234 56 7890",                      "234 56 7890"),

    # ── Phone ────────────────────────────────────────────────────────────────
    ("phone parens",              "Phone: (806) 555-0192",                             "555-0192"),
    ("phone dots",                "Cell: 806.555.0193",                                "806.555.0193"),
    ("fax number",                "Fax: (806) 555-0195",                               "555-0195"),

    # ── Email ────────────────────────────────────────────────────────────────
    ("email basic",               "Email: jharker@example.com",                        "jharker@example.com"),
    ("email subdomain",           "Contact: patient.doe@mail.hospital.org",            "patient.doe@mail.hospital.org"),

    # ── Date of birth ────────────────────────────────────────────────────────
    ("dob slash",                 "DOB: 03/15/1958",                                   "03/15/1958"),
    ("dob labeled written",       "Date of Birth: January 5, 1962",                   "January 5, 1962"),
    ("dob iso",                   "Birthdate: 1975-08-22",                             "1975-08-22"),

    # ── Address ──────────────────────────────────────────────────────────────
    ("street address",            "Address: 452 Elm Street, Lubbock TX 79401",         "452 Elm Street"),
    ("avenue address",            "Home: 18 Oak Avenue, Dallas TX 75201",              "18 Oak Avenue"),
    ("apt address",               "Mailing: 301 Maple Dr Apt 4B, Austin TX",           "301 Maple Dr"),

    # ── MRN ──────────────────────────────────────────────────────────────────
    ("mrn labeled",               "MRN: 8834291",                                      "8834291"),
    ("mrn prefixed",              "Record: MRN-2291847",                               "2291847"),

    # ── Insurance ────────────────────────────────────────────────────────────
    ("medicare id",               "Medicare # 1EG4-TE5-MK72",                          "1EG4-TE5-MK72"),
    ("insurance member id",       "Member ID: XYZ987654",                              "XYZ987654"),
]


SHOULD_NOT_REDACT = [
    # ── Medications ──────────────────────────────────────────────────────────
    ("tamsulosin",          "Medications: Tamsulosin 0.4 mg daily",       "Tamsulosin"),
    ("metformin",           "Medications: Metformin 500 mg BID",          "Metformin"),
    ("lisinopril",          "Medications: Lisinopril 10 mg daily",        "Lisinopril"),
    ("atorvastatin",        "Medications: Atorvastatin 40 mg nightly",    "Atorvastatin"),
    ("albuterol",           "Medications: Albuterol inhaler PRN",         "Albuterol"),
    ("omeprazole",          "Medications: Omeprazole 20 mg daily",        "Omeprazole"),
    ("gabapentin",          "Medications: Gabapentin 300 mg TID",         "Gabapentin"),
    ("sertraline",          "Medications: Sertraline 50 mg daily",        "Sertraline"),
    ("furosemide",          "Medications: Furosemide 40 mg daily",        "Furosemide"),

    # ── Medical terms / symptoms ─────────────────────────────────────────────
    ("dyspnea",             "Chief Complaint: dyspnea on exertion",       "dyspnea"),
    ("epigastric",          "Pain: epigastric tenderness noted",          "epigastric"),
    ("musculoskeletal",     "MSK: musculoskeletal exam normal",           "musculoskeletal"),


    # ── Section headers ──────────────────────────────────────────────────────
    ("physical exam",       "Physical Exam:",                             "Physical Exam"),
    ("history header",      "History of Present Illness:",                "History"),
    ("assessment header",   "Assessment and Plan:",                       "Assessment"),
    ("allergies header",    "Allergies: Penicillin",                      "Allergies"),
    ("medications hdr",     "Current Medications:",                       "Medications"),
    ("vital signs",         "Vital Signs: BP 120/80",                     "Vital Signs"),
    ("review systems",      "Review of Systems:",                         "Review"),
    ("social history",      "Social History: non-smoker",                 "Social History"),
    ("family history",      "Family History: no known cardiac disease",   "Family History"),
    ("discharge summary",   "Discharge Summary",                          "Discharge"),
    ("laboratory",          "Laboratory Results:",                        "Laboratory"),
    ("radiology",           "Radiology: chest X-ray normal",              "Radiology"),
    ("diagnosis header",    "Diagnosis: Type 2 Diabetes Mellitus",        "Diagnosis"),

    # ── Facility / org names ─────────────────────────────────────────────────
    ("hospital name",       "Facility: University Medical Center",        "Medical Center"),
    ("clinic name",         "Referred to: University Health Clinic",      "University"),

    # ── Labels that precede PII (the label itself must survive) ──────────────
    ("label patient",       "Patient: Jonathan Harker",                   "Patient"),
    ("label ssn",           "SSN: 123-45-6789",                           "SSN"),
    ("label email",         "Email: jharker@example.com",                 "Email"),
    ("label dob",           "DOB: 03/15/1958",                            "DOB"),
    ("label mrn",           "MRN: 8834291",                               "MRN"),
    ("label phone",         "Phone: (806) 555-0192",                      "Phone"),
    ("label address",       "Address: 452 Elm Street, Lubbock TX 79401",  "Address"),
    ("label emerg contact", "Emergency Contact:Mina Murray",              "Emergency Contact"),

    # ── Non-DOB dates ────────────────────────────────────────────────────────
    ("procedure date lbl",  "Procedure Date: 03/15/2023",                 "Procedure Date"),
    ("visit date lbl",      "Visit Date: 04/10/2024",                     "Visit Date"),
    ("April",               "Follow-up: April 2024",                     "April 2024"),
    (   "month year",          "Birth •",                                   "Birth"),
    

    # ── Month words in normal prose (must NOT be redacted) ───────────────────
    ("may be",             "This may be a reaction to the medication",  "may be"),
    ("may include",        "Symptoms may include nausea and fatigue",   "may include"),
    ("june visit",         "Scheduled for June follow-up",              "June"),
    ("april date",         "Initial visit April 2019",                  "April 2019"),
    ("august note",        "Reviewed in August per protocol",           "August"),

    # ── Clinical values that must not be flagged ──────────────────────────────
    ("bp reading",          "Vital Signs: BP 120/80",                     "120/80"),
    ("o2 sat",              "SpO2: 98%",                                  "SpO2"),
    ("dosage mg",           "Dose: 500 mg",                               "500"),
    ("room number",         "Room: 214B",                                 "214B"),
    ("icd code",            "ICD-10: E11.9",                              "E11.9"),
    ("height weight",       "Ht: 5 ft 10 in  Wt: 185 lbs",              "185"),
    ("temp reading",        "Temp: 98.6 F",                               "98.6"),
    ("heart rate",          "HR: 72 bpm",                                 "72"),
]


# ---------------------------------------------------------------------------
# Build one shared redacted PDF for all checks
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def redacted_text(pipeline, tmp_path_factory):
    extract_pii_terms_from_pdf, redact_pdf_names, DEFAULT_ENTITIES = pipeline

    tmp = tmp_path_factory.mktemp("redact_test")
    input_pdf = tmp / "input.pdf"
    output_pdf = tmp / "output.pdf"

    all_lines = list(dict.fromkeys(line for _, line, _ in SHOULD_REDACT + SHOULD_NOT_REDACT))
    _make_pdf(all_lines, input_pdf)

    terms = extract_pii_terms_from_pdf(input_pdf, entities=DEFAULT_ENTITIES, min_score=0.6)
    if terms:
        redact_pdf_names(input_pdf=input_pdf, output_pdf=output_pdf, names=terms)
    else:
        output_pdf.write_bytes(input_pdf.read_bytes())

    return _pdf_text(output_pdf)


# ---------------------------------------------------------------------------
# Parametrised tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,line,term", SHOULD_REDACT, ids=[x[0] for x in SHOULD_REDACT])
def test_pii_is_redacted(redacted_text, label, line, term):
    """PII terms must not appear verbatim in the redacted output."""
    assert term not in redacted_text, (
        f"[{label}] '{term}' should be redacted but still appears in output."
    )


@pytest.mark.parametrize("label,line,term", SHOULD_NOT_REDACT, ids=[x[0] for x in SHOULD_NOT_REDACT])
def test_non_pii_is_preserved(redacted_text, label, line, term):
    """Non-PII terms must survive redaction unchanged."""
    assert term in redacted_text, (
        f"[{label}] '{term}' should be preserved but was redacted."
    )


# ---------------------------------------------------------------------------
# Regex / structural checks
# ---------------------------------------------------------------------------


def test_ssn_pattern_gone(redacted_text):
    assert not re.search(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b", redacted_text), \
        "An SSN pattern is still present in output."


def test_email_pattern_gone(redacted_text):
    assert not re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", redacted_text), \
        "An email address is still present in output."


def test_credit_card_pattern_gone(redacted_text):
    assert not re.search(r"\b4\d{3}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}\b", redacted_text), \
        "A Visa card number is still present in output."


def test_known_patient_names_gone(redacted_text):
    for name in ("Jonathan Harker", "Emily Clarke", "Mina Murray",
                 "Arthur Holmwood", "Lucy Westenra", "Renfield Smith",
                 "Rachel Green", "Marcus Webb", "James Morrison"):
        assert name not in redacted_text, f"Patient name '{name}' still present in output."


def test_mrns_gone(redacted_text):
    assert "8834291" not in redacted_text
    assert "2291847" not in redacted_text


def test_dob_dates_gone(redacted_text):
    assert "03/15/1958" not in redacted_text
    assert "1975-08-22" not in redacted_text


def test_medications_all_present(redacted_text):
    for med in ("Tamsulosin", "Metformin", "Lisinopril", "Atorvastatin",
                "Albuterol", "Omeprazole", "Gabapentin", "Sertraline", "Furosemide"):
        assert med in redacted_text, f"Medication '{med}' was incorrectly redacted."


def test_section_headers_all_present(redacted_text):
    for h in ("Physical Exam", "Assessment", "Allergies", "Vital Signs",
              "Laboratory", "Discharge", "Social History", "Family History",
              "Radiology", "Diagnosis"):
        assert h in redacted_text, f"Section header '{h}' was incorrectly redacted."


def test_pii_labels_all_present(redacted_text):
    for lbl in ("Patient", "SSN", "Email", "DOB", "MRN",
                "Phone", "Address", "Emergency Contact"):
        assert lbl in redacted_text, f"PII label '{lbl}' was incorrectly redacted."


def test_clinical_values_preserved(redacted_text):
    for val in ("120/80", "SpO2", "98.6", "214B"):
        assert val in redacted_text, f"Clinical value '{val}' was incorrectly redacted."
