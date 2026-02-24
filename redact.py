from __future__ import annotations

import re
from pathlib import Path
import fitz  # PyMuPDF

'''
25. Personally identifiable information (PII) redaction from text documents (from Ready Tensor)
Developing a system for personally identifiable information (PII) redaction involves implementing
processes and tools to identify and remove sensitive information from documents or data. This is an
important problem in many industries, especially healthcare. Before doing AI/ML tasks on health
records, organizations prefer to remove patient names, emails, phones, etc. The system may use
techniques like pattern matching, regular expressions, or machine learning to identify and redact
sensitive information. Text replacement is one of the ways of PII redaction, which replaces specific PII
elements with generic placeholders. For example, replace names with "Person A" or "John Doe,"
addresses with generic locations, and numbers with placeholders. Blackout or Whiteout is another way
of PII redaction, which physically blocks out or covers PII using black or white rectangles. It is a
straightforward method commonly used in printed documents.
'''

# pip install presidio-analyzer presidio-anonymizer spacy
# python -m spacy download en_core_web_lg

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer import PatternRecognizer, Pattern

############################################################################################

DEFAULT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    #"DATE_TIME",
    "ZIP_CODE",
    "AGE",
    "LOCATION",
    "US_ADDRESS",
    "STRICT_DATE",
    "MEDICAL_RECORD_NUMBER",
]

'''
['US_PASSPORT', 'CREDIT_CARD', 'US_ITIN', 'EMAIL_ADDRESS', 'IBAN_CODE', 
'ORGANIZATION', 'ID', 'PERSON', 'US_SSN', 'AGE', 'DATE_TIME', 'US_BANK_NUMBER', 
'MEDICAL_LICENSE', 'CRYPTO', 'LOCATION', 'URL', 'PHONE_NUMBER', 'UK_NHS', 
'US_DRIVER_LICENSE', 'NRP', 'EMAIL', 'IP_ADDRESS']
'''

def add_strict_date_recognizer(analyzer: AnalyzerEngine) -> None:
    date_patterns = [
        Pattern(
            name="mm_dd_yyyy",
            score=0.95,
            regex=r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        ),
        Pattern(
            name="iso_date",
            score=0.95,
            regex=r"\b\d{4}-\d{2}-\d{2}\b",
        ),
        Pattern(
            name="month_day_year",
            score=0.95,
            regex=r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2},?\s+\d{4}\b",
        ),
        Pattern(
            name="month.year",
            score=0.95,
            regex=r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)\.?\s+\d{4}\b",
        ),
    ]
    recognizer = PatternRecognizer(
        supported_entity="STRICT_DATE",
        patterns=date_patterns,
    )

    analyzer.registry.add_recognizer(recognizer)

def add_mrn_recognizer(analyzer: AnalyzerEngine) -> None:
    mrn_patterns = [
        Pattern(
            name="mrn_with_label",
            score=0.99,
            regex=r"\bMRN[-:\s]?\d{4,10}\b",
        ),

        Pattern(
            name="generic_medical_id",
            score=0.85,
            regex=r"\b\d{6,10}\b",
        ),
    ]

    recognizer = PatternRecognizer(
        supported_entity="MEDICAL_RECORD_NUMBER",
        patterns=mrn_patterns,
        context=["medical record", "mrn", "record number"],
    )

    analyzer.registry.add_recognizer(recognizer)


def add_us_address_recognizer(analyzer: AnalyzerEngine) -> None:

    us_address_patterns = [
        Pattern(
            name="us_address_full",
            score=0.85,
            regex=r"\b\d{1,6}(?:\s+[\w.'-]+){1,10}\s+(?:St|Street|Square|Hall|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Cir|Circle|Way|Pkwy|Parkway|Pl|Broadway|Place|Ter|Terrace|Center)\b(?:\s*,?\s*(?:Apt|Apartment|Unit|Ste|Suite|#)\s*\w+)?(?:\s*,?\s*[A-Za-z .'-]{2,30})?(?:\s*,?\s*(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY))?(?:\s+\d{5}(?:-\d{4})?)?\b",
        ),
        Pattern(
            name="us_address_numbered_street",
            score=0.90,
            regex=r"""
            \b
            \d{1,6}                                           # house number
            (?:\s+(?:N|S|E|W|NE|NW|SE|SW)\.?)?                 # optional direction
            \s+
            (?:\d{1,5}(?:st|nd|rd|th)\b|[\w.'-]+)              # street name: 14th OR normal token
            (?:\s+[\w.'-]+){0,5}                               # remaining street name tokens
            \s+
            (?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Cir|Circle|Way|Pkwy|Parkway|Pl|Place|Ter|Terrace|Broadway|Center|Sq|Square)\.?
            (?:\s*,?\s*(?:Apt|Apartment|Unit|Ste|Suite|#)\s*[\w-]+)?     # optional unit
            (?:\s*,?\s*[A-Za-z .'-]{2,30})?                    # optional city
            (?:\s*,?\s*(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY))?  # optional state
            (?:\s+\d{5}(?:-\d{4})?)?                           # optional ZIP
            \b
            """.strip(),
        ),
        Pattern(
            name="us_street_general",
            score=0.90,
            regex=r"""
            \b
            (?:
                \d{1,6}(?:\s+(?:N|S|E|W|NE|NW|SE|SW)\.?)?   # house number + optional direction
                |
                (?:North|South|East|West|N|S|E|W)\.?        # OR leading direction only
            )
            \s+
            (?:\d{1,5}\s*(?:st|nd|rd|th)\b|[\w.'-]+)        # street name
            (?:\s+[\w.'-]+){0,4}
            \s+
            (?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Cir|Circle|Way|Pkwy|Parkway|Pl|Place|Ter|Terrace|Broadway|Center|Sq|Square)\.?
            \b
            """.strip(),
        ),
        Pattern(
            name="numbered_street_optional_suffix",
            score=0.95,
            regex=r"""
            \b
            \d{1,6}                                   # house number
            (?:\s+(?:N|S|E|W|NE|NW|SE|SW)\.?)?         # optional direction
            \s+
            (?:\d{1,5}(?:st|nd|rd|th)\b|[\w.'-]+)      # street name (14th OR word)
            (?:\s+[\w.'-]+){0,3}                       # optional additional words
            (?:                                         
                \s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Cir|Circle|Way|Pkwy|Parkway|Pl|Place|Ter|Terrace|Sq|Square)\.?
            )?                                          # suffix OPTIONAL
            \b
            """.strip(),
        ),
    ]
    recognizer = PatternRecognizer(
        supported_entity="US_ADDRESS",
        patterns=us_address_patterns,
        context=["address","st", "broadway", "street", "ave", "road", "blvd", "apt", "suite", "zip"],
    )
    analyzer.registry.add_recognizer(recognizer)


def build_presidio():
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }
    )
    nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

    add_us_address_recognizer(analyzer)
    add_strict_date_recognizer(analyzer)
    add_mrn_recognizer(analyzer)

    return analyzer

ANALYZER = build_presidio()

############################################################################################
#allow list

ALLOWLIST_EXACT = {     # words/phrases you never want redacted
    "inc", "llc", "ltd",
    "university", "department",
    "texas", "california", "lubbock",
    "usa", "united states", "weeks ago", "months ago", "minutes",
    "date", "date", "date:",  "birth", "birth", "email", "biographies",
    "dyspnea", "epigastric", "lumbosacral", "lumbo", "musculoskeletal", "the", "is", "in", "ist",
}

DATE_PATTERNS = [
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),              # 6/2/04, 03/14/1982
    re.compile(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?$", re.I),
    re.compile(r"^\d{1,2},?$"),                             # day: 2 or 2,
    re.compile(r"^\d{4}$"),                                 # year: 2024
]

def should_keep_term(term: str) -> bool:
    t = (term or "").strip()
    if not t:
        return False
    # Don't redact 1-letter tokens (A, B, C...)
    if len(t) == 1 and t.isalpha():
        return False
    return True

def is_allowlisted(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    s_norm = re.sub(r"\s+", " ", s).lower()

    if s_norm in {x.lower() for x in ALLOWLIST_EXACT}:
        return True
    return False

def split_and_filter_allowlisted(snippet: str) -> list[str]:
    s = re.sub(r"\s+", " ", (snippet or "")).strip()
    if not s:
        return []

    parts = s.split()
    kept = []
    for p in parts:
        p_clean = re.sub(r"^[^\w]+|[^\w]+$", "", p).strip()
        if not p_clean:
            continue
        if is_allowlisted(p_clean):
            continue
        kept.append(p_clean)

    # If we removed something (like Date) but still have meaningful parts, use those
    if kept and (len(kept) != len(parts)):
        return kept

    return [s]

############################################################################################


_WORD_CH = re.compile(r"[A-Za-z0-9_]")

def is_whole_word_span(text: str, start: int, end: int) -> bool:
    
    if start < 0 or end > len(text) or start >= end:
        return False
    
    left_ok = (start == 0) or (not _WORD_CH.match(text[start - 1]))
    right_ok = (end == len(text)) or (not _WORD_CH.match(text[end]))
    return left_ok and right_ok

def _norm_token(s: str) -> str:
    # strip leading/trailing punctuation like commas, periods, quotes, etc.
    return re.sub(r"^[^\w]+|[^\w]+$", "", (s or "")).lower()

ENTITY_MIN_SCORE = {
    "PERSON": 0.80,
    "STRICT_DATE": 0.60,
    "US_ADDRESS": 0.70,
    "MEDICAL_RECORD_NUMBER": 0.85,
}

def detect_pii_spans(text: str, entities=DEFAULT_ENTITIES, default_min=0.85):
    results = ANALYZER.analyze(text=text, language="en")
    wanted = set(entities)

    kept = []
    for r in results:
        thresh = ENTITY_MIN_SCORE.get(r.entity_type, default_min)
        if r.entity_type in wanted and r.score >= thresh:
            kept.append(r)
    return kept
def print_pii(results, text):
    if not results:
        print("No PII detected.")
        return

    for r in results:
        snippet = text[r.start:r.end]
        print(
            f"[{r.entity_type}] '{snippet}' "
            f"(score={r.score:.2f}, span={r.start}-{r.end})"
        )

def redact_text(text: str, results, mask="[REDACTED]"):
    # sort spans backwards so offsets don't shift
    for r in sorted(results, key=lambda x: x.start, reverse=True):
        text = text[:r.start] + mask + text[r.end:]
    return text

def extract_pii_terms_from_pdf(
    input_pdf: str | Path,
    *,
    entities=DEFAULT_ENTITIES,
    min_score=0.7,
    max_term_len=80,
) -> list[str]:

    input_pdf = Path(input_pdf)
    doc = fitz.open(str(input_pdf))

    terms: list[str] = []
    for page in doc:
        page_text = page.get_text("text") or ""
        results = detect_pii_spans(page_text, entities=entities, default_min=min_score)
       
        for r in results:
            
            raw = page_text[r.start:r.end]

            if not is_whole_word_span(page_text, r.start, r.end):
                continue    

            raw = re.sub(r"\s+", " ", raw).strip()

            if not raw or len(raw) > max_term_len:
                continue

            if r.entity_type == "STRICT_DATE":
                if not is_allowlisted(raw):
                    terms.append(raw)
                continue
        
            if r.entity_type == "US_ADDRESS":
                addr_tokens = [_norm_token(t) for t in raw.split() if _norm_token(t)]
                if addr_tokens and not is_allowlisted(raw):
                    terms.append(" ".join(addr_tokens))
                continue

            kept = []
            for token in raw.split():
                clean = re.sub(r"^[^\w]+|[^\w]+$", "", token).strip()
                if not clean:
                    continue
                if is_allowlisted(clean):
                    continue
                if not should_keep_term(clean):
                    continue
                kept.append(clean)
         
            terms.extend(kept)


            if r.entity_type == "PERSON" and len(kept) >= 2:
                # join first two tokens as a "full name" candidate
                terms.append(f"{kept[0]} {kept[1]}")


    doc.close()
    return _normalize_names(terms)


def _normalize_names(names: list[str]) -> list[str]:
    seen = set()
    out = []
    for n in names:
        n = (n or "").strip()
        if not n:
            continue
        key = n.lower()
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out

############################################################################################

def _find_exact_token_rects(page, target: str):
    t = _norm_token(target)
    if not t:
        return []

    rects = []
    for w in page.get_text("words"):
        token = _norm_token(w[4])
        if token == t:
            rects.append(fitz.Rect(w[0], w[1], w[2], w[3]))
    return rects


def redact_pdf_names(
    input_pdf: str | Path,
    output_pdf: str | Path,
    names: list[str],
    *,
    case_insensitive: bool = True, #still redacts lowercase names
    redact_color_rgb: tuple[int, int, int] = (0, 0, 0),
    ) -> dict:

    input_pdf = Path(input_pdf)
    output_pdf = Path(output_pdf)

    if not input_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    names = _normalize_names(names)
    names = [n for n in names if not is_allowlisted(n)]
    if not names:
        raise ValueError("No names provided to redact.")

    doc = fitz.open(str(input_pdf))

    total_matches = 0
    names_found: dict[str, int] = {n: 0 for n in names}

    # PyMuPDF expects fill colors in 0..1 floats
    fill = tuple(c / 255 for c in redact_color_rgb)

    for page in doc:
        for name in names:
            # Fast path: exact search
            name = name.strip()
            if not name:
                continue

            if " " in name:
                rects = []
                words = page.get_text("words")
                words.sort(key=lambda w: (w[5], w[6], w[7]))
                target = name.lower()

                current_line = None
                line_words = []
                for w in words + [None]:
                    if w is None:
                        if line_words:
                            rects += _find_name_rects_in_line(line_words, target)
                        break

                    line_id = (w[5], w[6])
                    if current_line is None:
                        current_line = line_id

                    if line_id != current_line:
                        rects += _find_name_rects_in_line(line_words, target)
                        line_words = []
                        current_line = line_id

                    line_words.append(w)

            else:
                rects = _find_exact_token_rects(page, name)

            for r in rects:
                # Expand a tiny bit so the box fully covers text
                rr = fitz.Rect(r)
                rr.x0 -= 0.5
                rr.y0 -= 0.5
                rr.x1 += 0.5
                rr.y1 += 0.5

                page.add_redact_annot(rr, fill=fill)
                total_matches += 1
                names_found[name] += 1

    # Apply redactions (this is the key that removes underlying text)

    for page in doc:
        page.apply_redactions()

    pages_count = doc.page_count

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_pdf))
    doc.close()

    return {
        "pages": pages_count,
        "total_matches": total_matches,
        "names_found": {k: v for k, v in names_found.items() if v > 0},
        "output": str(output_pdf),
    }



def _find_name_rects_in_line(line_words, target_lower: str):
    """
    line_words: list of word tuples from page.get_text("words")
    Returns list of fitz.Rect covering matched sequences.
    """

    tokens_and_rects = []
    cleaned = []

    for w in line_words:
        raw = (w[4] or "").strip()
        rect = fitz.Rect(w[0], w[1], w[2], w[3])
        tokens_and_rects.append((raw, rect))
        cleaned.append(_norm_token(raw))

    target_tokens = [t for t in (_norm_token(x) for x in re.split(r"\s+", target_lower.strip())) if t]
    if not target_tokens:
        return []

    rects = []
    n = len(target_tokens)

    for i in range(0, len(cleaned) - n + 1):
        window = cleaned[i : i + n]
        if window == target_tokens:
            rr = tokens_and_rects[i][1]
            for j in range(i + 1, i + n):
                rr |= tokens_and_rects[j][1]
            rects.append(rr)

    return rects

############################################################################################

if __name__ == "__main__":
    INPUT_DIR = Path("input")
    OUTPUT_DIR = Path("output")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(INPUT_DIR.glob("*.pdf"))

    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {INPUT_DIR.resolve()}")

    for input_pdf in pdfs:
        print(f"Processing: {input_pdf.name}")

        output_pdf = OUTPUT_DIR / f"{input_pdf.stem}_REDACTED.pdf"
        pii_terms = extract_pii_terms_from_pdf(
            input_pdf,
            entities=DEFAULT_ENTITIES,
            min_score=0.6,
        )

        if not pii_terms:
            input_pdf.replace(output_pdf)
            continue
        
        stats = redact_pdf_names(
            input_pdf=input_pdf,
            output_pdf=output_pdf,
            names=pii_terms,
            case_insensitive=True,
        )

        print("Output file:", stats["output"])
        print("Total redactions:", stats["total_matches"])

        if stats["names_found"]:
            print("PII found:")
            for name, count in stats["names_found"].items():
                print(f"  {name}: {count}")
        else:
            print("No visual matches found (possible layout mismatch)")
    
