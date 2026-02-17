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

def build_presidio():
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }
    )
    nlp_engine = provider.create_engine()
    return AnalyzerEngine(nlp_engine=nlp_engine)

ANALYZER = build_presidio()

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
]

ALLOWLIST_EXACT = {
    # words/phrases you never want redacted
    "inc", "llc", "ltd",
    "university", "department",
    "texas", "california", "lubbock",
    "usa", "united states", "weeks ago", "months ago", "minutes",
    "date", "date", "date:",  "birth", "birth", "email", 
}

DATE_PATTERNS = [
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),              # 6/2/04, 03/14/1982
    re.compile(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?$", re.I),
    re.compile(r"^\d{1,2},?$"),                             # day: 2 or 2,
    re.compile(r"^\d{4}$"),                                 # year: 2024
]

ALLOWLIST_REGEX = [
    r"^\d{4}$",          # allow years like 2024
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
        print("In allow list:", s_norm)
        return True

    for pat in ALLOWLIST_REGEX:
        if re.fullmatch(pat, s, flags=re.IGNORECASE):
            
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

def _norm_token(s: str) -> str:
    # strip leading/trailing punctuation like commas, periods, quotes, etc.
    return re.sub(r"^[^\w]+|[^\w]+$", "", (s or "")).lower()


def detect_pii_spans(text: str, entities=DEFAULT_ENTITIES, min_score=0.85):
    results = ANALYZER.analyze(text=text, language="en", entities=entities)
    # results: start, end, entity_type, score ...
    return [r for r in results if r.score >= min_score]

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
    min_score=0.6,
    max_term_len=80,
) -> list[str]:

    input_pdf = Path(input_pdf)
    doc = fitz.open(str(input_pdf))

    terms: list[str] = []
    for page in doc:
        page_text = page.get_text("text") or ""
        results = detect_pii_spans(page_text, entities=entities, min_score=min_score)

        for r in results:
            raw = page_text[r.start:r.end]

            raw = re.sub(r"\s+", " ", raw).strip()

            if not raw or len(raw) > max_term_len:
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
            rects = page.search_for(name)
            
            if case_insensitive and not rects:
                # Word-level scan; joins adjacent words to match multi-word names.
                words = page.get_text("words")  # (x0,y0,x1,y1,"word", block, line, word_no)
                # Group by line for more reliable adjacency
                words.sort(key=lambda w: (w[5], w[6], w[7]))
                target = name.lower()

                # Build per-line strings with mapping back to word rects
                current_line = None
                line_words = []
                for w in words + [None]:
                    if w is None:
                        # flush
                        if line_words:
                            rects += _find_name_rects_in_line(line_words, target)
                        break

                    line_id = (w[5], w[6])  # (block, line)
                    if current_line is None:
                        current_line = line_id

                    if line_id != current_line:
                        rects += _find_name_rects_in_line(line_words, target)
                        line_words = []
                        current_line = line_id

                    line_words.append(w)

            # Add redaction annotations for each found instance
            for r in rects:
                # Expand a tiny bit so the box fully covers glyphs
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
        print("Contains Rogers term?:", any("rogers" in t.lower() for t in pii_terms))
        print("Rogers-like terms:", [t for t in pii_terms if "rogers" in t.lower()])


        if not pii_terms:

            input_pdf.replace(output_pdf)
            continue

        print(f"Detected {len(pii_terms)} unique PII terms")

        stats = redact_pdf_names(
            input_pdf=input_pdf,
            output_pdf=output_pdf,
            names=pii_terms,
            case_insensitive=True,
        )

        print("Output file:", stats["output"])
        print("Pages:", stats["pages"])
        print("Total redactions:", stats["total_matches"])

        if stats["names_found"]:
            print("PII found:")
            for name, count in stats["names_found"].items():
                print(f"  {name}: {count}")
        else:
            print("No visual matches found (possible layout mismatch)")
