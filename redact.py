from __future__ import annotations

import re
from pathlib import Path
import fitz  # PyMuPDF


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
    "DATE_TIME",
    "LOCATION",
]

def detect_pii_spans(text: str, entities=DEFAULT_ENTITIES, min_score=0.6):
    results = ANALYZER.analyze(text=text, language="en", entities=entities)
    # results: start, end, entity_type, score ...
    return [r for r in results if r.score >= min_score]


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

    ws = [(w[4].strip(), fitz.Rect(w[0], w[1], w[2], w[3])) for w in line_words]
    lowers = [w.lower() for w, _ in ws]

    target_tokens = re.split(r"\s+", target_lower.strip())
    if not target_tokens:
        return []

    rects = []
    n = len(target_tokens)

    for i in range(0, len(lowers) - n + 1):
        window = lowers[i : i + n]
        if window == target_tokens:
            # Union rectangles across the token span
            rr = ws[i][1]
            for j in range(i + 1, i + n):
                rr |= ws[j][1]
            rects.append(rr)

    return rects


if __name__ == "__main__":
    input_pdf="pretend_redaction_test.pdf"
    output_pdf= "output/"+input_pdf.strip(".pdf")+"_REDACTED.pdf"
    stats = redact_pdf_names(
        input_pdf= input_pdf,
        output_pdf= output_pdf,
        names=["Rogers Pamela", "Rogers", "Pamela", "Walker Moody", "Walker", "Moody", "Texas Tech"], # introduce the abilty for it to detect names itself without me needing to input names
        case_insensitive=True,
    )
    print("Redaction completed and saved to:", stats['output'])
    print("Total Redactions:", stats['total_matches'])
    print("Names found and counts:", stats['names_found'])
    print("Page count:", stats['pages'])

    input_pdf="UMNwriteup.pdf"
    output_pdf= "output/"+input_pdf.strip(".pdf")+"_REDACTED.pdf"
    stats = redact_pdf_names(
        input_pdf= input_pdf,
        output_pdf= output_pdf,
        names=["Rogers Pamela", "Rogers", "Pamela", "Walker Moody", "Walker", "Moody", "Texas Tech"], # introduce the abilty for it to detect names itself without me needing to input names
        case_insensitive=True,
    )
    print("Redaction completed and saved to:", stats['output'])
    print("Total Redactions:", stats['total_matches'])
    print("Names found and counts:", stats['names_found'])
    print("Page count:", stats['pages'])
