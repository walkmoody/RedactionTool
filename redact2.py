from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider


# =============================================================================
# CONFIG
# =============================================================================
MODEL_DIR = "xlmr_pii_ner"
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output_better")

# Safer defaults: LOCATION and AGE create a lot of false positives in PDFs.
DEFAULT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "ZIP_CODE",
    "US_ADDRESS",
    "STRICT_DATE",
    "MEDICAL_RECORD_NUMBER",
]

TRANSFORMER_MAX_LENGTH = 512
TRANSFORMER_STRIDE = 128
TRANSFORMER_SCORE_THRESHOLD = 0.85

ENTITY_MIN_SCORE = {
    "PERSON": 0.86,
    "STRICT_DATE": 0.85,
    "US_ADDRESS": 0.88,
    "MEDICAL_RECORD_NUMBER": 0.92,
    "ZIP_CODE": 0.90,
}

# Higher priority wins on conflicting overlaps.
ENTITY_PRIORITY = {
    "US_SSN": 100,
    "CREDIT_CARD": 95,
    "MEDICAL_RECORD_NUMBER": 92,
    "EMAIL_ADDRESS": 90,
    "PHONE_NUMBER": 88,
    "US_ADDRESS": 85,
    "ZIP_CODE": 84,
    "STRICT_DATE": 83,
    "PERSON": 75,
    "LOCATION": 50,
    "AGE": 40,
}

MODEL_TO_OLD_ENTITY = {
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
    "ADDRESS": "US_ADDRESS",
    "ZIP_CODE": "ZIP_CODE",
    "CREDIT_CARD": "CREDIT_CARD",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "DATE_TIME": "STRICT_DATE",
}

NAME_TITLES = {"dr", "mr", "mrs", "ms", "miss", "prof", "doctor", "professor"}
PUBLIC_ORG_KEYWORDS = {
    "university",
    "college",
    "school",
    "department",
    "library",
    "center",
    "hall",
    "station",
    "building",
    "office",
    "hospital",
    "faculty",
    "records office",
    "medical center",
}

ALLOWLIST_EXACT = {
    "inc",
    "llc",
    "ltd",
    "usa",
    "united states",
    "the",
    "is",
    "in",
    "date",
    "date:",
    "birth",
    "email",
}
DEBUG = False
REDACT_COLOR_RGB = (0, 0, 0)

_WORD_CH = re.compile(r"[A-Za-z0-9_]")
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class SimpleResult:
    start: int
    end: int
    entity_type: str
    score: float
    source: str = "unknown"


@dataclass
class WordBox:
    text: str
    rect: fitz.Rect
    start: int
    end: int
    block: int
    line: int
    word: int


# =============================================================================
# PRESIDIO SETUP
# =============================================================================
def add_strict_date_recognizer(analyzer: AnalyzerEngine) -> None:
    date_patterns = [
        Pattern(
            name="mm_dd_yyyy",
            regex=r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
            score=0.99,
        ),
        Pattern(
            name="iso_date",
            regex=r"\b\d{4}-\d{2}-\d{2}\b",
            score=0.99,
        ),
        Pattern(
            name="month_day_year",
            regex=r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2},?\s+\d{4}\b",
            score=0.99,
        ),
        Pattern(
            name="month_year",
            regex=r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)\.?\s+\d{4}\b",
            score=0.92,
        ),
    ]
    analyzer.registry.add_recognizer(
        PatternRecognizer(supported_entity="STRICT_DATE", patterns=date_patterns)
    )


def add_mrn_recognizer(analyzer: AnalyzerEngine) -> None:
    mrn_patterns = [
        Pattern(
            name="mrn_labeled",
            regex=r"\b(?:MRN|Medical Record Number)\s*[-:#]?\s*[A-Z]*\d{4,12}\b",
            score=0.99,
        ),
        Pattern(
            name="record_number_labeled",
            regex=r"\b(?:Record Number|Chart Number)\s*[:#-]?\s*[A-Z]*\d{4,12}\b",
            score=0.95,
        ),
    ]
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="MEDICAL_RECORD_NUMBER",
            patterns=mrn_patterns,
            context=["mrn", "medical record", "record number", "chart number"],
        )
    )


def add_us_address_recognizer(analyzer: AnalyzerEngine) -> None:
    address_patterns = [
        Pattern(
            name="full_us_address",
            regex=r"\b\d{1,6}(?:\s+(?:N|S|E|W|NE|NW|SE|SW)\.?)?\s+(?:[\w.'-]+\s+){0,6}(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Cir|Circle|Way|Pkwy|Parkway|Pl|Place|Ter|Terrace|Sq|Square|Broadway)\b(?:\s*,?\s*[A-Za-z .'-]{2,30})?(?:\s*,?\s*(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY))?(?:\s+\d{5}(?:-\d{4})?)?\b",
            score=0.95,
        ),
    ]
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="US_ADDRESS",
            patterns=address_patterns,
            context=["address", "home address", "resides at", "lives at", "mailing address"],
        )
    )


def build_presidio() -> AnalyzerEngine:
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }
    )
    nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    add_strict_date_recognizer(analyzer)
    add_mrn_recognizer(analyzer)
    add_us_address_recognizer(analyzer)
    return analyzer


ANALYZER = build_presidio()


# =============================================================================
# TRANSFORMER SETUP
# =============================================================================
try:
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    MODEL = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    MODEL.to(_DEVICE)
    MODEL.eval()
    TRANSFORMER_READY = True
    print(f"Loaded transformer model from: {MODEL_DIR}")
    print(f"Transformer device: {_DEVICE}")
except Exception as exc:
    TOKENIZER = None
    MODEL = None
    TRANSFORMER_READY = False
    print(f"Transformer model not loaded: {exc}")
    print("Continuing in Presidio-only mode.")


# =============================================================================
# SMALL HELPERS
# =============================================================================
def is_allowlisted(text: str) -> bool:
    norm = re.sub(r"\s+", " ", (text or "").strip()).lower()
    return bool(norm) and norm in {x.lower() for x in ALLOWLIST_EXACT}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def is_whole_word_span(text: str, start: int, end: int) -> bool:
    if start < 0 or end > len(text) or start >= end:
        return False
    left_ok = (start == 0) or (not _WORD_CH.match(text[start - 1]))
    right_ok = (end == len(text)) or (not _WORD_CH.match(text[end]))
    return left_ok and right_ok


def trim_span_whitespace(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def looks_like_public_org(text: str) -> bool:
    s = normalize_space(text).lower()
    if not s:
        return False
    return any(keyword in s for keyword in PUBLIC_ORG_KEYWORDS)


def looks_like_strict_address(text: str) -> bool:
    s = normalize_space(text)
    return bool(
        re.search(
            r"\b\d{1,6}(?:\s+(?:N|S|E|W|NE|NW|SE|SW)\.?)?\s+(?:[\w.'-]+\s+){0,6}(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Cir|Circle|Way|Pkwy|Parkway|Pl|Place|Ter|Terrace|Sq|Square|Broadway)\b",
            s,
            flags=re.I,
        )
    )


def debug_print_results(results: Iterable[SimpleResult], text: str, title: str) -> None:
    if not DEBUG:
        return
    print(f"\n--- {title} ---")
    for r in results:
        print(f"[{r.entity_type}] '{text[r.start:r.end]}' score={r.score:.3f} source={r.source} span={r.start}-{r.end}")


# =============================================================================
# PAGE TEXT / WORD MAP
# =============================================================================
def build_page_text_map(page: fitz.Page) -> tuple[str, list[WordBox]]:
    words = page.get_text("words")
    words.sort(key=lambda w: (w[5], w[6], w[7]))

    pieces: list[str] = []
    mapping: list[WordBox] = []
    char_pos = 0
    prev_line = None

    for w in words:
        raw = (w[4] or "").strip()
        if not raw:
            continue

        line_id = (w[5], w[6])
        sep = ""
        if prev_line is None:
            sep = ""
        elif line_id != prev_line:
            sep = "\n"
        else:
            sep = " "

        if sep:
            pieces.append(sep)
            char_pos += len(sep)

        start = char_pos
        pieces.append(raw)
        char_pos += len(raw)
        end = char_pos

        mapping.append(
            WordBox(
                text=raw,
                rect=fitz.Rect(w[0], w[1], w[2], w[3]),
                start=start,
                end=end,
                block=w[5],
                line=w[6],
                word=w[7],
            )
        )
        prev_line = line_id

    return "".join(pieces), mapping


# =============================================================================
# DETECTORS
# =============================================================================
def detect_pii_spans_presidio(text: str, entities: list[str], default_min: float = 0.85) -> list[SimpleResult]:
    raw_results = ANALYZER.analyze(text=text, language="en")
    wanted = set(entities)
    kept: list[SimpleResult] = []

    for r in raw_results:
        if r.entity_type not in wanted:
            continue
        threshold = ENTITY_MIN_SCORE.get(r.entity_type, default_min)
        if r.score < threshold:
            continue
        kept.append(SimpleResult(r.start, r.end, r.entity_type, r.score, source="presidio"))

    return kept


def _decode_transformer_chunk(offsets: list[tuple[int, int]], labels: list[int], scores: list[float], id2label: dict[int, str]) -> list[SimpleResult]:
    spans: list[SimpleResult] = []
    current: SimpleResult | None = None

    for pred_id, score, (start, end) in zip(labels, scores, offsets):
        if start == end:
            continue

        label = id2label[int(pred_id)]
        if label == "O" or score < TRANSFORMER_SCORE_THRESHOLD:
            if current is not None:
                spans.append(current)
                current = None
            continue

        if "-" not in label:
            if current is not None:
                spans.append(current)
                current = None
            continue

        prefix, entity_type = label.split("-", 1)
        mapped = MODEL_TO_OLD_ENTITY.get(entity_type)
        if not mapped:
            if current is not None:
                spans.append(current)
                current = None
            continue

        if prefix == "B" or current is None or current.entity_type != mapped:
            if current is not None:
                spans.append(current)
            current = SimpleResult(start, end, mapped, float(score), source="transformer")
        else:
            current.end = end
            current.score = max(current.score, float(score))

    if current is not None:
        spans.append(current)

    return spans


def detect_pii_spans_transformer(text: str) -> list[SimpleResult]:
    if not TRANSFORMER_READY:
        return []

    encoded = TOKENIZER(
        text,
        truncation=True,
        max_length=TRANSFORMER_MAX_LENGTH,
        stride=TRANSFORMER_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
        return_tensors="pt",
    )

    offset_mappings = encoded.pop("offset_mapping")
    if "overflow_to_sample_mapping" in encoded:
        encoded.pop("overflow_to_sample_mapping")

    encoded = {k: v.to(_DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = MODEL(**encoded)

    probs = torch.softmax(outputs.logits, dim=-1)
    pred_ids = torch.argmax(probs, dim=-1).cpu().tolist()
    pred_scores = torch.max(probs, dim=-1).values.cpu().tolist()
    id2label = MODEL.config.id2label

    all_spans: list[SimpleResult] = []
    for chunk_idx in range(len(offset_mappings)):
        offsets = [tuple(map(int, pair)) for pair in offset_mappings[chunk_idx].tolist()]
        chunk_spans = _decode_transformer_chunk(offsets, pred_ids[chunk_idx], pred_scores[chunk_idx], id2label)
        all_spans.extend(chunk_spans)

    return all_spans


# =============================================================================
# POSTPROCESSING
# =============================================================================
def merge_same_type_overlaps(results: list[SimpleResult], text: str) -> list[SimpleResult]:
    if not results:
        return []

    results = sorted(results, key=lambda r: (r.entity_type, r.start, r.end))
    merged: list[SimpleResult] = [results[0]]

    for cur in results[1:]:
        prev = merged[-1]
        gap_text = text[prev.end:cur.start] if cur.start >= prev.end else ""
        same_type = prev.entity_type == cur.entity_type
        overlap_or_touch = cur.start <= prev.end
        person_gap_ok = prev.entity_type == "PERSON" and re.fullmatch(r"[\s.'-]*", gap_text or "")
        if same_type and (overlap_or_touch or person_gap_ok):
            prev.end = max(prev.end, cur.end)
            prev.score = max(prev.score, cur.score)
            prev.source = f"{prev.source}+{cur.source}"
        else:
            merged.append(cur)

    return merged


def expand_person_titles(results: list[SimpleResult], word_map: list[WordBox], text: str) -> list[SimpleResult]:
    if not results or not word_map:
        return results

    for r in results:
        if r.entity_type != "PERSON":
            continue
        prev_word = None
        for wb in word_map:
            if wb.end <= r.start:
                prev_word = wb
            else:
                break
        if prev_word is None:
            continue
        gap = text[prev_word.end:r.start]
        token = re.sub(r"^[^\w]+|[^\w]+$", "", prev_word.text).lower()
        if token in NAME_TITLES and re.fullmatch(r"[\s.]*", gap or ""):
            r.start = prev_word.start
    return results


def resolve_conflicting_overlaps(results: list[SimpleResult], text: str) -> list[SimpleResult]:
    if not results:
        return []

    results = sorted(results, key=lambda r: (r.start, -(r.end - r.start), -ENTITY_PRIORITY.get(r.entity_type, 0), -r.score))
    kept: list[SimpleResult] = []

    for cur in results:
        cur.start, cur.end = trim_span_whitespace(text, cur.start, cur.end)
        if cur.start >= cur.end:
            continue
        if not is_whole_word_span(text, cur.start, cur.end):
            continue

        suppress = False
        for existing in kept:
            overlap = not (cur.end <= existing.start or cur.start >= existing.end)
            if not overlap:
                continue

            same_type = cur.entity_type == existing.entity_type
            if same_type:
                existing.start = min(existing.start, cur.start)
                existing.end = max(existing.end, cur.end)
                existing.score = max(existing.score, cur.score)
                existing.source = f"{existing.source}+{cur.source}"
                suppress = True
                break

            cur_pri = ENTITY_PRIORITY.get(cur.entity_type, 0)
            ex_pri = ENTITY_PRIORITY.get(existing.entity_type, 0)
            if cur_pri > ex_pri or (cur_pri == ex_pri and cur.score > existing.score):
                existing.start = cur.start
                existing.end = cur.end
                existing.entity_type = cur.entity_type
                existing.score = cur.score
                existing.source = cur.source
            suppress = True
            break

        if not suppress:
            kept.append(cur)

    return sorted(kept, key=lambda r: (r.start, r.end))


def page_looks_like_public_directory(results: list[SimpleResult], text: str) -> bool:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.entity_type] = counts.get(r.entity_type, 0) + 1

    address_count = counts.get("US_ADDRESS", 0) + counts.get("ZIP_CODE", 0)
    sensitive_count = sum(counts.get(k, 0) for k in ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "MEDICAL_RECORD_NUMBER"])
    lower = text.lower()
    has_personal_context = any(x in lower for x in ["patient", "name:", "email:", "phone:", "address:", "home address", "date of birth", "mrn"])
    return address_count >= 8 and sensitive_count <= 2 and not has_personal_context


def filter_results(results: list[SimpleResult], text: str) -> list[SimpleResult]:
    filtered: list[SimpleResult] = []
    directory_like = page_looks_like_public_directory(results, text)

    for r in results:
        snippet = normalize_space(text[r.start:r.end])
        if not snippet:
            continue
        if is_allowlisted(snippet):
            continue

        if r.entity_type == "PERSON":
            if looks_like_public_org(snippet):
                continue
            if len(snippet) == 1:
                continue
            # Avoid redacting ordinary sentence fragments that do not look name-like.
            cleaned_tokens = [re.sub(r"^[^\w]+|[^\w]+$", "", t) for t in snippet.split()]
            cleaned_tokens = [t for t in cleaned_tokens if t]
            if not cleaned_tokens:
                continue
            if not all(re.match(r"^[A-Z][a-zA-Z.'-]*$|^[A-Z]\.$", t) for t in cleaned_tokens):
                continue

        if r.entity_type == "US_ADDRESS":
            if directory_like:
                continue
            if not looks_like_strict_address(snippet):
                continue

        if r.entity_type == "ZIP_CODE" and directory_like:
            continue

        if r.entity_type == "STRICT_DATE":
            # Keep full dates; suppress vague month/year-ish fragments.
            if not re.search(r"\d", snippet):
                continue
            if re.fullmatch(r"(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)\.?\s+\d{4}", snippet, flags=re.I):
                continue

        filtered.append(r)

    return filtered


# =============================================================================
# CHAR-SPAN -> PDF RECTS
# =============================================================================
def span_to_rects(span: SimpleResult, word_map: list[WordBox]) -> list[fitz.Rect]:
    overlapping = [wb for wb in word_map if not (wb.end <= span.start or wb.start >= span.end)]
    if not overlapping:
        return []

    # Group by block/line so wrapped spans produce one box per line instead of a giant rectangle.
    grouped: dict[tuple[int, int], fitz.Rect] = {}
    for wb in overlapping:
        key = (wb.block, wb.line)
        if key not in grouped:
            grouped[key] = fitz.Rect(wb.rect)
        else:
            grouped[key] |= wb.rect

    rects = []
    for _, rr in sorted(grouped.items()):
        rr = fitz.Rect(rr)
        rr.x0 -= 0.6
        rr.y0 -= 0.6
        rr.x1 += 0.6
        rr.y1 += 0.6
        rects.append(rr)
    return rects


# =============================================================================
# PAGE PROCESSING
# =============================================================================
def detect_page_spans(page_text: str, entities: list[str], min_score: float) -> list[SimpleResult]:
    presidio_results = detect_pii_spans_presidio(page_text, entities, default_min=min_score)
    transformer_results = detect_pii_spans_transformer(page_text)

    debug_print_results(presidio_results, page_text, "Presidio")
    debug_print_results(transformer_results, page_text, "Transformer")

    combined = merge_same_type_overlaps(presidio_results + transformer_results, page_text)
    combined = resolve_conflicting_overlaps(combined, page_text)
    combined = filter_results(combined, page_text)
    combined = merge_same_type_overlaps(combined, page_text)
    return combined


def redact_pdf(
    input_pdf: str | Path,
    output_pdf: str | Path,
    entities: list[str] = DEFAULT_ENTITIES,
    min_score: float = 0.80,
    redact_color_rgb: tuple[int, int, int] = REDACT_COLOR_RGB,
) -> dict:
    input_pdf = Path(input_pdf)
    output_pdf = Path(output_pdf)

    if not input_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    doc = fitz.open(str(input_pdf))
    fill = tuple(c / 255 for c in redact_color_rgb)

    total_spans = 0
    total_boxes = 0
    pages_with_hits = 0
    entity_counts: dict[str, int] = {}

    for page_num, page in enumerate(doc, start=1):
        page_text, word_map = build_page_text_map(page)
        if not page_text.strip() or not word_map:
            continue

        page_results = detect_page_spans(page_text, entities=entities, min_score=min_score)
        page_results = expand_person_titles(page_results, word_map, page_text)
        page_results = merge_same_type_overlaps(page_results, page_text)
        page_results = resolve_conflicting_overlaps(page_results, page_text)
        page_results = filter_results(page_results, page_text)

        if DEBUG:
            debug_print_results(page_results, page_text, f"Final page {page_num}")

        if not page_results:
            continue

        pages_with_hits += 1
        total_spans += len(page_results)

        for r in page_results:
            entity_counts[r.entity_type] = entity_counts.get(r.entity_type, 0) + 1
            rects = span_to_rects(r, word_map)
            for rr in rects:
                page.add_redact_annot(rr, fill=fill)
                total_boxes += 1

    for page in doc:
        page.apply_redactions()

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_pdf))
    doc.close()

    return {
        "output": str(output_pdf),
        "pages": pages_with_hits,
        "total_spans": total_spans,
        "total_boxes": total_boxes,
        "entity_counts": entity_counts,
    }


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {INPUT_DIR.resolve()}")

    for input_pdf in pdfs:
        print(f"\nProcessing: {input_pdf.name}")
        output_pdf = OUTPUT_DIR / f"{input_pdf.stem}_REDACTED.pdf"

        try:
            stats = redact_pdf(
                input_pdf=input_pdf,
                output_pdf=output_pdf,
                entities=DEFAULT_ENTITIES,
                min_score=0.80,
            )
            print("Output file:", stats["output"])
            print("Pages with hits:", stats["pages"])
            print("Total spans:", stats["total_spans"])
            print("Total boxes:", stats["total_boxes"])
            if stats["entity_counts"]:
                print("Entity counts:")
                for entity, count in sorted(stats["entity_counts"].items()):
                    print(f"  {entity}: {count}")
            else:
                print("No entities redacted.")
        except Exception as exc:
            print(f"Failed on {input_pdf.name}: {exc}")
            fallback = OUTPUT_DIR / input_pdf.name
            shutil.copy2(input_pdf, fallback)
            print(f"Copied original to: {fallback}")


if __name__ == "__main__":
    main()