"""
FastAPI server for PII Redaction Tool.
Exposes POST /api/redact — accepts a PDF + entity list, returns a redacted PDF.

Run with:
    conda run -n redact uvicorn api.server:app --reload --port 8000
    python -u -m uvicorn api.server:app --port 8000
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import re as _re

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redact import DEFAULT_ENTITIES, extract_pii_terms_from_pdf, redact_pdf_names, redact_pdf_names_term, _log, TOKENIZER, MODEL, _DEVICE, TRANSFORMER_READY, ANALYZER  # noqa: E402
from contextlib import asynccontextmanager
import torch


def _warmup() -> None:
    dummy = "Patient John Doe, DOB 01/01/1980, SSN 123-45-6789, 123 Main St, Phoenix AZ 85001."

    print("Warming up Presidio/spaCy...")
    ANALYZER.analyze(text=dummy, language="en")
    print("Presidio warmup complete.")

    if not TRANSFORMER_READY:
        print("Transformer not loaded — skipping warmup.")
        return
    print("Warming up transformer model...")
    inputs = TOKENIZER(dummy, return_tensors="pt", truncation=True, max_length=64).to(_DEVICE)
    with torch.no_grad():
        MODEL(**inputs)
    print("Transformer warmup complete.")


@asynccontextmanager
async def lifespan(app):
    _warmup()
    yield


app = FastAPI(title="PII Redaction API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_ENTITIES = set(DEFAULT_ENTITIES)


@app.post("/api/redact")
async def redact(
    file: UploadFile = File(..., description="PDF to redact"),
    entities: str = Form(
        default=json.dumps(DEFAULT_ENTITIES),
        description="JSON array of PII entity types to redact",
    ),
    exclusions: str = Form(
        default="[]",
        description="JSON array of names to exclude from redaction",
    ),
) -> FileResponse:
    print(f"Received file: {file.filename} with entities: {entities}")
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # ── Parse + validate entity list ─────────────────────────────────────
    try:
        requested: list[str] = json.loads(entities)
        if not isinstance(requested, list):
            raise ValueError
    except (ValueError, json.JSONDecodeError):
        raise HTTPException(status_code=400, detail="'entities' must be a JSON array of strings.")

    selected = [e for e in requested if e in ALLOWED_ENTITIES]
    if not selected:
        raise HTTPException(status_code=400, detail="No valid entity types selected.")

    # ── Save upload to a temp file ────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        stem = Path(file.filename or "document").stem
        input_path = tmp / f"{stem}.pdf"
        output_path = tmp / f"{stem}_REDACTED.pdf"

        input_path.write_bytes(await file.read())

        try:
            excluded: list[str] = json.loads(exclusions)
            if not isinstance(excluded, list):
                excluded = []
        except (ValueError, json.JSONDecodeError):
            excluded = []

        _SAFE_EXCLUSION = _re.compile(r"^[^\x00-\x1f<>`;\|\\]{1,200}$")
        excluded = [e for e in excluded if isinstance(e, str) and _SAFE_EXCLUSION.match(e.strip())]

        excluded_lower = set()
        for e in excluded:
            e = e.strip().lower()
            if not e:
                continue
            excluded_lower.add(e)

        pii_terms = extract_pii_terms_from_pdf(
            input_path,
            entities=selected,
            min_score=0.6,
        )

        def _core(t: str) -> str:
            return _re.sub(r"[^\w\s]", "", t).strip().lower()

        def _is_excluded(t: str) -> bool:
            core = _core(t)
            if core in excluded_lower:
                return True
            core_words = set(core.split())
            return bool(core_words & excluded_lower)

        _log(f"\n=== Exclusion filter (excluded_lower={excluded_lower}) ===")
        print(f"\n=== Exclusion filter (excluded_lower={excluded_lower}) ===")
        kept, dropped = [], []
        for t in pii_terms:
            if _is_excluded(t):
                dropped.append(t)
            else:
                kept.append(t)
        for t in dropped:
            _log(f"  [EXCLUDED] {repr(t)}")
            print(f"  [EXCLUDED] {repr(t)}")
        for t in kept:
            _log(f"  [SENDING TO REDACT] {repr(t)}")
            print(f"  [SENDING TO REDACT] {repr(t)}")
        pii_terms = kept

        if not pii_terms:
            _log("  No terms remain after exclusion — returning original file.")
            print("  No terms remain after exclusion — returning original file.")

            output_path = input_path
        else:
            redact_pdf_names(
                input_pdf=input_path,
                output_pdf=output_path,
                names=pii_terms,
                case_insensitive=True,
            )

        final_path = Path(tempfile.mktemp(suffix=".pdf"))
        final_path.write_bytes(output_path.read_bytes())

    return FileResponse(
        path=str(final_path),
        media_type="application/pdf",
        filename=f"{stem}_REDACTED.pdf",
        background=None,
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/redact-terms")
async def redact_terms(
    file: UploadFile = File(..., description="PDF to redact"),
    terms: str = Form(..., description="JSON array of exact terms to redact"),
) -> FileResponse:
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        term_list: list[str] = json.loads(terms)
        if not isinstance(term_list, list):
            raise ValueError
    except (ValueError, json.JSONDecodeError):
        raise HTTPException(status_code=400, detail="'terms' must be a JSON array of strings.")

    term_list = [t.strip() for t in term_list if t.strip()]
    if not term_list:
        raise HTTPException(status_code=400, detail="No terms provided.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        stem = Path(file.filename or "document").stem
        input_path = tmp / f"{stem}.pdf"
        output_path = tmp / f"{stem}_REDACTED.pdf"
        input_path.write_bytes(await file.read())

        redact_pdf_names_term(
            input_pdf=input_path,
            output_pdf=output_path,
            names=term_list,
            case_insensitive=True,
        )

        final_path = Path(tempfile.mktemp(suffix=".pdf"))
        final_path.write_bytes(output_path.read_bytes())

    return FileResponse(
        path=str(final_path),
        media_type="application/pdf",
        filename=f"{stem}_REDACTED.pdf",
        background=None,
        headers={"Cache-Control": "no-store"},
    )
