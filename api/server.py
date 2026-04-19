"""
FastAPI server for PII Redaction Tool.
Exposes POST /api/redact — accepts a PDF + entity list, returns a redacted PDF.

Run with:
    conda run -n redact uvicorn api.server:app --reload --port 8000
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Ensure project root is on the path so redact.py is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redactt import DEFAULT_ENTITIES, extract_pii_terms_from_pdf, redact_pdf_names  # noqa: E402

app = FastAPI(title="PII Redaction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["POST"],
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
) -> FileResponse:
    # ── Validate file type ────────────────────────────────────────────────
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

        # ── Run redaction ─────────────────────────────────────────────────
        pii_terms = extract_pii_terms_from_pdf(
            input_path,
            entities=selected,
            min_score=0.6,
        )

        if not pii_terms:
            # Nothing to redact — return the original file unchanged
            output_path = input_path
        else:
            redact_pdf_names(
                input_pdf=input_path,
                output_pdf=output_path,
                names=pii_terms,
                case_insensitive=True,
            )

        # ── Stream back the redacted PDF ──────────────────────────────────
        # Copy to a stable path outside the temp dir before it's cleaned up
        final_path = Path(tempfile.mktemp(suffix=".pdf"))
        final_path.write_bytes(output_path.read_bytes())

    return FileResponse(
        path=str(final_path),
        media_type="application/pdf",
        filename=f"{stem}_REDACTED.pdf",
        background=None,
    )
