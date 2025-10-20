"""
Cognitive API for EPPN

Provides simple JSON-serializable interfaces for storing/retrieving atoms and
invoking cognitive reasoning pipelines.
"""

from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi import UploadFile, File, Form, Depends
from typing import Union
import requests
from PyPDF2 import PdfReader
import uuid
import os
import json

from ..atomspace.atom_types import Atom, AtomType
from ..atomspace.atomspace_manager import AtomSpaceManager
from ..reasoning.ethical_analyzer import EthicalAnalyzer
from ..reasoning.metta_interpreter import evaluate as evaluate_metta_program


class CognitiveAPI:
    """High-level API for agents to interact with the cognitive core."""

    def __init__(self, storage_path: str = "cognitive_core/atomspace/storage"):
        self.atomspace = AtomSpaceManager(storage_path=storage_path)
        self.ethics = EthicalAnalyzer(self.atomspace)

    # Atom CRUD
    def create_atom(self, data: Dict[str, Any]) -> str:
        atom = Atom.from_dict(data) if "uuid" in data else Atom(
            atom_type=AtomType(data.get("atom_type", AtomType.POLICY_CONCEPT.value)),
            name=data.get("name", ""),
            content=data.get("content", {}),
            metadata=data.get("metadata", {})
        )
        return self.atomspace.add_atom(atom)

    def get_atom(self, uuid: str) -> Optional[Dict[str, Any]]:
        atom = self.atomspace.get_atom(uuid)
        return atom.to_dict() if atom else None

    def link_atoms(self, source_uuid: str, target_uuid: str, link_type: str = "RELATED") -> bool:
        self.atomspace.create_link(source_uuid, target_uuid, link_type)
        return True

    def search_atoms(self, query: str, atom_type: Optional[str] = None) -> List[Dict[str, Any]]:
        atype = AtomType(atom_type) if atom_type else None
        atoms = self.atomspace.search_atoms(query, atype)
        return [a.to_dict() for a in atoms]

    # Ethics pipeline
    def analyze_policy(self, atom_uuids: List[str]) -> Dict[str, Any]:
        atoms = [self.atomspace.get_atom(u) for u in atom_uuids]
        atoms = [a for a in atoms if a is not None]
        return self.ethics.analyze_policy(atoms)

    # Utilities
    def export(self, filepath: str) -> None:
        self.atomspace.export_atomspace(filepath)

    def import_(self, filepath: str) -> None:
        self.atomspace.import_atomspace(filepath)

    # MeTTa evaluation (small embedded interpreter)
    def evaluate_metta(self, program: str) -> Dict[str, object]:
        """Evaluate a small MeTTa-like program using the embedded interpreter.

        Returns result dict with facts, derived, and trace.
        """
        return evaluate_metta_program(program)


# --- FastAPI wrapper so the cognitive core can be run with uvicorn ---
app = FastAPI(title="Cognitive Core - MeTTa-enabled")

# Single backend instance used by the HTTP API
_backend = CognitiveAPI()


class MeTTaRequest(BaseModel):
    program: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/metta/evaluate")
def metta_evaluate(req: MeTTaRequest):
    try:
        result = _backend.evaluate_metta(req.program)
        # Convert sets/tuples into JSON-serializable lists
        def serialize(d):
            if isinstance(d, set):
                return [list(t) for t in d]
            if isinstance(d, tuple):
                return list(d)
            if isinstance(d, dict):
                return {k: serialize(v) for k, v in d.items()}
            if isinstance(d, list):
                return [serialize(x) for x in d]
            return d

        return serialize(result)
    except Exception as e:  # pragma: no cover - surfacing runtime errors
        raise HTTPException(status_code=500, detail=str(e))


class PipelineRunRequest(BaseModel):
    urls: list[str] = []


def download_pdf(url: str, out_dir: str = "docs") -> str:
    os.makedirs(out_dir, exist_ok=True)
    fname = url.split("/")[-1] or f"doc-{uuid.uuid4()}.pdf"
    path = os.path.join(out_dir, fname)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def extract_sections(pdf_path: str):
    reader = PdfReader(pdf_path)
    sections = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        sections.append({"title": f"Page {i+1}", "text": text, "page": i+1})
    return sections


def stringify_contradiction(c):
    # Try common fields, otherwise fallback to str()
    try:
        if isinstance(c, dict):
            return json.dumps({k: (str(v) if not isinstance(v, (str, int, float, bool, list, dict)) else v) for k, v in c.items()})
        # object with attributes
        for attr in ("conclusion", "premises", "evidence", "confidence"):
            if hasattr(c, attr):
                val = getattr(c, attr)
                return f"{attr}: {val}"
        return str(c)
    except Exception:
        return str(c)


@app.post("/pipeline/run")
def pipeline_run(req: PipelineRunRequest | None = None, file: UploadFile | None = File(None)):
    """Run a simple ingestion->analysis pipeline.

    Accepts either JSON body with `urls: [..]` or a single PDF upload (multipart/form-data `file`).
    Returns a concise JSON report and saves artifacts under `docs/` and `data/`.
    """
    try:
        os.makedirs("docs", exist_ok=True)
        os.makedirs("data", exist_ok=True)

        targets: list[str] = []
        if file is not None:
            # save uploaded file
            fname = file.filename or f"upload-{uuid.uuid4()}.pdf"
            path = os.path.join("docs", fname)
            with open(path, "wb") as f:
                f.write(file.file.read())
            targets.append(path)
        if req and getattr(req, "urls", None):
            for url in req.urls:
                path = download_pdf(url)
                targets.append(path)

        if not targets:
            raise HTTPException(status_code=400, detail="No urls or file provided")

        reports = []
        for path in targets:
            # extract sections
            sections = extract_sections(path)

            # create atoms
            atom_ids = []
            for s in sections:
                aid = _backend.create_atom({
                    "atom_type": "PolicySection",
                    "name": s.get("title", "section"),
                    "content": s,
                    "metadata": {"source": path}
                })
                atom_ids.append(aid)

            report = _backend.analyze_policy(atom_ids)

            # Build human-friendly summary
            summary_counts = report.get("summary", {})
            key_points = [s.get("title") for s in sections[:5]]

            contradictions = [stringify_contradiction(c) for c in report.get("contradictions", [])]

            out = {
                "doc": os.path.basename(path),
                "summary": {
                    "short": f"Detected {summary_counts.get('contradictions', 0)} contradictions, {summary_counts.get('ethical_implications', 0)} ethical implications, and {summary_counts.get('patterns', 0)} patterns.",
                    "counts": summary_counts
                },
                "key_points": key_points,
                "contradictions": contradictions,
                "artifacts": {
                    "pdf": path,
                    "json": f"data/{os.path.basename(path)}.report.json"
                }
            }

            # Save JSON report shallowly
            try:
                with open(out["artifacts"]["json"], "w", encoding="utf-8") as f:
                    json.dump({"report": report, "doc": out["doc"]}, f, default=str, indent=2)
            except Exception:
                # fallback: save limited info
                with open(out["artifacts"]["json"], "w", encoding="utf-8") as f:
                    json.dump({"doc": out["doc"], "summary": out["summary"]}, f, indent=2)

            reports.append(out)

        return {"status": "ok", "results": reports}

    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline/run_url")
def pipeline_run_url(req: PipelineRunRequest):
    """Accept JSON body with `urls: [...]` and run the pipeline (URL-only).
    This avoids multipart/form-data requirements when calling with application/json.
    """
    # reuse logic from pipeline_run but only for URLs
    try:
        if not req or not getattr(req, 'urls', None):
            raise HTTPException(status_code=400, detail="No urls provided")

        os.makedirs("docs", exist_ok=True)
        os.makedirs("data", exist_ok=True)

        reports = []
        for url in req.urls:
            path = download_pdf(url)
            sections = extract_sections(path)

            atom_ids = []
            for s in sections:
                aid = _backend.create_atom({
                    "atom_type": "PolicySection",
                    "name": s.get("title", "section"),
                    "content": s,
                    "metadata": {"source": path}
                })
                atom_ids.append(aid)

            report = _backend.analyze_policy(atom_ids)

            summary_counts = report.get("summary", {})
            key_points = [s.get("title") for s in sections[:5]]
            contradictions = [stringify_contradiction(c) for c in report.get("contradictions", [])]

            out = {
                "doc": os.path.basename(path),
                "summary": {
                    "short": f"Detected {summary_counts.get('contradictions', 0)} contradictions, {summary_counts.get('ethical_implications', 0)} ethical implications, and {summary_counts.get('patterns', 0)} patterns.",
                    "counts": summary_counts
                },
                "key_points": key_points,
                "contradictions": contradictions,
                "artifacts": {
                    "pdf": path,
                    "json": f"data/{os.path.basename(path)}.report.json"
                }
            }

            try:
                with open(out["artifacts"]["json"], "w", encoding="utf-8") as f:
                    json.dump({"report": report, "doc": out["doc"]}, f, default=str, indent=2)
            except Exception:
                with open(out["artifacts"]["json"], "w", encoding="utf-8") as f:
                    json.dump({"doc": out["doc"], "summary": out["summary"]}, f, indent=2)

            reports.append(out)

        return {"status": "ok", "results": reports}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


