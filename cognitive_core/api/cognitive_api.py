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
from tools.agentverse_client import get_client
from tools.huggingface_client import get_client as get_hf_client
from tools.deepseek_client import get_client as get_deepseek_client
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None
from tools.huggingface_client import get_client as get_hf_client
try:
    from transformers import pipeline as hf_pipeline
except Exception:  # pragma: no cover - optional dependency
    hf_pipeline = None


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
_agentverse = get_client()
_hf_client = get_hf_client()
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "auto").lower()


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


def sections_to_text(sections: List[Dict[str, Any]]) -> str:
    parts = []
    for s in sections:
        title = s.get("title", "")
        text = s.get("text", "")
        if title:
            parts.append(title + "\n")
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


def call_hosted_summarizer(sections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    # Prefer based on MODEL_PROVIDER: 'agentverse', 'hf', 'local', or 'auto'
    if MODEL_PROVIDER in ("agentverse", "auto") and _agentverse.enabled:
        try:
            summarizer_prompt = json.dumps({
                "task": "summarize_policy",
                "instructions": "Produce a 2-3 sentence executive summary and 5 bullet key points aimed at policymakers.",
                "sections": sections
            })
            return _agentverse.call_model("gema-2", summarizer_prompt, params={"temperature": 0.0}, timeout=120)
        except Exception:
            pass

    if MODEL_PROVIDER in ("hf", "auto") and _hf_client:
        try:
            text = sections_to_text(sections)
            # Use a summarization model on HF
            res = _hf_client.call_model("sshleifer/distilbart-cnn-12-6", text, params={"max_length": 120}, timeout=120)
            # Normalize response
            if isinstance(res, list):
                first = res[0]
                if isinstance(first, dict):
                    s = first.get("summary_text") or first.get("generated_text") or str(first)
                else:
                    s = str(first)
            elif isinstance(res, dict):
                s = res.get("summary_text") or res.get("generated_text") or json.dumps(res)
            else:
                s = str(res)
            sentences = [p.strip() for p in s.split('.') if p.strip()]
            short = " ".join(sentences[:2]) if sentences else s
            key_points = sentences[2:7] if len(sentences) > 2 else []
            return {"short_summary": short, "key_points": key_points}
        except Exception:
            pass

    # Local fallback
    return local_summarize_sections(sections)


def call_hosted_ethics(sections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if MODEL_PROVIDER in ("agentverse", "auto") and _agentverse.enabled:
        try:
            ethics_prompt = json.dumps({
                "task": "ethical_analysis",
                "instructions": "Return JSON with keys: ethical_summary (string), contradictions (list), recommendations (list).",
                "sections": sections,
                "frameworks": ["justice", "sustainability", "inclusion"]
            })
            return _agentverse.call_model("gema-2", ethics_prompt, params={"temperature": 0.0}, timeout=180)
        except Exception:
            pass

    if MODEL_PROVIDER in ("hf", "auto") and _hf_client:
        try:
            # Build a text prompt asking HF model to output JSON
            text = "Perform an ethical analysis and return a JSON object with keys: 'ethical_summary', 'contradictions', 'recommendations'. Sections:\n\n" + sections_to_text(sections)
            res = _hf_client.call_model("google/flan-t5-small", text, timeout=180)
            # Parse HF result: try JSON extraction
            if isinstance(res, list) and res:
                out_text = res[0].get("generated_text") if isinstance(res[0], dict) else str(res[0])
            elif isinstance(res, dict):
                out_text = res.get("generated_text") or json.dumps(res)
            else:
                out_text = str(res)
            # Try to parse JSON from the model output
            try:
                parsed = json.loads(out_text)
                return parsed if isinstance(parsed, dict) else {"ethical_summary": out_text}
            except Exception:
                # fallback: return text as ethical_summary
                return {"ethical_summary": out_text, "contradictions": [], "recommendations": []}
        except Exception:
            pass

    return None


def local_summarize_sections(sections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Produce a short summary and key points using a local transformers model.

    This is a best-effort fallback when hosted summarization is unavailable.
    """
    if hf_pipeline is None:
        return None
    try:
        # Use a lightweight summarization model
        summarizer = hf_pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        texts = [s.get("text", "") for s in sections if s.get("text")]
        if not texts:
            return None
        # Chunk text to avoid oversized inputs
        max_chunk = 800
        chunks = []
        current = ""
        for t in texts:
            if len(current) + len(t) + 1 <= max_chunk:
                current = (current + " " + t).strip()
            else:
                if current:
                    chunks.append(current)
                current = t[:max_chunk]
        if current:
            chunks.append(current)

        summaries = []
        for c in chunks:
            out = summarizer(c, max_length=120, min_length=30, do_sample=False)
            summaries.append(out[0]["summary_text"] if isinstance(out, list) and out else str(out))

        combined = " ".join(summaries)
        # Derive short summary and key points (naive sentence split)
        sentences = [s.strip() for s in combined.split('.') if s.strip()]
        short = " ".join(sentences[:2]) if sentences else combined
        key_points = sentences[2:7] if len(sentences) > 2 else []
        return {"short_summary": short, "key_points": key_points}
    except Exception:
        return None


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

            # If AgentVerse client enabled, call hosted models for nicer output
            hosted_summary = None
            hosted_ethics = None
            if _agentverse.enabled:
                try:
                    summarizer_prompt = json.dumps({
                        "task": "summarize_policy",
                        "instructions": "Produce a 2-3 sentence executive summary and 5 bullet key points aimed at policymakers.",
                        "sections": sections
                    })
                    hosted_summary = _agentverse.call_model("gema-2", summarizer_prompt, params={"temperature": 0.0}, timeout=120)
                except Exception:
                    hosted_summary = None

                try:
                    ethics_prompt = json.dumps({
                        "task": "ethical_analysis",
                        "instructions": "Return JSON with keys: ethical_summary (string), contradictions (list), recommendations (list).",
                        "sections": sections,
                        "frameworks": ["justice", "sustainability", "inclusion"]
                    })
                    hosted_ethics = _agentverse.call_model("gema-2", ethics_prompt, params={"temperature": 0.0}, timeout=180)
                except Exception:
                    hosted_ethics = None

            if hosted_ethics is None:
                # Try a simple HF prompt for ethics if available
                try:
                    hf = get_hf_client()
                    combined = "\n\n".join([s.get("text", "") for s in sections])
                    ethics_prompt = f"Analyze the following policy text for ethical implications, contradictions, and recommendations. Return JSON: {{'ethical_summary': ..., 'contradictions': [...], 'recommendations': [...]}}. Text:\n\n{combined}"
                    hf_res = hf.summarize(ethics_prompt, max_length=200, min_length=50)
                    if isinstance(hf_res, dict) and hf_res.get("summary_text"):
                        # Put the result into a simple structure
                        hosted_ethics = {"ethical_summary": hf_res.get("summary_text"), "contradictions": [], "recommendations": []}
                except Exception:
                    hosted_ethics = None

            # Try Hugging Face if AgentVerse not available or did not return a hosted summary
            if hosted_summary is None:
                try:
                    hf = get_hf_client()
                    model_name = os.environ.get("HUGGINGFACE_MODEL_SUMMARY", "sshleifer/distilbart-cnn-12-6")
                    combined = "\n\n".join([s.get("text", "") for s in sections])
                    # DeepSeek works on images; if selected, call deepseek image OCR pipeline
                    if model_name.startswith("deepseek-ai/"):
                        ds = get_deepseek_client(model_name)
                        # Convert first page of PDF to image
                        pdf_path = path
                        page_image = None
                        if fitz is not None:
                            doc = fitz.open(pdf_path)
                            pix = doc.load_page(0).get_pixmap()
                            tmp = os.path.join("/tmp", f"{os.path.basename(pdf_path)}.png")
                            pix.save(tmp)
                            page_image = tmp
                        elif convert_from_path is not None:
                            imgs = convert_from_path(pdf_path, first_page=1, last_page=1)
                            tmp = os.path.join("/tmp", f"{os.path.basename(pdf_path)}.png")
                            imgs[0].save(tmp)
                            page_image = tmp
                        if page_image:
                            hosted_summary = {"summary_text": ds.infer_image_to_markdown(page_image, prompt=None)}
                        else:
                            hosted_summary = None
                    else:
                        hosted_summary = hf.summarize(combined)
                except Exception:
                    hosted_summary = None
            else:
                # Try Hugging Face Inference API if available
                hf_client = get_hf_client()
                try:
                    hosted_summary = hf_client.call_model("sshleifer/distilbart-cnn-12-6", sections, params={"max_length": 120}, timeout=120)
                except Exception:
                    hosted_summary = None

                try:
                    # No direct ethics model assumed; try a text2text inference model
                    hosted_ethics = hf_client.call_model("google/flan-t5-small", {
                        "task": "ethical_analysis",
                        "instructions": "Return JSON with keys: ethical_summary, contradictions, recommendations.",
                        "sections": sections
                    }, timeout=180)
                except Exception:
                    hosted_ethics = None

            out = {
                "doc": os.path.basename(path),
                "summary": {
                    "short": (hosted_summary.get("short_summary") if isinstance(hosted_summary, dict) and hosted_summary.get("short_summary") else (hosted_summary.get("summary_text") if isinstance(hosted_summary, dict) and hosted_summary.get("summary_text") else f"Detected {summary_counts.get('contradictions', 0)} contradictions, {summary_counts.get('ethical_implications', 0)} ethical implications, and {summary_counts.get('patterns', 0)} patterns.")),
                    "counts": summary_counts
                },
                "key_points": (hosted_summary.get("key_points") if isinstance(hosted_summary, dict) and hosted_summary.get("key_points") else key_points),
                "contradictions": (hosted_ethics.get("contradictions") if isinstance(hosted_ethics, dict) and hosted_ethics.get("contradictions") else contradictions),
                "recommendations": (hosted_ethics.get("recommendations") if isinstance(hosted_ethics, dict) and hosted_ethics.get("recommendations") else []),
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

            hosted_summary = None
            hosted_ethics = None
            if _agentverse.enabled:
                try:
                    summarizer_prompt = json.dumps({
                        "task": "summarize_policy",
                        "instructions": "Produce a 2-3 sentence executive summary and 5 bullet key points aimed at policymakers.",
                        "sections": sections
                    })
                    hosted_summary = _agentverse.call_model("gema-2", summarizer_prompt, params={"temperature": 0.0}, timeout=120)
                except Exception:
                    hosted_summary = None

                try:
                    ethics_prompt = json.dumps({
                        "task": "ethical_analysis",
                        "instructions": "Return JSON with keys: ethical_summary (string), contradictions (list), recommendations (list).",
                        "sections": sections,
                        "frameworks": ["justice", "sustainability", "inclusion"]
                    })
                    hosted_ethics = _agentverse.call_model("gema-2", ethics_prompt, params={"temperature": 0.0}, timeout=180)
                except Exception:
                    hosted_ethics = None

            out = {
                "doc": os.path.basename(path),
                "summary": {
                    "short": (hosted_summary.get("short_summary") if isinstance(hosted_summary, dict) and hosted_summary.get("short_summary") else f"Detected {summary_counts.get('contradictions', 0)} contradictions, {summary_counts.get('ethical_implications', 0)} ethical implications, and {summary_counts.get('patterns', 0)} patterns."),
                    "counts": summary_counts
                },
                "key_points": (hosted_summary.get("key_points") if isinstance(hosted_summary, dict) and hosted_summary.get("key_points") else key_points),
                "contradictions": (hosted_ethics.get("contradictions") if isinstance(hosted_ethics, dict) and hosted_ethics.get("contradictions") else contradictions),
                "recommendations": (hosted_ethics.get("recommendations") if isinstance(hosted_ethics, dict) and hosted_ethics.get("recommendations") else []),
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


