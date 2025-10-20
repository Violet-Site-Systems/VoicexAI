"""
Cognitive API for EPPN

Provides simple JSON-serializable interfaces for storing/retrieving atoms and
invoking cognitive reasoning pipelines.
"""

from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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


