"""
ECAN-style Attention Controller for EPPN Cognitive Core

Manages attention allocation across atoms to prioritize reasoning focus,
inspired by OpenCog's ECAN (Economic Attention Networks).
"""

from typing import List

from ..atomspace.atom_types import Atom
from ..atomspace.atomspace_manager import AtomSpaceManager


class AttentionController:
    """Controls attention values and prioritization for atoms."""

    def __init__(self, atomspace_manager: AtomSpaceManager):
        self.atomspace = atomspace_manager

    def boost_attention(self, atoms: List[Atom], short_term_boost: float = 0.2) -> None:
        """Boost short-term attention for a list of atoms."""
        for atom in atoms:
            av = atom.attention_value
            av.short_term = min(1.0, av.short_term + short_term_boost)
            self.atomspace.update_attention(atom.uuid, av)

    def normalize_attention(self) -> None:
        """Normalize attention across AtomSpace to keep values bounded."""
        atoms = list(self.atomspace.atoms.values())
        if not atoms:
            return

        totals = [a.attention_value.get_total() for a in atoms]
        max_total = max(totals) if totals else 1.0
        if max_total <= 0:
            return

        for atom in atoms:
            av = atom.attention_value
            factor = av.get_total() / max_total
            av.long_term = min(1.0, max(av.long_term, factor * 0.5))
            self.atomspace.update_attention(atom.uuid, av)

    def decay_all(self, factor: float = 0.95) -> None:
        """Apply global attention decay."""
        self.atomspace.decay_attention(factor)

    def get_top_k(self, k: int = 10) -> List[Atom]:
        """Return top-k atoms by total attention."""
        atoms = list(self.atomspace.atoms.values())
        atoms.sort(key=lambda a: a.attention_value.get_total(), reverse=True)
        return atoms[:k]


