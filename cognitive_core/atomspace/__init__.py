"""
AtomSpace module for EPPN Cognitive Core

Provides concept graph representations and atom storage for policy data.
Inspired by OpenCog's AtomSpace for symbolic AI and knowledge representation.
"""

from .atomspace_manager import AtomSpaceManager
from .atom_types import AtomType, Atom
from .concept_graph import ConceptGraph

__all__ = [
    "AtomSpaceManager",
    "AtomType", 
    "Atom",
    "ConceptGraph"
]
