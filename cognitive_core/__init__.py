"""
EPPN Cognitive Core - OpenCog-inspired cognitive reasoning system

This module provides the core cognitive capabilities for the Ethical Policy Pipeline Network,
including AtomSpace management, reasoning engines, and ethical analysis tools.
"""

from .atomspace.atomspace_manager import AtomSpaceManager
from .reasoning.pln_reasoner import PLNReasoner
from .reasoning.pattern_miner import PatternMiner
from .api.cognitive_api import CognitiveAPI

__version__ = "1.0.0"
__author__ = "EPPN Development Team"

__all__ = [
    "AtomSpaceManager",
    "PLNReasoner", 
    "PatternMiner",
    "CognitiveAPI"
]
