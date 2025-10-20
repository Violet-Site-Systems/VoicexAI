"""
Reasoning module for EPPN Cognitive Core

Provides probabilistic logic reasoning, pattern mining, and ethical analysis
capabilities inspired by OpenCog's PLN (Probabilistic Logic Networks).
"""

from .pln_reasoner import PLNReasoner
from .pattern_miner import PatternMiner
from .ethical_analyzer import EthicalAnalyzer
from .attention_controller import AttentionController

__all__ = [
    "PLNReasoner",
    "PatternMiner", 
    "EthicalAnalyzer",
    "AttentionController"
]
