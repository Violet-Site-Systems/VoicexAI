"""
Ethical Analyzer for EPPN Cognitive Core

Provides high-level ethical analysis APIs combining PLN reasoning, pattern
mining, and attention to produce ethics reports.
"""

from typing import Dict, List, Any

from ..atomspace.atom_types import Atom
from ..atomspace.atomspace_manager import AtomSpaceManager
from .pln_reasoner import PLNReasoner
from .pattern_miner import PatternMiner
from .attention_controller import AttentionController


class EthicalAnalyzer:
    """Coordinates reasoning modules to produce ethics assessments."""

    def __init__(self, atomspace_manager: AtomSpaceManager):
        self.atomspace = atomspace_manager
        self.reasoner = PLNReasoner(atomspace_manager)
        self.miner = PatternMiner(atomspace_manager)
        self.attention = AttentionController(atomspace_manager)

    def analyze_policy(self, policy_atoms: List[Atom]) -> Dict[str, Any]:
        """Run a complete ethical analysis pipeline on policy atoms."""
        # Boost attention to focus the system on provided atoms
        self.attention.boost_attention(policy_atoms)

        contradictions = self.reasoner.detect_contradictions(policy_atoms)
        fairness = self.reasoner.detect_fairness_patterns(policy_atoms)

        ethical_implications: List[dict] = []
        for atom in policy_atoms:
            ethical_implications.extend(self.reasoner.analyze_ethical_implications(atom))

        # Urban planning specific analysis
        urban_planning_ethics = self.reasoner.analyze_urban_planning_ethics(policy_atoms)
        resource_allocation_fairness = self.reasoner.analyze_resource_allocation_fairness(policy_atoms)

        text_patterns = self.miner.mine_text_patterns(policy_atoms)
        ethical_patterns = self.miner.mine_ethical_patterns(policy_atoms)
        structural_patterns = self.miner.mine_structural_patterns(policy_atoms)
        relationship_patterns = self.miner.mine_concept_relationships(policy_atoms)

        report = {
            "summary": {
                "contradictions": len(contradictions),
                "fairness_signals": len(fairness),
                "ethical_implications": len(ethical_implications),
                "urban_planning_ethics": len(urban_planning_ethics),
                "resource_allocation_issues": len(resource_allocation_fairness),
                "patterns": len(text_patterns) + len(ethical_patterns) + len(structural_patterns) + len(relationship_patterns)
            },
            "contradictions": [c.__dict__ for c in contradictions],
            "fairness": [f.__dict__ for f in fairness],
            "ethical_implications": [e.__dict__ for e in ethical_implications],
            "urban_planning_ethics": [u.__dict__ for u in urban_planning_ethics],
            "resource_allocation_fairness": [r.__dict__ for r in resource_allocation_fairness],
            "patterns": {
                "text": [p.__dict__ for p in text_patterns],
                "ethical": [p.__dict__ for p in ethical_patterns],
                "structural": [p.__dict__ for p in structural_patterns],
                "relationships": [p.__dict__ for p in relationship_patterns]
            }
        }

        # Normalize attention after analysis
        self.attention.normalize_attention()

        return report


