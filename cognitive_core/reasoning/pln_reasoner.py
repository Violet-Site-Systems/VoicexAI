"""
Probabilistic Logic Networks (PLN) Reasoner for EPPN Cognitive Core

Implements probabilistic reasoning capabilities inspired by OpenCog's PLN
for ethical analysis and policy evaluation.
"""

import math
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum

from ..atomspace.atom_types import Atom, AtomType, TruthValue
from ..atomspace.atomspace_manager import AtomSpaceManager


class InferenceRule(Enum):
    """PLN inference rules for reasoning."""
    
    # Deduction rules
    MODUS_PONENS = "ModusPonens"
    MODUS_TOLLENS = "ModusTollens"
    
    # Induction rules
    GENERALIZATION = "Generalization"
    ABDUCTION = "Abduction"
    
    # Probabilistic rules
    BAYESIAN_INFERENCE = "BayesianInference"
    CONJUNCTION = "Conjunction"
    DISJUNCTION = "Disjunction"
    NEGATION = "Negation"
    
    # Ethical reasoning rules
    ETHICAL_IMPLICATION = "EthicalImplication"
    FAIRNESS_ANALYSIS = "FairnessAnalysis"
    CONTRADICTION_DETECTION = "ContradictionDetection"


@dataclass
class InferenceResult:
    """Result of a PLN inference operation."""
    conclusion: str
    premises: List[str]
    rule_used: InferenceRule
    truth_value: TruthValue
    confidence: float
    evidence: List[str] = None


class PLNReasoner:
    """
    Probabilistic Logic Networks reasoner for ethical policy analysis.
    
    Provides:
    - Probabilistic inference using various PLN rules
    - Ethical reasoning and contradiction detection
    - Fairness analysis and bias detection
    - Evidence-based confidence scoring
    """
    
    def __init__(self, atomspace_manager: AtomSpaceManager):
        """Initialize the PLN reasoner."""
        self.atomspace = atomspace_manager
        self.inference_history: List[InferenceResult] = []
        
        # Ethical frameworks and principles
        self.ethical_frameworks = {
            "utilitarian": ["maximize_happiness", "minimize_suffering"],
            "deontological": ["respect_autonomy", "duty_based", "universal_principles"],
            "virtue_ethics": ["honesty", "justice", "compassion", "wisdom"],
            "care_ethics": ["relationships", "context", "responsibility"],
            "justice_theory": ["fairness", "equality", "procedural_justice"]
        }
        
        # Urban Planning and Resource Allocation specific frameworks
        self.urban_planning_frameworks = {
            "sustainable_development": [
                "environmental_protection", "economic_viability", "social_equity",
                "intergenerational_justice", "resource_conservation", "climate_resilience"
            ],
            "spatial_justice": [
                "equitable_access", "spatial_distribution", "territorial_rights",
                "place_based_equity", "geographic_fairness", "accessibility_standards"
            ],
            "resource_allocation_ethics": [
                "efficiency_optimization", "equity_considerations", "transparency",
                "stakeholder_participation", "accountability", "proportional_distribution"
            ],
            "urban_governance": [
                "democratic_participation", "institutional_transparency", "public_interest",
                "regulatory_fairness", "administrative_justice", "citizen_empowerment"
            ],
            "environmental_justice": [
                "environmental_equity", "pollution_burden_distribution", "green_space_access",
                "climate_vulnerability", "ecological_sustainability", "natural_resource_rights"
            ]
        }
        
        # Bias patterns to detect
        self.bias_patterns = {
            "discrimination": ["unequal_treatment", "protected_class", "disparate_impact"],
            "confirmation_bias": ["selective_evidence", "cherry_picking"],
            "anchoring": ["initial_bias", "reference_point"],
            "availability": ["recent_events", "salient_examples"]
        }
        
        # Urban planning specific bias patterns
        self.urban_planning_bias_patterns = {
            "spatial_discrimination": ["redlining", "zoning_bias", "infrastructure_inequality", "service_deserts"],
            "economic_gentrification": ["displacement", "affordability_crisis", "cultural_erasure", "wealth_concentration"],
            "environmental_racism": ["toxic_siting", "pollution_burden", "green_space_inequality", "climate_vulnerability"],
            "participation_bias": ["elite_capture", "consultation_theater", "language_barriers", "digital_divide"],
            "resource_hoarding": ["infrastructure_inequality", "service_concentration", "investment_bias", "maintenance_neglect"]
        }
    
    def detect_contradictions(self, policy_atoms: List[Atom]) -> List[InferenceResult]:
        """Detect contradictions in policy statements."""
        contradictions = []
        
        for i, atom1 in enumerate(policy_atoms):
            for atom2 in policy_atoms[i+1:]:
                contradiction = self._check_contradiction(atom1, atom2)
                if contradiction:
                    contradictions.append(contradiction)
        
        return contradictions
    
    def analyze_ethical_implications(self, policy_atom: Atom) -> List[InferenceResult]:
        """Analyze ethical implications of a policy."""
        ethical_results = []
        
        # Check against ethical frameworks
        for framework, principles in self.ethical_frameworks.items():
            for principle in principles:
                implication = self._check_ethical_implication(policy_atom, framework, principle)
                if implication:
                    ethical_results.append(implication)
        
        return ethical_results
    
    def detect_fairness_patterns(self, policy_atoms: List[Atom]) -> List[InferenceResult]:
        """Detect fairness patterns and potential bias."""
        fairness_results = []
        
        for atom in policy_atoms:
            for bias_type, patterns in self.bias_patterns.items():
                for pattern in patterns:
                    fairness_analysis = self._check_fairness_pattern(atom, bias_type, pattern)
                    if fairness_analysis:
                        fairness_results.append(fairness_analysis)
        
        return fairness_results
    
    def analyze_urban_planning_ethics(self, policy_atoms: List[Atom]) -> List[InferenceResult]:
        """Analyze urban planning and resource allocation specific ethical implications."""
        urban_ethics_results = []
        
        for atom in policy_atoms:
            # Check against urban planning frameworks
            for framework, principles in self.urban_planning_frameworks.items():
                for principle in principles:
                    implication = self._check_urban_planning_implication(atom, framework, principle)
                    if implication:
                        urban_ethics_results.append(implication)
            
            # Check for urban planning specific bias patterns
            for bias_type, patterns in self.urban_planning_bias_patterns.items():
                for pattern in patterns:
                    bias_analysis = self._check_urban_planning_bias(atom, bias_type, pattern)
                    if bias_analysis:
                        urban_ethics_results.append(bias_analysis)
        
        return urban_ethics_results
    
    def analyze_resource_allocation_fairness(self, policy_atoms: List[Atom]) -> List[InferenceResult]:
        """Analyze fairness in resource allocation decisions."""
        resource_results = []
        
        for atom in policy_atoms:
            # Check for resource allocation patterns
            allocation_analysis = self._analyze_allocation_patterns(atom)
            if allocation_analysis:
                resource_results.extend(allocation_analysis)
        
        return resource_results
    
    def probabilistic_inference(self, premises: List[str], rule: InferenceRule) -> Optional[InferenceResult]:
        """Perform probabilistic inference using specified rule."""
        premise_atoms = [self.atomspace.get_atom(uuid) for uuid in premises]
        premise_atoms = [atom for atom in premise_atoms if atom is not None]
        
        if len(premise_atoms) < 2:
            return None
        
        if rule == InferenceRule.MODUS_PONENS:
            return self._modus_ponens(premise_atoms)
        elif rule == InferenceRule.BAYESIAN_INFERENCE:
            return self._bayesian_inference(premise_atoms)
        elif rule == InferenceRule.CONJUNCTION:
            return self._conjunction(premise_atoms)
        elif rule == InferenceRule.ETHICAL_IMPLICATION:
            return self._ethical_implication(premise_atoms)
        
        return None
    
    def calculate_truth_value(self, atom: Atom, evidence: List[Atom]) -> TruthValue:
        """Calculate truth value based on evidence."""
        if not evidence:
            return atom.truth_value
        
        # Simple evidence-based truth calculation
        supporting_evidence = 0
        contradicting_evidence = 0
        
        for evidence_atom in evidence:
            if self._supports(atom, evidence_atom):
                supporting_evidence += 1
            elif self._contradicts(atom, evidence_atom):
                contradicting_evidence += 1
        
        total_evidence = supporting_evidence + contradicting_evidence
        if total_evidence == 0:
            return atom.truth_value
        
        strength = supporting_evidence / total_evidence
        confidence = min(1.0, total_evidence / 10.0)  # More evidence = higher confidence
        
        return TruthValue(strength=strength, confidence=confidence)
    
    def _check_contradiction(self, atom1: Atom, atom2: Atom) -> Optional[InferenceResult]:
        """Check if two atoms contradict each other."""
        # Simple contradiction detection based on content analysis
        content1 = str(atom1.content).lower()
        content2 = str(atom2.content).lower()
        
        # Check for explicit contradictions
        contradiction_pairs = [
            ("allow", "prohibit"), ("require", "forbid"), ("mandatory", "optional"),
            ("increase", "decrease"), ("include", "exclude"), ("positive", "negative")
        ]
        
        for pos, neg in contradiction_pairs:
            if (pos in content1 and neg in content2) or (neg in content1 and pos in content2):
                return InferenceResult(
                    conclusion=f"Contradiction detected between {atom1.name} and {atom2.name}",
                    premises=[atom1.uuid, atom2.uuid],
                    rule_used=InferenceRule.CONTRADICTION_DETECTION,
                    truth_value=TruthValue(strength=0.8, confidence=0.7),
                    confidence=0.7,
                    evidence=[atom1.uuid, atom2.uuid]
                )
        
        return None
    
    def _check_ethical_implication(self, policy_atom: Atom, framework: str, principle: str) -> Optional[InferenceResult]:
        """Check ethical implications of a policy against a framework principle."""
        content = str(policy_atom.content).lower()
        
        # Simple keyword-based ethical analysis
        ethical_keywords = {
            "utilitarian": ["benefit", "utility", "happiness", "welfare"],
            "deontological": ["duty", "right", "obligation", "principle"],
            "virtue_ethics": ["virtue", "character", "excellence", "wisdom"],
            "care_ethics": ["care", "relationship", "responsibility", "context"],
            "justice_theory": ["justice", "fairness", "equality", "rights"]
        }
        
        if framework in ethical_keywords:
            framework_keywords = ethical_keywords[framework]
            matches = sum(1 for keyword in framework_keywords if keyword in content)
            
            if matches > 0:
                strength = min(1.0, matches / len(framework_keywords))
                return InferenceResult(
                    conclusion=f"Policy aligns with {framework} principle: {principle}",
                    premises=[policy_atom.uuid],
                    rule_used=InferenceRule.ETHICAL_IMPLICATION,
                    truth_value=TruthValue(strength=strength, confidence=0.6),
                    confidence=0.6,
                    evidence=[policy_atom.uuid]
                )
        
        return None
    
    def _check_fairness_pattern(self, atom: Atom, bias_type: str, pattern: str) -> Optional[InferenceResult]:
        """Check for fairness patterns and potential bias."""
        content = str(atom.content).lower()
        
        # Bias detection keywords
        bias_keywords = {
            "discrimination": ["discriminate", "unequal", "bias", "prejudice"],
            "confirmation_bias": ["only", "exclusively", "always", "never"],
            "anchoring": ["first", "initial", "baseline", "starting"],
            "availability": ["recent", "notable", "famous", "well-known"]
        }
        
        if bias_type in bias_keywords:
            bias_keywords_list = bias_keywords[bias_type]
            matches = sum(1 for keyword in bias_keywords_list if keyword in content)
            
            if matches > 0:
                strength = min(1.0, matches / len(bias_keywords_list))
                return InferenceResult(
                    conclusion=f"Potential {bias_type} detected: {pattern}",
                    premises=[atom.uuid],
                    rule_used=InferenceRule.FAIRNESS_ANALYSIS,
                    truth_value=TruthValue(strength=strength, confidence=0.5),
                    confidence=0.5,
                    evidence=[atom.uuid]
                )
        
        return None
    
    def _modus_ponens(self, premises: List[Atom]) -> Optional[InferenceResult]:
        """Modus Ponens: If P then Q, P, therefore Q."""
        if len(premises) < 2:
            return None
        
        # Simple implementation - look for implication patterns
        premise1, premise2 = premises[0], premises[1]
        
        # Calculate combined truth value
        combined_strength = (premise1.truth_value.strength + premise2.truth_value.strength) / 2
        combined_confidence = min(premise1.truth_value.confidence, premise2.truth_value.confidence)
        
        return InferenceResult(
            conclusion=f"Modus Ponens inference from {premise1.name} and {premise2.name}",
            premises=[premise1.uuid, premise2.uuid],
            rule_used=InferenceRule.MODUS_PONENS,
            truth_value=TruthValue(strength=combined_strength, confidence=combined_confidence),
            confidence=combined_confidence
        )
    
    def _bayesian_inference(self, premises: List[Atom]) -> Optional[InferenceResult]:
        """Bayesian inference for probabilistic reasoning."""
        if not premises:
            return None
        
        # Simple Bayesian update
        prior_strength = premises[0].truth_value.strength
        likelihood = premises[1].truth_value.strength if len(premises) > 1 else 0.5
        
        # Simplified Bayes' theorem
        posterior_strength = (likelihood * prior_strength) / (
            likelihood * prior_strength + (1 - likelihood) * (1 - prior_strength)
        )
        
        return InferenceResult(
            conclusion="Bayesian inference result",
            premises=[atom.uuid for atom in premises],
            rule_used=InferenceRule.BAYESIAN_INFERENCE,
            truth_value=TruthValue(strength=posterior_strength, confidence=0.7),
            confidence=0.7
        )
    
    def _conjunction(self, premises: List[Atom]) -> Optional[InferenceResult]:
        """Logical conjunction of premises."""
        if not premises:
            return None
        
        # Conjunction strength is the minimum of all premises
        min_strength = min(atom.truth_value.strength for atom in premises)
        min_confidence = min(atom.truth_value.confidence for atom in premises)
        
        return InferenceResult(
            conclusion=f"Conjunction of {len(premises)} premises",
            premises=[atom.uuid for atom in premises],
            rule_used=InferenceRule.CONJUNCTION,
            truth_value=TruthValue(strength=min_strength, confidence=min_confidence),
            confidence=min_confidence
        )
    
    def _ethical_implication(self, premises: List[Atom]) -> Optional[InferenceResult]:
        """Ethical implication reasoning."""
        if not premises:
            return None
        
        # Analyze ethical implications
        ethical_scores = []
        for atom in premises:
            score = self._calculate_ethical_score(atom)
            ethical_scores.append(score)
        
        avg_ethical_score = sum(ethical_scores) / len(ethical_scores)
        
        return InferenceResult(
            conclusion=f"Ethical implication analysis (score: {avg_ethical_score:.2f})",
            premises=[atom.uuid for atom in premises],
            rule_used=InferenceRule.ETHICAL_IMPLICATION,
            truth_value=TruthValue(strength=avg_ethical_score, confidence=0.6),
            confidence=0.6
        )
    
    def _supports(self, atom1: Atom, atom2: Atom) -> bool:
        """Check if atom2 supports atom1."""
        # Simple support detection based on content similarity
        content1 = str(atom1.content).lower()
        content2 = str(atom2.content).lower()
        
        # Count common words
        words1 = set(content1.split())
        words2 = set(content2.split())
        common_words = words1.intersection(words2)
        
        return len(common_words) > 0
    
    def _contradicts(self, atom1: Atom, atom2: Atom) -> bool:
        """Check if atom2 contradicts atom1."""
        return self._check_contradiction(atom1, atom2) is not None
    
    def _calculate_ethical_score(self, atom: Atom) -> float:
        """Calculate ethical score for an atom."""
        content = str(atom.content).lower()
        
        # Positive ethical keywords
        positive_keywords = ["fair", "just", "equal", "right", "good", "benefit", "protect"]
        negative_keywords = ["unfair", "unjust", "unequal", "wrong", "harm", "discriminate"]
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in content)
        negative_count = sum(1 for keyword in negative_keywords if keyword in content)
        
        if positive_count + negative_count == 0:
            return 0.5  # Neutral
        
        return positive_count / (positive_count + negative_count)
    
    def _check_urban_planning_implication(self, policy_atom: Atom, framework: str, principle: str) -> Optional[InferenceResult]:
        """Check urban planning specific ethical implications."""
        content = str(policy_atom.content).lower()
        
        # Urban planning specific keywords
        urban_keywords = {
            "sustainable_development": ["sustainability", "green", "renewable", "conservation", "climate", "environment"],
            "spatial_justice": ["accessibility", "equity", "distribution", "territory", "geographic", "spatial"],
            "resource_allocation_ethics": ["allocation", "distribution", "efficiency", "transparency", "accountability"],
            "urban_governance": ["governance", "participation", "democracy", "transparency", "public", "citizen"],
            "environmental_justice": ["environmental", "pollution", "green_space", "climate", "ecological", "natural"]
        }
        
        if framework in urban_keywords:
            framework_keywords = urban_keywords[framework]
            matches = sum(1 for keyword in framework_keywords if keyword in content)
            
            if matches > 0:
                strength = min(1.0, matches / len(framework_keywords))
                return InferenceResult(
                    conclusion=f"Urban planning policy aligns with {framework} principle: {principle}",
                    premises=[policy_atom.uuid],
                    rule_used=InferenceRule.ETHICAL_IMPLICATION,
                    truth_value=TruthValue(strength=strength, confidence=0.7),
                    confidence=0.7,
                    evidence=[policy_atom.uuid]
                )
        
        return None
    
    def _check_urban_planning_bias(self, atom: Atom, bias_type: str, pattern: str) -> Optional[InferenceResult]:
        """Check for urban planning specific bias patterns."""
        content = str(atom.content).lower()
        
        # Urban planning bias keywords
        urban_bias_keywords = {
            "spatial_discrimination": ["zoning", "redlining", "infrastructure", "service", "accessibility"],
            "economic_gentrification": ["displacement", "affordability", "gentrification", "housing", "rent"],
            "environmental_racism": ["pollution", "toxic", "environmental", "burden", "vulnerability"],
            "participation_bias": ["consultation", "participation", "stakeholder", "public", "community"],
            "resource_hoarding": ["investment", "infrastructure", "maintenance", "service", "allocation"]
        }
        
        if bias_type in urban_bias_keywords:
            bias_keywords_list = urban_bias_keywords[bias_type]
            matches = sum(1 for keyword in bias_keywords_list if keyword in content)
            
            if matches > 0:
                strength = min(1.0, matches / len(bias_keywords_list))
                return InferenceResult(
                    conclusion=f"Potential urban planning {bias_type} detected: {pattern}",
                    premises=[atom.uuid],
                    rule_used=InferenceRule.FAIRNESS_ANALYSIS,
                    truth_value=TruthValue(strength=strength, confidence=0.6),
                    confidence=0.6,
                    evidence=[atom.uuid]
                )
        
        return None
    
    def _analyze_allocation_patterns(self, atom: Atom) -> List[InferenceResult]:
        """Analyze resource allocation patterns for fairness."""
        content = str(atom.content).lower()
        results = []
        
        # Resource allocation keywords
        allocation_keywords = ["budget", "funding", "investment", "allocation", "distribution", "resource"]
        equity_keywords = ["equity", "fairness", "equal", "proportional", "need-based", "merit-based"]
        
        has_allocation = any(keyword in content for keyword in allocation_keywords)
        has_equity = any(keyword in content for keyword in equity_keywords)
        
        if has_allocation:
            if has_equity:
                results.append(InferenceResult(
                    conclusion="Resource allocation policy includes equity considerations",
                    premises=[atom.uuid],
                    rule_used=InferenceRule.FAIRNESS_ANALYSIS,
                    truth_value=TruthValue(strength=0.8, confidence=0.7),
                    confidence=0.7,
                    evidence=[atom.uuid]
                ))
            else:
                results.append(InferenceResult(
                    conclusion="Resource allocation policy may lack equity considerations",
                    premises=[atom.uuid],
                    rule_used=InferenceRule.FAIRNESS_ANALYSIS,
                    truth_value=TruthValue(strength=0.3, confidence=0.6),
                    confidence=0.6,
                    evidence=[atom.uuid]
                ))
        
        return results
    
    def get_inference_history(self) -> List[InferenceResult]:
        """Get the history of all inferences made."""
        return self.inference_history.copy()
    
    def clear_inference_history(self):
        """Clear the inference history."""
        self.inference_history.clear()
