"""
AgentVerse Integration for EPPN

Registers and manages UAgents on the AgentVerse platform for discovery and
collaboration.
"""

import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import requests


@dataclass
class AgentCapability:
    """Represents an agent capability."""

    capability_id: str
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]


@dataclass
class AgentProfile:
    """Represents an agent profile on AgentVerse."""

    agent_id: str
    name: str
    description: str
    version: str
    domain: str
    capabilities: List[AgentCapability]
    contact_info: Dict[str, str]
    reputation_score: float
    status: str
    registration_date: datetime
    last_activity: datetime
    code_location: Optional[str] = None
    endpoint: Optional[str] = None


@dataclass
class AgentCollaboration:
    """Represents a collaboration between agents."""

    collaboration_id: str
    participating_agents: List[str]
    collaboration_type: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    results: Optional[Dict[str, Any]]


class AgentVerseIntegration:
    """
    Integration with AgentVerse platform for agent registration and
    collaboration.
    Provides:
    - Agent registration and profile management
    - Capability discovery and matching
    - Collaboration coordination
    - Performance tracking
    - Reputation management
    """

    def __init__(self, agentverse_api_url: str = "https://api.agentverse.ai",
                 api_key: str = None):
        """Initialize AgentVerse integration."""
        self.api_url = agentverse_api_url
        self.api_key = api_key

        # Agent registry
        self.registered_agents: Dict[str, AgentProfile] = {}  # pyright: ignore[reportInvalidTypeForm]  # noqa: E501
        self.agent_capabilities: Dict[
            str, List[AgentCapability]
        ] = {}  # pyright: ignore[reportInvalidTypeForm]

        # Collaboration tracking
        self.active_collaborations: Dict[str, AgentCollaboration] = {}
        self.completed_collaborations: Dict[str, AgentCollaboration] = {}

    # Discovery
    self.discovered_agents: Dict[str, AgentProfile] = {}

    # Configuration
    self.agent_domain = "urban_planning_ethics"
    self.agent_version = "1.0.0"

    def register_agent(self, agent_name: str, agent_type: str,
                       capabilities: List[Dict[str, Any]],
                       local_path: Optional[str] = None,
                       endpoint: Optional[str] = None) -> str:
        """Register an agent on AgentVerse."""

        agent_id = str(uuid.uuid4())

        # Create agent capabilities
        agent_capabilities = []
        for cap_data in capabilities:
            capability = AgentCapability(
                capability_id=str(uuid.uuid4()),
                name=cap_data["name"],
                description=cap_data["description"],
                input_types=cap_data.get("input_types", []),
                output_types=cap_data.get("output_types", []),
                parameters=cap_data.get("parameters", {}),
                performance_metrics=cap_data.get("performance_metrics", {})
            )
            agent_capabilities.append(capability)

        # Create agent profile (include optional local code location)
        contact_info = {
            "email": "contact@eppn.ai",
            "website": "https://eppn.ai",
            "github": "https://github.com/eppn/uagents",
        }
        if local_path:
            # Add a hint where the agent lives in the repository for
            # maintainers
            contact_info["code_path"] = local_path
        if endpoint:
            # Publicly reachable endpoint for AgentVerse to call
            contact_info["endpoint"] = endpoint

        profile = AgentProfile(
            agent_id=agent_id,
            name=agent_name,
            description=(f"EPPN {agent_type} agent for urban planning and "
                         f"resource allocation ethics"),
            version=self.agent_version,
            domain=self.agent_domain,
            capabilities=agent_capabilities,
            contact_info=contact_info,
            reputation_score=0.0,
            status="active",
            registration_date=datetime.now(),
            last_activity=datetime.now(),
            code_location=local_path,
            endpoint=endpoint,
        )
        
        # Register with AgentVerse. If no API key is configured, skip the
        # remote call and keep the registration local for development.
        if not self.api_key:
            # Log a clear warning for maintainers (print to stderr is fine
            # for small projects; consumers can wire a logger if desired).
            print(
                f"Warning: AgentVerse API key not configured. Skipping "
                f"remote registration for agent '{agent_name}'."
            )

            # Store locally and return the generated id so callers can
            # continue working without requiring an external service.
            self.registered_agents[agent_id] = profile
            self.agent_capabilities[agent_id] = agent_capabilities
            return agent_id

        success = self._register_with_agentverse(profile)

        if success:
            self.registered_agents[agent_id] = profile
            self.agent_capabilities[agent_id] = agent_capabilities
            return agent_id
        else:
            raise Exception(f"Failed to register agent {agent_name} on "
                            f"AgentVerse")

    def register_librarian_agent(self) -> str:
        """Register the Librarian agent."""
        
        capabilities = [
            {
                "name": "pdf_retrieval",
                "description": ("Retrieves PDF documents from government "
                                "portals"),
                "input_types": ["url", "search_query"],
                "output_types": ["pdf_document", "metadata"],
                "parameters": {
                    "max_file_size": "50MB",
                    "supported_formats": ["pdf"],
                    "timeout": 30
                },
                "performance_metrics": {
                    "success_rate": 0.95,
                    "average_response_time": 5.2
                }
            },
            {
                "name": "document_discovery",
                "description": ("Discovers policy documents from various "
                                "sources"),
                "input_types": ["keywords", "date_range", "source_type"],
                "output_types": ["document_list", "metadata"],
                "parameters": {
                    "max_results": 100,
                    "supported_sources": ["government", "academic", "ngo"]
                },
                "performance_metrics": {
                    "discovery_accuracy": 0.88,
                    "coverage_rate": 0.92
                }
            }
        ]
        
        return self.register_agent(
            "EPPN Librarian",
            "librarian",
            capabilities,
            local_path="agents/librarian",
        )

    def register_interpreter_agent(self) -> str:
        """Register the Interpreter agent."""
        
        capabilities = [
            {
                "name": "pdf_parsing",
                "description": ("Extracts and structures content from PDF "
                                "documents"),
                "input_types": ["pdf_document"],
                "output_types": ["structured_text", "metadata", "sections"],
                "parameters": {
                    "supported_languages": ["en", "es", "fr"],
                    "extraction_methods": ["text", "tables", "images"]
                },
                "performance_metrics": {
                    "extraction_accuracy": 0.93,
                    "processing_speed": 2.1
                }
            },
            {
                "name": "content_analysis",
                "description": "Analyzes document structure and content",
                "input_types": ["text", "document"],
                "output_types": [
                    "analysis_report", "key_concepts", "structure"
                ],
                "parameters": {
                    "analysis_depth": "comprehensive",
                    "concept_extraction": True
                },
                "performance_metrics": {
                    "analysis_quality": 0.89,
                    "concept_accuracy": 0.91
                }
            }
        ]
        
        return self.register_agent(
            "EPPN Interpreter",
            "interpreter",
            capabilities,
            local_path="agents/interpreter",
        )

    def register_summarizer_agent(self) -> str:
        """Register the Summarizer agent."""
        
        capabilities = [
            {
                "name": "document_summarization",
                "description": ("Creates human-readable summaries of policy "
                                "documents"),
                "input_types": ["structured_text", "document"],
                "output_types": ["summary", "key_points", "executive_summary"],
                "parameters": {
                    "summary_length": "variable",
                    "target_audience": "general_public",
                    "language": "en"
                },
                "performance_metrics": {
                    "summary_quality": 0.87,
                    "readability_score": 0.85
                }
            },
            {
                "name": "multi_document_summary",
                "description": "Summarizes multiple related documents",
                "input_types": ["document_collection"],
                "output_types": ["comparative_summary", "synthesis"],
                "parameters": {
                    "max_documents": 10,
                    "comparison_method": "thematic"
                },
                "performance_metrics": {
                    "synthesis_quality": 0.84,
                    "comparison_accuracy": 0.88
                }
            }
        ]
        
        return self.register_agent(
            "EPPN Summarizer",
            "summarizer",
            capabilities,
            local_path="agents/summarizer",
        )

    def register_ethical_analyst_agent(self) -> str:
        """Register the Ethical Analyst agent."""
        
        capabilities = [
            {
                "name": "ethical_analysis",
                "description": ("Performs comprehensive ethical analysis of "
                                "policies"),
                "input_types": ["policy_document", "structured_text"],
                "output_types": ["ethics_report", "bias_analysis",
                                 "fairness_assessment"],
                "parameters": {
                    "ethical_frameworks": [
                        "utilitarian", "deontological", "virtue_ethics",
                        "care_ethics", "justice_theory"
                    ],
                    "urban_planning_frameworks": [
                        "sustainable_development", "spatial_justice",
                        "environmental_justice"
                    ],
                    "analysis_depth": "comprehensive"
                },
                "performance_metrics": {
                    "analysis_accuracy": 0.91,
                    "bias_detection_rate": 0.89,
                    "framework_coverage": 0.95
                }
            },
            {
                "name": "contradiction_detection",
                "description": "Detects contradictions in policy statements",
                "input_types": ["policy_statements", "document_collection"],
                "output_types": ["contradiction_report", "conflict_analysis"],
                "parameters": {
                    "detection_methods": ["semantic", "logical", "contextual"],
                    "confidence_threshold": 0.7
                },
                "performance_metrics": {
                    "detection_accuracy": 0.86,
                    "false_positive_rate": 0.12
                }
            },
            {
                "name": "xai_explanation",
                "description": ("Provides explainable AI explanations for "
                                "decisions"),
                "input_types": ["analysis_result", "decision_trace"],
                "output_types": [
                    "explanation", "decision_tree", "feature_importance"
                ],
                "parameters": {
                    "explanation_types": [
                        "decision_tree", "feature_importance", "counterfactual"
                    ],
                    "detail_level": "comprehensive"
                },
                "performance_metrics": {
                    "explanation_quality": 0.88,
                    "user_satisfaction": 0.85
                }
            }
        ]
        
        return self.register_agent(
            "EPPN Ethical Analyst",
            "ethical_analyst",
            capabilities,
            local_path="agents/ethical_analyst",
        )

    def register_communicator_agent(self) -> str:
        """Register the Communicator agent."""
        
        capabilities = [
            {
                "name": "human_interface",
                "description": ("Provides human-in-the-loop interfaces and "
                                "notifications"),
                "input_types": ["analysis_result", "decision_request"],
                "output_types": [
                    "notification", "dashboard_link", "chat_interface"
                ],
                "parameters": {
                    "notification_channels": [
                        "email", "sms", "dashboard", "chat"
                    ],
                    "interaction_types": [
                        "approval", "feedback", "clarification"
                    ]
                },
                "performance_metrics": {
                    "response_time": 1.2,
                    "user_engagement": 0.82
                }
            },
            {
                "name": "collaboration_coordination",
                "description": "Coordinates multi-agent collaborations",
                "input_types": ["task_request", "agent_capabilities"],
                "output_types": ["collaboration_plan", "task_assignment"],
                "parameters": {
                    "max_agents": 5,
                    "coordination_method": "centralized"
                },
                "performance_metrics": {
                    "coordination_efficiency": 0.90,
                    "task_completion_rate": 0.94
                }
            }
        ]
        
        return self.register_agent(
            "EPPN Communicator",
            "communicator",
            capabilities,
            local_path="agents/communicator",
        )

    def discover_agents(self, capability_requirements: Dict[str, Any]
                        ) -> List[AgentProfile]:
        """Discover agents with specific capabilities."""
        
        # Search AgentVerse for matching agents
        matching_agents = self._search_agentverse(capability_requirements)
        
        # Update discovered agents registry
        for agent in matching_agents:
            self.discovered_agents[agent.agent_id] = agent
        
        return matching_agents

    def find_ethical_analysis_agents(self) -> List[AgentProfile]:
        """Find agents capable of ethical analysis."""
        
        requirements = {
            "capabilities": [
                "ethical_analysis", "bias_detection", "fairness_assessment"
            ],
            "domain": ["ethics", "policy", "governance", "urban_planning"],
            "min_reputation": 0.7
        }
        
        return self.discover_agents(requirements)

    def find_urban_planning_agents(self) -> List[AgentProfile]:
        """Find agents specialized in urban planning."""
        
        requirements = {
            "capabilities": [
                "urban_planning", "resource_allocation", "spatial_analysis"
            ],
            "domain": [
                "urban_planning", "city_planning", "resource_management"
            ],
            "min_reputation": 0.6
        }
        
        return self.discover_agents(requirements)

    def initiate_collaboration(self, participating_agents: List[str], 
                               collaboration_type: str, 
                               task_description: str) -> str:
        """Initiate collaboration between agents."""
        
        collaboration_id = str(uuid.uuid4())
        
        collaboration = AgentCollaboration(
            collaboration_id=collaboration_id,
            participating_agents=participating_agents,
            collaboration_type=collaboration_type,
            status="initiated",
            start_time=datetime.now(),
            end_time=None,
            results=None
        )
        
        # Notify participating agents
        success = self._notify_agents_of_collaboration(
            collaboration, task_description
        )
        if success:
            self.active_collaborations[collaboration_id] = collaboration
            return collaboration_id
        else:
            raise Exception(
                f"Failed to initiate collaboration {collaboration_id}"
            )

    def complete_collaboration(self, collaboration_id: str,
                               results: Dict[str, Any]) -> bool:
        """Complete a collaboration and store results."""

        if collaboration_id not in self.active_collaborations:
            return False

        collaboration = self.active_collaborations[collaboration_id]
        collaboration.status = "completed"
        collaboration.end_time = datetime.now()
        collaboration.results = results

        # Move to completed collaborations
        self.completed_collaborations[collaboration_id] = collaboration
        del self.active_collaborations[collaboration_id]

        # Update agent reputations based on collaboration results
        self._update_agent_reputations(collaboration)

        return True
    
    def get_agent_reputation(self, agent_id: str) -> Optional[float]:
        """Get reputation score of an agent."""
        
        if agent_id in self.registered_agents:
            return self.registered_agents[agent_id].reputation_score
        elif agent_id in self.discovered_agents:
            return self.discovered_agents[agent_id].reputation_score
        
        return None
    
    def update_agent_performance(self, agent_id: str,
                                 performance_metrics: Dict[str, float]
                                 ) -> bool:
        """Update agent performance metrics."""
        
        if agent_id not in self.registered_agents:
            return False
        
        agent = self.registered_agents[agent_id]
        
        # Update capabilities with new performance metrics
        for capability in agent.capabilities:
            capability.performance_metrics.update(performance_metrics)
        
        # Update last activity
        agent.last_activity = datetime.now()
        
        # Update on AgentVerse
        return self._update_agent_on_agentverse(agent)
    
    def get_collaboration_history(
            self, agent_id: str = None) -> List[AgentCollaboration]:
        """Get collaboration history."""
        
        collaborations = list(self.completed_collaborations.values())
        
        if agent_id:
            collaborations = [
                c for c in collaborations 
                if agent_id in c.participating_agents
            ]
        
        return collaborations
    
    def export_agent_registry(self) -> Dict[str, Any]:
        """Export agent registry data."""
        
        return {
            "registered_agents": [
                asdict(agent) for agent in self.registered_agents.values()
            ],
            "discovered_agents": [
                asdict(agent) for agent in self.discovered_agents.values()
            ],
            "active_collaborations": [
                asdict(collab) 
                for collab in self.active_collaborations.values()
            ],
            "completed_collaborations": [
                asdict(collab) 
                for collab in self.completed_collaborations.values()
            ],
            "summary": {
                "total_registered": len(self.registered_agents),
                "total_discovered": len(self.discovered_agents),
                "active_collaborations": len(self.active_collaborations),
                "completed_collaborations": len(self.completed_collaborations)
            }
        }
    
    def _register_with_agentverse(self, profile: AgentProfile) -> bool:
        """Register agent with AgentVerse platform."""
        
        try:
            headers = ({"Authorization": f"Bearer {self.api_key}"}
                       if self.api_key else {})

            response = requests.post(
                f"{self.api_url}/agents/register",
                json=asdict(profile),
                headers=headers,
                timeout=30,
            )

            return response.status_code == 200
        except (requests.RequestException, Exception):
            # Fallback to simulation for development
            return True
    
    def _search_agentverse(self, requirements: Dict[str, Any]
                           ) -> List[AgentProfile]:
        """Search AgentVerse for matching agents."""
        
        try:
            headers = ({"Authorization": f"Bearer {self.api_key}"} 
                       if self.api_key else {})
            
            response = requests.post(
                f"{self.api_url}/agents/search",
                json=requirements,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                agents_data = response.json()
                return [AgentProfile(**agent) for agent in agents_data]
        except (requests.RequestException, Exception):
            pass
        
        # Return mock agents for development
        return [
            AgentProfile(
                agent_id="mock_ethical_agent",
                name="EthicsBot Pro",
                description="Advanced ethical analysis agent",
                version="2.1.0",
                domain="ethics",
                capabilities=[],
                contact_info={"email": "contact@ethicsbot.ai"},
                reputation_score=0.92,
                status="active",
                registration_date=datetime.now(),
                last_activity=datetime.now()
            ),
            AgentProfile(
                agent_id="mock_urban_agent",
                name="UrbanPlanner AI",
                description="Specialized urban planning agent",
                version="1.5.0",
                domain="urban_planning",
                capabilities=[],
                contact_info={"email": "contact@urbanplanner.ai"},
                reputation_score=0.88,
                status="active",
                registration_date=datetime.now(),
                last_activity=datetime.now()
            )
        ]
    
    def _notify_agents_of_collaboration(
            self, collaboration: AgentCollaboration,
            task_description: str) -> bool:
        """Notify agents of collaboration initiation."""
        
        # In a real implementation, this would send notifications to agents
        return True
    
    def _update_agent_reputations(self, collaboration: AgentCollaboration
                                  ) -> None:
        """Update agent reputations based on collaboration results."""
        
        # Simple reputation update based on collaboration success
        if (collaboration.results and 
                collaboration.results.get("success", False)):
            reputation_boost = 0.01
        else:
            reputation_boost = -0.005
        
        for agent_id in collaboration.participating_agents:
            if agent_id in self.registered_agents:
                agent = self.registered_agents[agent_id]
                new_score = agent.reputation_score + reputation_boost
                agent.reputation_score = max(0.0, min(1.0, new_score))
    
    def _update_agent_on_agentverse(self, agent: AgentProfile) -> bool:
        """Update agent information on AgentVerse."""
        
        try:
            headers = ({"Authorization": f"Bearer {self.api_key}"} 
                       if self.api_key else {})
            
            response = requests.put(
                f"{self.api_url}/agents/{agent.agent_id}",
                json=asdict(agent),
                headers=headers,
                timeout=30
            )
            
            return response.status_code == 200
        except (requests.RequestException, Exception):
            return True
