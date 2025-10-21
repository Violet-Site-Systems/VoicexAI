"""
EPPN Integration Manager

Orchestrates all integrations: Cudos, AgentVerse, XAI, Human-in-the-Loop,
and Urban Planning Ethics.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from cognitive_core.atomspace.atomspace_manager import AtomSpaceManager
from cognitive_core.reasoning.ethical_analyzer import EthicalAnalyzer
from cognitive_core.reasoning.xai_explainer import XAIExplainer
from cognitive_core.human_in_loop import HumanInTheLoop
from cudos_integration import CudosIntegration
from agentverse_integration import AgentVerseIntegration


class EPPNIntegrationManager:
    """
    Main integration manager for EPPN system.
    Coordinates:
    - Urban planning ethics analysis
    - XAI explanations
    - Human-in-the-loop interactions
    - Cudos blockchain integration
    - AgentVerse agent registration and collaboration
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the integration manager."""
        self.config = config

        # Initialize core components
        self.atomspace_manager = AtomSpaceManager()
        self.ethical_analyzer = EthicalAnalyzer(self.atomspace_manager)
        self.xai_explainer = XAIExplainer(self.ethical_analyzer.reasoner)
        self.human_in_loop = HumanInTheLoop(
            self.ethical_analyzer,
            self.xai_explainer,
            config.get("base_url", "http://localhost:8000")
        )

        # Initialize external integrations
        self.cudos_integration = CudosIntegration(
            cudos_rpc_url=config.get("cudos_rpc_url", "https://rpc.cudos.org"),
            cudos_rest_url=config.get(
                "cudos_rest_url", "https://rest.cudos.org"
            ),
            wallet_address=config.get("cudos_wallet_address"),
            private_key=config.get("cudos_private_key")
        )

        self.agentverse_integration = AgentVerseIntegration(
            agentverse_api_url=config.get(
                "agentverse_api_url", "https://api.agentverse.ai"
            ),
            api_key=config.get("agentverse_api_key")
            )

        # Optional mapping role -> public endpoint (for AgentVerse to call)
        # Example: {"librarian": "https://mydomain.com/agents/librarian"}
        self.agent_endpoints: Dict[str, str] = config.get("agent_endpoints", {})

        # Agent registry
        self.registered_agents: Dict[str, str] = {}

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def initialize_system(self) -> bool:
        """Initialize the entire EPPN system."""
        try:
            self.logger.info("Initializing EPPN system...")

            # Register agents on AgentVerse
            await self._register_all_agents()

            # Initialize Cudos integration
            await self._initialize_cudos()

            # Setup human-in-the-loop workflows
            await self._setup_human_workflows()

            self.logger.info("EPPN system initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize EPPN system: {e}")
            return False

    async def analyze_policy_with_full_pipeline(
        self, policy_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run complete policy analysis pipeline with all integrations."""

        try:
            self.logger.info("Starting comprehensive policy analysis...")

            # Step 1: Convert policy to atoms
            policy_atoms = await self._convert_policy_to_atoms(policy_data)

            # Step 2: Perform ethical analysis
            ethical_report = self.ethical_analyzer.analyze_policy(policy_atoms)

            # Step 3: Generate XAI explanations
            xai_explanations = await self._generate_xai_explanations(
                ethical_report, policy_atoms
            )

            # Step 4: Check if human approval is needed
            human_interaction = None
            if self._requires_human_approval(ethical_report):
                human_interaction = await self._request_human_approval(
                    ethical_report, policy_atoms
                )

            # Step 5: Submit to Cudos for distributed analysis
            cudos_tasks = await self._submit_to_cudos_distributed_analysis(
                policy_data
            )

            # Step 6: Coordinate with other agents via AgentVerse
            agent_collaboration = await self._coordinate_agent_collaboration(
                policy_data
            )

            # Step 7: Aggregate results
            final_results = await self._aggregate_all_results(
                ethical_report, xai_explanations, human_interaction,
                cudos_tasks, agent_collaboration
            )

            self.logger.info("Policy analysis completed successfully")
            return final_results

        except Exception as e:
            self.logger.error(f"Policy analysis failed: {e}")
            return {"error": str(e), "status": "failed"}

    async def _register_all_agents(self) -> None:
        """Register all EPPN agents on AgentVerse."""
        self.logger.info("Registering agents on AgentVerse...")
        # Register each agent type

        # When available, pass configured public endpoints to registration
        self.registered_agents["librarian"] = (
            self.agentverse_integration.register_librarian_agent(
                endpoint=self.agent_endpoints.get("librarian")
            )
        )
        self.registered_agents["interpreter"] = (
            self.agentverse_integration.register_interpreter_agent(
                endpoint=self.agent_endpoints.get("interpreter")
            )
        )
        self.registered_agents["summarizer"] = (
            self.agentverse_integration.register_summarizer_agent(
                endpoint=self.agent_endpoints.get("summarizer")
            )
        )
        self.registered_agents["ethical_analyst"] = (
            self.agentverse_integration.register_ethical_analyst_agent(
                endpoint=self.agent_endpoints.get("ethical_analyst")
            )
        )
        self.registered_agents["communicator"] = (
            self.agentverse_integration.register_communicator_agent(
                endpoint=self.agent_endpoints.get("communicator")
            )
        )

        self.logger.info(f"Registered agents: {self.registered_agents}")

    async def _initialize_cudos(self) -> None:
        """Initialize Cudos blockchain integration."""

        self.logger.info("Initializing Cudos integration...")

        # Get available nodes
        nodes = self.cudos_integration.get_available_nodes()
        self.logger.info(f"Found {len(nodes)} available Cudos nodes")

        # Check staking status
        staking_info = self.cudos_integration.get_staking_info()
        self.logger.info(f"Staking info: {staking_info}")

    async def _setup_human_workflows(self) -> None:
        """Setup human-in-the-loop workflows."""

        self.logger.info("Setting up human-in-the-loop workflows...")

        # Configure interaction callbacks
        self.human_in_loop.interaction_callbacks = {
            self.human_in_loop.HumanInteractionType.APPROVAL: (
                self._handle_approval_response
            ),
            self.human_in_loop.HumanInteractionType.FEEDBACK: (
                self._handle_feedback_response
            ),
            self.human_in_loop.HumanInteractionType.CLARIFICATION: (
                self._handle_clarification_response
            ),
            self.human_in_loop.HumanInteractionType.ESCALATION: (
                self._handle_escalation_response
            )
        }

    async def _convert_policy_to_atoms(
        self, policy_data: Dict[str, Any]
    ) -> List:
        """Convert policy data to atoms for analysis."""

        # This would typically involve parsing the policy document
        # and creating atoms for each significant statement

        atoms = []
        # Implementation would depend on the specific policy format
        # For now, we'll create mock atoms

        return atoms

    async def _generate_xai_explanations(self, ethical_report: Dict[str, Any],
                                         policy_atoms: List) -> Dict[str, Any]:
        """Generate XAI explanations for the analysis."""

        explanations = {}

        # Generate explanations for different aspects
        if "contradictions" in ethical_report:
            for contradiction in ethical_report["contradictions"]:
                explanation = (
                    self.xai_explainer.explain_contradiction_detection(
                        contradiction
                    )
                )
                if explanation:
                    key = f"contradiction_{contradiction.get('id', 'unknown')}"
                    explanations[key] = explanation

        if "fairness" in ethical_report:
            for fairness_issue in ethical_report["fairness"]:
                explanation = self.xai_explainer.explain_bias_detection(
                    fairness_issue
                )
                if explanation:
                    key = f"bias_{fairness_issue.get('id', 'unknown')}"
                    explanations[key] = explanation

        if "ethical_implications" in ethical_report:
            for implication in ethical_report["ethical_implications"]:
                explanation = self.xai_explainer.explain_ethical_implication(
                    implication
                )
                if explanation:
                    key = f"ethics_{implication.get('id', 'unknown')}"
                    explanations[key] = explanation

        return explanations

    def _requires_human_approval(self, ethical_report: Dict[str, Any]) -> bool:
        """Determine if human approval is required."""

        # Check for high-risk indicators
        if ethical_report.get("summary", {}).get("contradictions", 0) > 3:
            return True

        if (ethical_report.get("summary", {})
                .get("urban_planning_ethics", 0) > 5):
            return True

        resource_issues = ethical_report.get("summary", {}).get(
            "resource_allocation_issues", 0
        )
        if resource_issues > 2:
            return True

        return False

    async def _request_human_approval(self, ethical_report: Dict[str, Any],
                                      policy_atoms: List) -> Optional[Any]:
        """Request human approval for the analysis."""

        # Create a decision trace for human review
        decision_trace = self.xai_explainer.DecisionTrace(
            decision_id=f"policy_analysis_{datetime.now().isoformat()}",
            input_atoms=[atom.uuid for atom in policy_atoms],
            reasoning_steps=ethical_report.get("reasoning_steps", []),
            final_conclusion=("Policy analysis completed with ethical "
                              "implications identified"),
            confidence_score=0.7
        )

        # Request human approval
        interaction = self.human_in_loop.request_human_approval(
            decision_trace, {
                "ethical_report": ethical_report,
                "policy_type": "urban_planning",
                "risk_level": "medium"
            }
        )

        return interaction

    async def _submit_to_cudos_distributed_analysis(
        self, policy_data: Dict[str, Any]
    ) -> List:
        """Submit policy analysis to Cudos for distributed processing."""

        self.logger.info("Submitting to Cudos for distributed analysis...")

        # Submit ethical analysis task
        ethical_task = self.cudos_integration.submit_ethical_analysis_task(
            policy_data, "comprehensive"
        )

        # Submit data storage task
        storage_task = self.cudos_integration.submit_data_storage_task(
            policy_data, "policy_analysis"
        )

        return [ethical_task, storage_task]

    async def _coordinate_agent_collaboration(
        self, policy_data: Dict[str, Any]
    ) -> Optional[str]:
        """Coordinate collaboration with other agents via AgentVerse."""

        self.logger.info("Coordinating agent collaboration...")

        # Find other ethical analysis agents
        ethical_agents = (
            self.agentverse_integration.find_ethical_analysis_agents()
        )
        urban_planning_agents = (
            self.agentverse_integration.find_urban_planning_agents()
        )

        if ethical_agents or urban_planning_agents:
            # Initiate collaboration
            participating_agents = [self.registered_agents["ethical_analyst"]]

            if ethical_agents:
                participating_agents.extend([
                    agent.agent_id for agent in ethical_agents[:2]
                ])

            if urban_planning_agents:
                participating_agents.extend([
                    agent.agent_id for agent in urban_planning_agents[:2]
                ])

            collaboration_id = (
                self.agentverse_integration.initiate_collaboration(
                    participating_agents=participating_agents,
                    collaboration_type="distributed_ethical_analysis",
                    task_description=("Analyze urban planning policy for "
                                      "ethical implications and resource "
                                      "allocation fairness")
                )
            )

            return collaboration_id

        return None

    async def _aggregate_all_results(self, ethical_report: Dict[str, Any],
                                     xai_explanations: Dict[str, Any],
                                     human_interaction: Optional[Any],
                                     cudos_tasks: List,
                                     agent_collaboration: Optional[str]
                                     ) -> Dict[str, Any]:
        """Aggregate results from all analysis components."""

        # Collect Cudos results
        cudos_results = []
        for task in cudos_tasks:
            result = self.cudos_integration.retrieve_task_result(task.task_id)
            if result:
                cudos_results.append(result)

        # Get agent collaboration results
        collaboration_results = None
        if agent_collaboration:
            collaboration_results = (
                self.agentverse_integration.get_collaboration_history(
                    agent_collaboration
                )
            )

        # Aggregate everything
        final_results = {
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "ethical_analysis": ethical_report,
            "xai_explanations": xai_explanations,
            "human_interaction": (
                human_interaction.__dict__ if human_interaction else None
            ),
            "cudos_results": cudos_results,
            "agent_collaboration": collaboration_results,
            "summary": {
                "total_contradictions": (
                    ethical_report.get("summary", {}).get("contradictions", 0)
                ),
                "total_fairness_issues": (
                    ethical_report.get("summary", {})
                    .get("fairness_signals", 0)
                ),
                "urban_planning_ethics_issues": (
                    ethical_report.get("summary", {})
                    .get("urban_planning_ethics", 0)
                ),
                "resource_allocation_issues": (
                    ethical_report.get("summary", {})
                    .get("resource_allocation_issues", 0)
                ),
                "human_approval_required": human_interaction is not None,
                "distributed_analysis_tasks": len(cudos_tasks),
                "agent_collaborations": 1 if agent_collaboration else 0
            }
        }

        return final_results

    async def _handle_approval_response(
        self, interaction: Any, response: Dict[str, Any]
    ) -> bool:
        """Handle human approval response."""

        self.logger.info(f"Received approval response: {response}")

        if response.get("action") == "approve":
            self.logger.info("Human approved the analysis")
            return True
        elif response.get("action") == "reject":
            self.logger.info("Human rejected the analysis")
            return False
        elif response.get("action") == "modify":
            self.logger.info("Human requested modifications")
            return False
        elif response.get("action") == "escalate":
            self.logger.info("Human escalated the decision")
            return False

        return False

    async def _handle_feedback_response(
        self, interaction: Any, response: Dict[str, Any]
    ) -> bool:
        """Handle human feedback response."""

        self.logger.info(f"Received feedback: {response}")

        # Store feedback for learning
        feedback = self.human_in_loop.HumanFeedback(
            feedback_id=f"feedback_{datetime.now().isoformat()}",
            decision_id=interaction.context.get("decision_trace", {}).get(
                "decision_id", ""
            ),
            feedback_type="general",
            rating=response.get("rating", 3),
            comments=response.get("comments", ""),
            suggested_improvements=response.get("suggested_improvements", []),
            timestamp=datetime.now()
        )

        self.human_in_loop.feedback_history.append(feedback)

        return True

    async def _handle_clarification_response(
        self, interaction: Any, response: Dict[str, Any]
    ) -> bool:
        """Handle human clarification response."""

        self.logger.info(f"Received clarification: {response}")

        # Use clarification to improve analysis
        clarification_data = response.get("clarification", "")

        # This would typically update the analysis with the clarification
        # For now, we'll just log it
        self.logger.info(f"Processing clarification: {clarification_data}")

        return True

    async def _handle_escalation_response(
        self, interaction: Any, response: Dict[str, Any]
    ) -> bool:
        """Handle human escalation response."""

        self.logger.info(f"Received escalation response: {response}")

        # Escalation response provides final authority decision
        return True

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""

        status = {
            "timestamp": datetime.now().isoformat(),
            "system_health": "healthy",
            "components": {
                "cognitive_core": "active",
                "xai_explainer": "active",
                "human_in_loop": "active",
                "cudos_integration": "active",
                "agentverse_integration": "active"
            },
            "agents": {
                "registered_agents": len(self.registered_agents),
                "agent_ids": self.registered_agents
            },
            "cudos": {
                "available_nodes": len(
                    self.cudos_integration.get_available_nodes()
                ),
                "submitted_tasks": len(self.cudos_integration.submitted_tasks),
                "completed_tasks": len(self.cudos_integration.completed_tasks)
            },
            "agentverse": {
                "discovered_agents": len(
                    self.agentverse_integration.discovered_agents
                ),
                "active_collaborations": len(
                    self.agentverse_integration.active_collaborations
                ),
                "completed_collaborations": len(
                    self.agentverse_integration.completed_collaborations
                )
            },
            "human_interactions": {
                "pending_interactions": len(
                    self.human_in_loop.pending_interactions
                ),
                "completed_interactions": len(
                    self.human_in_loop.completed_interactions
                ),
                "total_feedback": len(self.human_in_loop.feedback_history)
            }
        }

        return status

    async def shutdown(self) -> None:
        """Gracefully shutdown the system."""

        self.logger.info("Shutting down EPPN system...")

        # Complete any pending human interactions
        for interaction in self.human_in_loop.pending_interactions.values():
            interaction.status = self.human_in_loop.InteractionStatus.CANCELLED

        # Complete any active collaborations
        for collaboration in (
            self.agentverse_integration.active_collaborations.values()
        ):
            collaboration.status = "cancelled"
            collaboration.end_time = datetime.now()

        self.logger.info("EPPN system shutdown complete")


# Example usage
async def main():
    """Example usage of the EPPN Integration Manager."""

    config = {
        "base_url": "http://localhost:8000",
        "cudos_rpc_url": "https://rpc.cudos.org",
        "cudos_rest_url": "https://rest.cudos.org",
        "cudos_wallet_address": "your_wallet_address",
        "cudos_private_key": "your_private_key",
        "agentverse_api_url": "https://api.agentverse.ai",
        "agentverse_api_key": "your_api_key"
    }

    # Initialize the system
    manager = EPPNIntegrationManager(config)

    try:
        # Initialize the system
        await manager.initialize_system()

        # Example policy analysis
        policy_data = {
            "title": "Urban Development Policy 2024",
            "content": ("This policy outlines the development of affordable "
                        "housing in urban areas..."),
            "type": "urban_planning",
            "domain": "resource_allocation"
        }

        # Run comprehensive analysis
        results = await manager.analyze_policy_with_full_pipeline(policy_data)

        print("Analysis Results:")
        print(f"Status: {results['status']}")
        print(f"Contradictions: {results['summary']['total_contradictions']}")
        print(f"Fairness Issues: "
              f"{results['summary']['total_fairness_issues']}")
        print(f"Urban Planning Ethics Issues: "
              f"{results['summary']['urban_planning_ethics_issues']}")
        print(f"Human Approval Required: "
              f"{results['summary']['human_approval_required']}")

        # Get system status
        status = await manager.get_system_status()
        print(f"\nSystem Status: {status['system_health']}")
        print(f"Registered Agents: {status['agents']['registered_agents']}")
        print(f"Cudos Nodes: {status['cudos']['available_nodes']}")
        print(f"AgentVerse Collaborations: "
              f"{status['agentverse']['active_collaborations']}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Shutdown gracefully
        await manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
