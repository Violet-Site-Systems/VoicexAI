"""
Ethical Analyst uAgent

Consumes parsed policy content, maps to AtomSpace, runs cognitive analysis
via the CognitiveAPI, and emits an EthicsReport.
"""

import json
from typing import List, Dict, Any

from uagents import Agent, Context, Model

from cognitive_core.api.cognitive_api import CognitiveAPI
from schemas.messages import ParsedText, EthicsReport


class AnalyzeRequest(Model):
    doc_id: str
    atoms: List[Dict[str, Any]]


agent = Agent(
    name="ethical_analyst",
    seed="eppn-ethical-analyst",
)


cog = CognitiveAPI()


@agent.on_message(model=ParsedText)
async def handle_parsed(ctx: Context, msg: ParsedText):
    # Map sections to atoms
    atom_ids: List[str] = []
    for section in msg.sections:
        atom_id = cog.create_atom({
            "atom_type": "PolicySection",
            "name": section.get("title", "section"),
            "content": section,
            "metadata": {"doc_id": msg.doc_id}
        })
        atom_ids.append(atom_id)

    # Run the existing ethical analysis pipeline
    report = cog.analyze_policy(atom_ids)

    # Build a tiny MeTTa program from the atoms to run simple inference
    # Each atom becomes a fact: (policy_section <doc_id> <atom_id>)
    metta_lines = []
    for aid in atom_ids:
        metta_lines.append(f"(policy_section {msg.doc_id} {aid})")

    # Example heuristic rule: if there's any policy_section for the doc, mark doc as 'has_policy'
    metta_lines.append(f"(=> (has_policy {msg.doc_id}) (policy_section {msg.doc_id} {atom_ids[0] if atom_ids else 'none'}))")

    metta_program = "\n".join(metta_lines)
    try:
        metta_result = cog.evaluate_metta(metta_program)
    except Exception as e:  # pragma: no cover - defensive
        metta_result = {"error": str(e)}

    ethics = EthicsReport(
        doc_id=msg.doc_id,
        report=report,
        risks=[r.get("conclusion", "risk") for r in report.get("contradictions", [])],
        recommendations=[],
        metadata={"source": "ethical_analyst", "metta": metta_result}
    )

    await ctx.send(ctx.sender, ethics)


if __name__ == "__main__":
    agent.run()


