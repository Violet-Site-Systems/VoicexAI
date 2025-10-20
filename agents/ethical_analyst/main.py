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

    report = cog.analyze_policy(atom_ids)

    ethics = EthicsReport(
        doc_id=msg.doc_id,
        report=report,
        risks=[r.get("conclusion", "risk") for r in report.get("contradictions", [])],
        recommendations=[],
        metadata={"source": "ethical_analyst"}
    )

    await ctx.send(ctx.sender, ethics)


if __name__ == "__main__":
    agent.run()


