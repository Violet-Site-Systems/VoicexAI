"""
Communicator uAgent

Interfaces with human ethics partner by logging summaries and ethics reports.
In a real deployment, this would back a dashboard/API.
"""

import json
import os
from uagents import Agent, Context

from schemas.messages import SummaryReady, EthicsReport

DATA_DIR = "data"
SUMMARY_FILE = os.path.join(DATA_DIR, "summaries.jsonl")
ETHICS_FILE = os.path.join(DATA_DIR, "ethics.jsonl")
os.makedirs(DATA_DIR, exist_ok=True)


agent = Agent(
    name="communicator",
    seed="eppn-communicator",
)


@agent.on_message(model=SummaryReady)
async def handle_summary(ctx: Context, msg: SummaryReady):
    ctx.logger.info(f"Summary for {msg.doc_id}: {msg.key_points[:3]}")
    with open(SUMMARY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(msg.__dict__) + "\n")


@agent.on_message(model=EthicsReport)
async def handle_ethics(ctx: Context, msg: EthicsReport):
    ctx.logger.info(f"Ethics report for {msg.doc_id}: {json.dumps(msg.report.get('summary', {}))}")
    with open(ETHICS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "doc_id": msg.doc_id,
            "report": msg.report,
            "risks": msg.risks,
            "recommendations": msg.recommendations,
            "metadata": msg.metadata
        }) + "\n")


if __name__ == "__main__":
    agent.run()


