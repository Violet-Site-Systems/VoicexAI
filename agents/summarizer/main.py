"""
Summarizer uAgent

Summarizes ParsedText into concise summaries and key points.
"""

from typing import List

from uagents import Agent, Context

from schemas.messages import ParsedText, SummaryReady


agent = Agent(
    name="summarizer",
    seed="eppn-summarizer",
)


def simple_summarize(texts: List[str]) -> str:
    text = "\n".join(texts)
    return (text[:800] + "...") if len(text) > 800 else text


@agent.on_message(model=ParsedText)
async def handle_parsed(ctx: Context, msg: ParsedText):
    texts = [s.get("text", "") for s in msg.sections]
    summary = simple_summarize(texts)
    key_points = [s.get("title", "") for s in msg.sections[:5]]

    out = SummaryReady(
        doc_id=msg.doc_id,
        summary=summary,
        key_points=key_points,
        metadata={"source": "summarizer"}
    )

    await ctx.send(ctx.sender, out)


if __name__ == "__main__":
    agent.run()


