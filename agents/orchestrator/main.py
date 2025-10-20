"""
Orchestrator uAgent

Receives messages and routes them to the appropriate next agent in the pipeline.
"""

from uagents import Agent, Context

from schemas.messages import CrawlRequest, PDFReady, ParsedText, SummaryReady, EthicsReport


agent = Agent(name="orchestrator", seed="eppn-orchestrator")


# In a real deployment, these could be loaded from env/config or registry
INTERPRETER_ADDR = "interpreter"
SUMMARIZER_ADDR = "summarizer"
ETHICS_ADDR = "ethical_analyst"
COMMUNICATOR_ADDR = "communicator"


@agent.on_message(model=CrawlRequest)
async def handle_crawl(ctx: Context, msg: CrawlRequest):
    await ctx.send("librarian", msg)


@agent.on_message(model=PDFReady)
async def route_pdf(ctx: Context, msg: PDFReady):
    await ctx.send(INTERPRETER_ADDR, msg)


@agent.on_message(model=ParsedText)
async def route_parsed(ctx: Context, msg: ParsedText):
    await ctx.send(SUMMARIZER_ADDR, msg)
    await ctx.send(ETHICS_ADDR, msg)


@agent.on_message(model=SummaryReady)
async def route_summary(ctx: Context, msg: SummaryReady):
    await ctx.send(COMMUNICATOR_ADDR, msg)


@agent.on_message(model=EthicsReport)
async def route_ethics(ctx: Context, msg: EthicsReport):
    await ctx.send(COMMUNICATOR_ADDR, msg)


if __name__ == "__main__":
    agent.run()


