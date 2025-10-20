"""
CLI: Send CrawlRequest to orchestrator with given URLs.
Usage:
  python tools/cli_crawl.py URL1 URL2 ...
"""

import sys
from typing import List

from uagents import Agent, Context

from schemas.messages import CrawlRequest


agent = Agent(name="cli_crawl", seed="eppn-cli-crawl")


@agent.on_event("startup")
async def on_start(ctx: Context):
    urls: List[str] = sys.argv[1:]
    if not urls:
        ctx.logger.error("Provide at least one URL")
        return
    req = CrawlRequest(urls=urls)
    await ctx.send("orchestrator", req)
    ctx.logger.info(f"Sent CrawlRequest for {len(urls)} URLs")


if __name__ == "__main__":
    agent.run()


