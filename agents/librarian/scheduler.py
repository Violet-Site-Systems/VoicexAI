"""
Librarian Scheduler

Loads Kenya policy sources and periodically sends CrawlRequest messages.
"""

import yaml
from datetime import timedelta
from typing import List

from uagents import Agent, Context
from uagents.setup import fund_agent_if_low

from schemas.messages import CrawlRequest


agent = Agent(name="librarian_scheduler", seed="eppn-librarian-scheduler")


def load_sources(path: str = "config/sources.yaml") -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    urls: List[str] = []
    for src in data.get("kenya_policy_sources", []):
        urls.extend(src.get("urls", []))
    return urls


@agent.on_interval(period=timedelta(hours=12))
async def scheduled_crawl(ctx: Context):
    urls = load_sources()
    if not urls:
        ctx.logger.warning("No sources loaded for scheduled crawl")
        return
    req = CrawlRequest(urls=urls)
    await ctx.send(ctx.sender, req)


if __name__ == "__main__":
    fund_agent_if_low(agent.wallet.address())
    agent.run()


