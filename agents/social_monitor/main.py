"""
Social Monitor uAgent

Polls RSS/social feeds (via feedparser) to discover new policy links and
triggers CrawlRequest to the orchestrator.
"""

from datetime import timedelta
from typing import List

import feedparser
from uagents import Agent, Context

from schemas.messages import CrawlRequest


agent = Agent(name="social_monitor", seed="eppn-social-monitor")


SOURCES: List[str] = [
    # Example RSS feeds; replace with official Kenyan gov comms feeds where available
    "https://www.kenyanews.go.ke/feed/",
    "https://www.capitalfm.co.ke/news/feed/",
]


@agent.on_interval(period=timedelta(hours=6))
async def poll_feeds(ctx: Context):
    urls: List[str] = []
    for src in SOURCES:
        try:
            feed = feedparser.parse(src)
            for entry in feed.entries[:10]:
                link = getattr(entry, "link", "")
                if link and ("policy" in link or "gazette" in link or "budget" in link):
                    urls.append(link)
        except Exception as e:
            ctx.logger.error(f"Feed parse error {src}: {e}")

    if urls:
        await ctx.send("orchestrator", CrawlRequest(urls=list(set(urls))))


if __name__ == "__main__":
    agent.run()


