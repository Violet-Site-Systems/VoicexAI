"""
Librarian uAgent

Fetches PDFs from public URLs and emits PDFReady messages.
"""

import os
from typing import List

import requests
from uagents import Agent, Context

from schemas.messages import CrawlRequest, PDFReady


agent = Agent(
    name="librarian",
    seed="eppn-librarian",
)


def download_pdf(url: str, out_dir: str = "docs") -> str:
    os.makedirs(out_dir, exist_ok=True)
    fname = url.split("/")[-1] or "policy.pdf"
    path = os.path.join(out_dir, fname)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


@agent.on_message(model=CrawlRequest)
async def handle_crawl(ctx: Context, msg: CrawlRequest):
    for url in msg.urls:
        try:
            path = download_pdf(url)
            pdf = PDFReady(url=url, source=path, metadata=msg.metadata or {})
            target = msg.interpreter_address or ctx.sender
            await ctx.send(target, pdf)
        except Exception as e:
            ctx.logger.error(f"Failed to download {url}: {e}")


if __name__ == "__main__":
    agent.run()


