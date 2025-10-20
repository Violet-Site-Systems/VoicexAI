a"""
Interpreter uAgent

Extracts text and structure from PDFs and emits ParsedText.
"""

import os
from typing import List, Dict, Any

from PyPDF2 import PdfReader
from uagents import Agent, Context

from schemas.messages import PDFReady, ParsedText, SummaryReady


agent = Agent(
    name="interpreter",
    seed="eppn-interpreter",
)


def extract_text_sections(pdf_path: str) -> List[Dict[str, Any]]:
    reader = PdfReader(pdf_path)
    sections: List[Dict[str, Any]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        sections.append({"title": f"Page {i+1}", "text": text, "page": i+1})
    return sections


@agent.on_message(model=PDFReady)
async def handle_pdf(ctx: Context, msg: PDFReady):
    doc_id = os.path.basename(msg.source)
    sections = extract_text_sections(msg.source)

    parsed = ParsedText(
        doc_id=doc_id,
        sections=sections,
        entities=[],
        metadata={"url": msg.url}
    )

    await ctx.send(ctx.sender, parsed)


if __name__ == "__main__":
    agent.run()


