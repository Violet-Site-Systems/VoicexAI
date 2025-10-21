#!/usr/bin/env python3
"""Run a demo prompt -> response using the Cognitive Core pipeline for one PDF.

This script posts a single PDF URL to /pipeline/run_url and formats a short
response that mimics a human executive summary and recommendations.
"""
import requests
import textwrap
import sys


def run_demo(pdf_url: str):
    url = "http://127.0.0.1:8001/pipeline/run_url"
    print(f"Posting PDF to pipeline: {pdf_url}")
    r = requests.post(url, json={"urls": [pdf_url]}, timeout=300)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "ok" or not data.get("results"):
        print("Pipeline returned no results.")
        return

    result = data["results"][0]

    # Build human-friendly response
    summary_short = result.get("summary", {}).get("short", "No summary available")
    key_points = result.get("key_points", [])
    contradictions = result.get("contradictions", [])
    counts = result.get("summary", {}).get("counts", {})
    recommendations = result.get("recommendations", [])

    # Generate fallback recommendations if none provided
    if not recommendations:
        if counts.get("contradictions", 0) > 0:
            recommendations.append("Investigate and reconcile contradictory statements, focusing on the cited evidence pages.")
        if counts.get("fairness_signals", 0) > 0:
            recommendations.append("Perform a fairness audit and document impacted groups and mitigation strategies.")
        recommendations.append("Improve transparency by publishing data sources and assumptions used in the policy.")

    # Format the response
    lines = []
    lines.append("Prompt: Provide an executive summary and 3 short recommendations for this policy document.")
    lines.append("")
    lines.append("Response:")
    lines.append("")
    lines.append("Executive summary:")
    lines.append(textwrap.fill(summary_short, width=80))
    lines.append("")
    lines.append("Key points:")
    for i, kp in enumerate(key_points[:5], start=1):
        lines.append(f"  {i}. {kp}")
    lines.append("")
    lines.append(f"Contradictions found: {len(contradictions)}")
    if contradictions:
        lines.append("Example contradiction:")
        lines.append(textwrap.fill(contradictions[0], width=80))
        lines.append("")
    lines.append("Recommendations:")
    for i, rcmd in enumerate(recommendations[:5], start=1):
        lines.append(f"  {i}. {rcmd}")

    print("\n".join(lines))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        pdf = sys.argv[1]
    else:
        # default sample PDF
        pdf = "https://arxiv.org/pdf/1706.03762.pdf"
    run_demo(pdf)
