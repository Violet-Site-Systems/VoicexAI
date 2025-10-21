#!/usr/bin/env python3
"""Run a minimal end-to-end pipeline locally:
- download a sample PDF
- extract text sections
- create atoms in CognitiveAPI
- run ethical analysis and summarization
- write outputs to data/
"""
import os
import sys
import json
import requests

from cognitive_core.api.cognitive_api import CognitiveAPI
from PyPDF2 import PdfReader


def extract_text_sections(pdf_path: str):
    reader = PdfReader(pdf_path)
    sections = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        sections.append({"title": f"Page {i+1}", "text": text, "page": i+1})
    return sections

SAMPLE_PDF = "https://arxiv.org/pdf/1706.03762.pdf"  # smallish public PDF

def download_pdf(url: str, out_dir: str = "docs") -> str:
    os.makedirs(out_dir, exist_ok=True)
    fname = url.split('/')[-1] or 'policy.pdf'
    path = os.path.join(out_dir, fname)
    print('Downloading', url)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, 'wb') as f:
        f.write(r.content)
    return path


def main():
    os.makedirs('data', exist_ok=True)
    cog = CognitiveAPI()

    pdf_path = download_pdf(SAMPLE_PDF)
    sections = extract_text_sections(pdf_path)

    # create atoms for each section
    atom_ids = []
    for i, s in enumerate(sections):
        atom_id = cog.create_atom({
            'atom_type': 'PolicySection',
            'name': s.get('title', f'section_{i}'),
            'content': s,
            'metadata': {'source': pdf_path}
        })
        atom_ids.append(atom_id)

    report = cog.analyze_policy(atom_ids)

    # Serialize only essential, shallow fields to avoid recursion/non-serializable types
    def shallow(obj):
        if hasattr(obj, '__dict__'):
            try:
                return {k: (v.__dict__ if hasattr(v, '__dict__') else str(v)) for k, v in obj.__dict__.items()}
            except Exception:
                return str(obj)
        return str(obj)

    contradictions = []
    for c in report.get('contradictions', []) or []:
        if hasattr(c, '__dict__'):
            # shallow map
            try:
                contradictions.append({k: (v.__dict__ if hasattr(v, '__dict__') else v) for k, v in c.__dict__.items()})
            except Exception:
                contradictions.append(str(c))
        else:
            contradictions.append(str(c))

    out = {
        'doc': os.path.basename(pdf_path),
        'summary': report.get('summary', {}),
        'contradictions': contradictions,
        'counts': {
            'contradictions': len(report.get('contradictions', []) or []),
            'fairness': len(report.get('fairness', []) or []),
            'ethical_implications': len(report.get('ethical_implications', []) or [])
        }
    }

    with open('data/pipeline_result.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print('Pipeline completed. Results in data/pipeline_result.json')


if __name__ == '__main__':
    main()
