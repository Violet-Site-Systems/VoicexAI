#!/usr/bin/env python3
"""Demo script to run DeepSeek on the first page of a PDF and output markdown.

This requires DeepSeek dependencies (torch, transformers, pdf2image or pymupdf).
Run this only if you installed the DeepSeek dependencies listed in the README.
"""
import sys
import os
import argparse
from tools.deepseek_client import get_client


def main(pdf_path: str, model_name: str = None, out: str = None):
    ds = get_client(model_name)
    try:
        # Convert first page to an image (let the deepseek client handle loading)
        # deepseek_client will attempt to use fitz or pdf2image internally when calling infer
        res = ds.infer_image_to_markdown(pdf_path, prompt=None)
        if out:
            with open(out, 'w', encoding='utf-8') as f:
                f.write(res)
            print(f"Wrote markdown output to {out}")
        else:
            print(res)
    except Exception as exc:
        print("DeepSeek demo failed:", exc)
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepSeek demo on a PDF file (first page)')
    parser.add_argument('pdf', help='Path to the PDF file')
    parser.add_argument('--model', help='DeepSeek model name (default from HUGGINGFACE_MODEL_SUMMARY env)')
    parser.add_argument('--out', help='Output markdown file path')
    args = parser.parse_args()
    main(args.pdf, args.model, args.out)
