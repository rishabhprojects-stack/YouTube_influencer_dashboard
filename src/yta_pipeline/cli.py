from __future__ import annotations

import argparse

import nltk

from yta_pipeline.pipeline import run_pipeline
from yta_pipeline.utils.config import load_config
from yta_pipeline.utils.logging import get_logger


def ensure_nltk_resources() -> None:
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local open-source YouTube analytics pipeline")
    p.add_argument("--url", required=True, help="YouTube video URL")
    p.add_argument("--output", default="outputs/report.json", help="Path for output JSON report")
    p.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logger = get_logger()
    ensure_nltk_resources()

    config = load_config(args.config)
    report = run_pipeline(args.url, args.output, config)

    logger.info("Analysis complete")
    logger.info("Output report: %s", args.output)
    logger.info("Detected brands: %s", ", ".join(report.get("brand_analysis", {}).get("brand_mentions", {}).get("brands", [])))


if __name__ == "__main__":
    main()
