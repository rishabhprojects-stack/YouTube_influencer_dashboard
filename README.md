# YouTube Video + Audience Analytics Pipeline (Local, Open-Source)

This project implements a local analytics pipeline that accepts a YouTube URL and produces a structured JSON report with:

- Video acquisition and metadata
- Transcript and NLP analysis
- Visual/behavioral indicators
- Audience/comment sentiment and topical insights
- Brand mention detection and brand sentiment estimates

## Principles

- Local-first execution after data retrieval
- Open-source tools/models only
- Modular architecture for replacing individual models/components

## Quick Start

1. Install system deps:
   - `ffmpeg`
   - (optional) `tesseract` for text-overlay detection
2. Create a virtual env and install package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

3. Run analysis:

```bash
yta-analyze --url "https://www.youtube.com/watch?v=<VIDEO_ID>" --output "outputs/report.json"
```

4. Run dashboard:

```bash
pip install -e .[dashboard]
streamlit run dashboard.py
```

## Output

The pipeline writes a JSON report with sections:

- `video_metadata`
- `engagement_metrics`
- `content_analysis`
- `visual_behavioral_analysis`
- `brand_analysis`
- `audience_analysis`
- `artifacts`

## Notes

- Comment retrieval depends on YouTube comment accessibility and downloader limits.
- Emotion/smile analysis uses baseline open-source CV methods and should be calibrated for production use.
- If optional models are missing, the pipeline degrades gracefully and records warnings.
