from __future__ import annotations

from pathlib import Path
from typing import Any


def transcribe_audio(
    audio_path: str | Path,
    model_size: str = "base",
    device: str = "auto",
    compute_type: str = "int8",
) -> dict[str, Any]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(str(audio_path), vad_filter=True)

    segs: list[dict[str, Any]] = []
    transcript_parts: list[str] = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            transcript_parts.append(text)
        segs.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
                "avg_logprob": getattr(seg, "avg_logprob", None),
                "no_speech_prob": getattr(seg, "no_speech_prob", None),
            }
        )

    return {
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "transcript": " ".join(transcript_parts),
        "segments": segs,
    }
