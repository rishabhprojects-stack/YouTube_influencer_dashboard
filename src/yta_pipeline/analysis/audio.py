from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import librosa
import numpy as np


def speech_clarity(
    transcript: str,
    segments: list[dict[str, Any]],
    filler_words: list[str] | None = None,
) -> dict[str, Any]:
    filler_words = filler_words or ["um", "uh", "like", "you know"]

    word_count = len(re.findall(r"\b\w+\b", transcript))
    total_duration = 0.0
    if segments:
        total_duration = max(0.0, float(segments[-1].get("end", 0)) - float(segments[0].get("start", 0)))

    wpm = (word_count / total_duration * 60.0) if total_duration > 0 else 0.0

    filler_count = 0
    low = transcript.lower()
    for fw in filler_words:
        filler_count += low.count(fw.lower())

    filler_rate = (filler_count / word_count) if word_count > 0 else 0.0

    if wpm < 110:
        pacing = "slow"
    elif wpm <= 170:
        pacing = "moderate"
    else:
        pacing = "fast"

    clarity = 100.0
    clarity -= min(40.0, filler_rate * 300.0)
    clarity -= 20.0 if pacing == "fast" else 0.0
    clarity -= 5.0 if pacing == "slow" else 0.0

    return {
        "words_per_minute": round(wpm, 2),
        "filler_word_count": filler_count,
        "filler_word_rate": round(filler_rate, 4),
        "clarity_score": round(max(0.0, clarity), 2),
        "speech_pacing": pacing,
    }


def speech_tone(audio_path: str | Path, sr: int = 16000, hop_length: int = 512) -> dict[str, Any]:
    y, sr = librosa.load(str(audio_path), sr=sr)
    if y.size == 0:
        return {
            "tone_classification": "unknown",
            "pitch_variation": 0.0,
            "emotional_tone_distribution": {},
            "monotone_score": 100.0,
        }

    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        hop_length=hop_length,
    )
    valid_f0 = f0[~np.isnan(f0)]

    if valid_f0.size < 2:
        return {
            "tone_classification": "monotone",
            "pitch_variation": 0.0,
            "emotional_tone_distribution": {"neutral": 1.0},
            "monotone_score": 100.0,
        }

    pitch_std = float(np.std(valid_f0))
    pitch_mean = float(np.mean(valid_f0))
    coeff_var = pitch_std / (pitch_mean + 1e-9)

    if coeff_var < 0.08:
        tone = "monotone"
    elif coeff_var < 0.18:
        tone = "moderately_dynamic"
    else:
        tone = "dynamic"

    monotone_score = max(0.0, min(100.0, 100.0 - coeff_var * 300.0))

    distribution = {
        "neutral": round(max(0.0, 1.0 - coeff_var * 2), 3),
        "excited": round(min(1.0, coeff_var * 1.4), 3),
        "calm": round(max(0.0, 0.8 - coeff_var), 3),
    }

    return {
        "tone_classification": tone,
        "pitch_variation": round(coeff_var, 5),
        "emotional_tone_distribution": distribution,
        "monotone_score": round(monotone_score, 2),
    }


def audio_signal_quality(audio_path: str | Path, sr: int = 16000) -> dict[str, Any]:
    y, _ = librosa.load(str(audio_path), sr=sr)
    if y.size == 0:
        return {"rms": 0.0, "snr_proxy": 0.0, "quality_score": 0.0}

    rms = float(np.sqrt(np.mean(y**2)))
    signal = np.percentile(np.abs(y), 95)
    noise = np.percentile(np.abs(y), 25) + 1e-9
    snr_proxy = float(signal / noise)

    quality = min(100.0, (snr_proxy / 8.0) * 100.0)
    return {
        "rms": round(rms, 6),
        "snr_proxy": round(snr_proxy, 4),
        "quality_score": round(quality, 2),
    }
