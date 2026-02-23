from __future__ import annotations

from pathlib import Path
from typing import Any

from yta_pipeline.acquisition.youtube import acquire
from yta_pipeline.analysis.audio import audio_signal_quality, speech_clarity, speech_tone
from yta_pipeline.analysis.audience import (
    audience_sentiment,
    comment_topic_classification,
    preprocess_comments,
    sentiment_toward_brands,
    sentiment_toward_video,
)
from yta_pipeline.analysis.nlp import (
    brand_mentions,
    extract_keywords,
    storytelling_pattern,
    topic_modeling,
    transcript_sentiment,
)
from yta_pipeline.analysis.transcription import transcribe_audio
from yta_pipeline.analysis.visual import editing_style, energy_estimation, extract_frames, smile_and_expression_metrics
from yta_pipeline.utils.io import write_json
from yta_pipeline.utils.logging import get_logger


def _engagement_metrics(meta: dict[str, Any]) -> dict[str, Any]:
    views = float(meta.get("views") or 0)
    likes = float(meta.get("likes") or 0)
    comments = float(meta.get("comment_count") or 0)
    if views <= 0:
        return {"like_to_view_ratio": 0.0, "comment_to_view_ratio": 0.0, "engagement_rate": 0.0}
    return {
        "like_to_view_ratio": likes / views,
        "comment_to_view_ratio": comments / views,
        "engagement_rate": (likes + comments) / views,
    }


def run_pipeline(url: str, output_path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    logger = get_logger()
    warnings: list[str] = []

    acq_cfg = config.get("acquisition", {})
    trans_cfg = config.get("transcription", {})
    nlp_cfg = config.get("nlp", {})
    vis_cfg = config.get("video_analysis", {})
    aud_cfg = config.get("audio_analysis", {})

    logger.info("Acquiring video, metadata, and comments")
    acq = acquire(
        url=url,
        work_dir=acq_cfg.get("work_dir", "outputs/artifacts"),
        max_comments=int(acq_cfg.get("max_comments", 1000)),
    )

    logger.info("Running transcription")
    transcription = transcribe_audio(
        acq.audio_path,
        model_size=trans_cfg.get("model_size", "base"),
        device=trans_cfg.get("device", "auto"),
        compute_type=trans_cfg.get("compute_type", "int8"),
    )
    transcript = transcription.get("transcript", "")
    segments = transcription.get("segments", [])

    logger.info("Analyzing transcript NLP")
    keywords = extract_keywords(transcript, top_n=int(nlp_cfg.get("top_keywords", 25)))
    topics = topic_modeling(segments, n_topics=int(nlp_cfg.get("topic_count", 5)))
    trans_sent = transcript_sentiment(segments)
    narrative = storytelling_pattern(segments)
    brands = brand_mentions(transcript, segments)

    logger.info("Running visual and behavioral analysis")
    frame_interval = int(acq_cfg.get("frame_interval_sec", 2))
    frames = extract_frames(acq.video_path, interval_sec=frame_interval)
    smiles_expr = smile_and_expression_metrics(
        frames,
        max_frames=int(vis_cfg.get("max_frames_for_expression", 1200)),
        smile_scale_factor=float(vis_cfg.get("smile_scale_factor", 1.7)),
        smile_min_neighbors=int(vis_cfg.get("smile_min_neighbors", 22)),
    )
    energy = energy_estimation(acq.video_path, sample_fps=int(vis_cfg.get("sample_fps", 1)))

    try:
        edit = editing_style(acq.video_path)
    except Exception as e:
        warnings.append(f"editing_style_failed: {e}")
        edit = {
            "jump_cuts_frequency": 0,
            "scene_transition_frequency": 0,
            "average_shot_length_sec": 0,
            "editing_intensity_score": 0,
            "editing_style_classification": "unknown",
        }

    logger.info("Running audio analysis")
    clarity = speech_clarity(transcript, segments, filler_words=nlp_cfg.get("filler_words"))
    tone = speech_tone(
        acq.audio_path,
        sr=int(aud_cfg.get("sr", 16000)),
        hop_length=int(aud_cfg.get("hop_length", 512)),
    )
    signal_quality = audio_signal_quality(acq.audio_path, sr=int(aud_cfg.get("sr", 16000)))

    logger.info("Running audience analysis")
    cleaned_comments = preprocess_comments(acq.comments)

    try:
        aud_sent = audience_sentiment(cleaned_comments)
    except Exception as e:
        warnings.append(f"audience_sentiment_failed: {e}")
        aud_sent = {
            "overall_average_score": 0.0,
            "distribution": {"positive": 0, "neutral": 0, "negative": 0},
            "counts": {"positive": 0, "neutral": 0, "negative": 0},
            "labeled_comments": cleaned_comments,
        }

    labeled_comments = aud_sent.get("labeled_comments", cleaned_comments)
    top_terms = [k["keyword"].split()[0].lower() for k in keywords.get("top_keywords", [])[:12]]

    video_sent = sentiment_toward_video(labeled_comments, video_keywords=top_terms)
    brand_sent = sentiment_toward_brands(labeled_comments, brands.get("brands", []))
    topic_cls = comment_topic_classification(labeled_comments, brands.get("brands", []), top_terms)

    report = {
        "video_metadata": {
            **acq.metadata,
            "top_10_comments": sorted(acq.comments, key=lambda c: int(c.get("likes", 0)), reverse=True)[:10],
        },
        "engagement_metrics": _engagement_metrics(acq.metadata),
        "content_analysis": {
            "transcription": {
                "language": transcription.get("language"),
                "full_transcript": transcript,
                "timestamped_transcript": segments,
            },
            "topic_recognition": topics,
            "keyword_extraction": keywords,
            "transcript_sentiment": trans_sent,
            "storytelling_pattern": narrative,
        },
        "visual_behavioral_analysis": {
            "smile_frequency": {
                "total_smile_occurrences": smiles_expr.get("smile_occurrences"),
                "smile_frequency_per_minute": smiles_expr.get("smile_frequency_per_min"),
                "smile_duration_percentage": smiles_expr.get("smile_duration_percentage"),
            },
            "facial_expression": {
                "distribution": smiles_expr.get("expression_distribution"),
                "timeline": smiles_expr.get("expression_timeline"),
            },
            "energy_estimation": energy,
            "speech_clarity": clarity,
            "speech_tone": tone,
            "audio_signal_quality": signal_quality,
            "editing_style": edit,
        },
        "brand_analysis": {
            "brand_mentions": brands,
            "brand_comment_sentiment": brand_sent,
        },
        "audience_analysis": {
            "comment_volume": len(acq.comments),
            "comments_after_cleaning": len(cleaned_comments),
            "general_sentiment": {
                "overall_average_sentiment_score": aud_sent.get("overall_average_score", 0.0),
                "sentiment_distribution_percent": aud_sent.get("distribution", {}),
                "sentiment_counts": aud_sent.get("counts", {}),
            },
            "sentiment_toward_video": video_sent,
            "comment_topic_classification": topic_cls,
        },
        "artifacts": {
            "video_path": str(acq.video_path),
            "audio_path": str(acq.audio_path),
            "frame_samples": len(frames),
        },
        "warnings": warnings,
    }

    write_json(output_path, report)
    return report
