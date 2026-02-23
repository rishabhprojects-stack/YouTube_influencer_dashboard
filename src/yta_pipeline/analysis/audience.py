from __future__ import annotations

import re
from typing import Any


def preprocess_comments(comments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out: list[dict[str, Any]] = []

    for c in comments:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        if normalized in seen:
            continue
        if len(normalized) < 2:
            continue
        # Basic spam heuristic
        if normalized.count("http") > 1 or normalized.count("subscribe") > 2:
            continue
        seen.add(normalized)
        out.append({**c, "normalized_text": normalized})

    return out


def _sentiment_label(text: str) -> tuple[str, float]:
    from nltk.sentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive", score
    if score <= -0.05:
        return "negative", score
    return "neutral", score


def audience_sentiment(comments: list[dict[str, Any]]) -> dict[str, Any]:
    if not comments:
        return {
            "overall_average_score": 0.0,
            "distribution": {"positive": 0, "neutral": 0, "negative": 0},
            "counts": {"positive": 0, "neutral": 0, "negative": 0},
        }

    dist = {"positive": 0, "neutral": 0, "negative": 0}
    scores = []
    labeled = []
    for c in comments:
        label, score = _sentiment_label(c["normalized_text"])
        dist[label] += 1
        scores.append(score)
        labeled.append({**c, "sentiment": label, "sentiment_score": score})

    n = len(comments)
    return {
        "overall_average_score": sum(scores) / n,
        "distribution": {k: (v / n) * 100.0 for k, v in dist.items()},
        "counts": dist,
        "labeled_comments": labeled,
    }


def sentiment_toward_video(comments: list[dict[str, Any]], video_keywords: list[str]) -> dict[str, Any]:
    in_favor = against = neutral = 0
    for c in comments:
        txt = c.get("normalized_text", "")
        if video_keywords and not any(k in txt for k in video_keywords):
            continue
        lbl = c.get("sentiment")
        if lbl == "positive":
            in_favor += 1
        elif lbl == "negative":
            against += 1
        else:
            neutral += 1

    total = in_favor + against + neutral
    if total == 0:
        return {"counts": {"in_favor": 0, "against": 0, "neutral": 0}, "percentages": {"in_favor": 0, "against": 0, "neutral": 0}}

    return {
        "counts": {"in_favor": in_favor, "against": against, "neutral": neutral},
        "percentages": {
            "in_favor": in_favor / total * 100,
            "against": against / total * 100,
            "neutral": neutral / total * 100,
        },
    }


def sentiment_toward_brands(comments: list[dict[str, Any]], brands: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for brand in brands:
        k = brand.lower()
        counts = {"positive": 0, "neutral": 0, "negative": 0}
        for c in comments:
            txt = c.get("normalized_text", "")
            if k in txt:
                lbl = c.get("sentiment", "neutral")
                counts[lbl] += 1
        total = sum(counts.values())
        result[brand] = {
            "counts": counts,
            "percentages": {
                s: (counts[s] / total * 100) if total else 0.0
                for s in ["positive", "neutral", "negative"]
            },
        }
    return result


def comment_topic_classification(comments: list[dict[str, Any]], brands: list[str], content_terms: list[str]) -> dict[str, Any]:
    content = brand = unrelated = 0
    brand_terms = [b.lower() for b in brands]

    for c in comments:
        txt = c.get("normalized_text", "")
        mentions_brand = any(b in txt for b in brand_terms) if brand_terms else False
        mentions_content = any(t in txt for t in content_terms) if content_terms else False

        if mentions_brand:
            brand += 1
        elif mentions_content:
            content += 1
        else:
            unrelated += 1

    total = len(comments)
    if total == 0:
        return {"counts": {"content": 0, "brand": 0, "unrelated": 0}, "percentages": {"content": 0, "brand": 0, "unrelated": 0}}

    return {
        "counts": {"content": content, "brand": brand, "unrelated": unrelated},
        "percentages": {
            "content": content / total * 100,
            "brand": brand / total * 100,
            "unrelated": unrelated / total * 100,
        },
    }
