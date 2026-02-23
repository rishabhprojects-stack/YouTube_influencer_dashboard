from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def _split_segments(segments: list[dict[str, Any]], window_sec: int = 30) -> list[str]:
    if not segments:
        return []
    buckets: dict[int, list[str]] = defaultdict(list)
    for seg in segments:
        start = int(seg.get("start", 0))
        text = seg.get("text", "")
        buckets[start // window_sec].append(text)
    return [" ".join(buckets[i]).strip() for i in sorted(buckets.keys()) if buckets[i]]


def extract_keywords(text: str, top_n: int = 25) -> dict[str, Any]:
    if not text.strip():
        return {"top_keywords": [], "frequency": {}}
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    matrix = vect.fit_transform([text])
    scores = matrix.toarray()[0]
    terms = np.array(vect.get_feature_names_out())

    idx = np.argsort(scores)[::-1][:top_n]
    keywords = [{"keyword": terms[i], "score": float(scores[i])} for i in idx if scores[i] > 0]

    token_counts = Counter(re.findall(r"\b[a-zA-Z][a-zA-Z0-9'-]+\b", text.lower()))
    return {
        "top_keywords": keywords,
        "frequency": dict(token_counts.most_common(top_n)),
    }


def topic_modeling(segments: list[dict[str, Any]], n_topics: int = 5) -> list[dict[str, Any]]:
    docs = _split_segments(segments, window_sec=45)
    if len(docs) < 2:
        return []

    cv = CountVectorizer(stop_words="english", max_df=0.95, min_df=1)
    dtm = cv.fit_transform(docs)
    if dtm.shape[1] == 0:
        return []

    n_topics = min(n_topics, max(1, dtm.shape[0]))
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)

    feature_names = cv.get_feature_names_out()
    results: list[dict[str, Any]] = []
    for i, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[::-1][:8]
        top_terms = [feature_names[j] for j in top_idx]
        confidence = float(np.max(comp) / (np.sum(comp) + 1e-9))
        results.append(
            {
                "topic_id": i,
                "top_terms": top_terms,
                "confidence": round(confidence, 4),
            }
        )
    return results


def transcript_sentiment(segments: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer

        sia = SentimentIntensityAnalyzer()
    except Exception:
        return {
            "overall": "neutral",
            "score": 0.0,
            "distribution": {"positive": 0, "neutral": len(segments), "negative": 0},
            "segment_scores": [],
        }

    scores = []
    dist = {"positive": 0, "neutral": 0, "negative": 0}
    for seg in segments:
        text = seg.get("text", "")
        comp = sia.polarity_scores(text)["compound"]
        if comp >= 0.05:
            label = "positive"
        elif comp <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        dist[label] += 1
        scores.append({"start": seg.get("start"), "end": seg.get("end"), "score": comp, "label": label})

    avg = float(np.mean([s["score"] for s in scores])) if scores else 0.0
    overall = "positive" if avg >= 0.05 else "negative" if avg <= -0.05 else "neutral"
    return {"overall": overall, "score": avg, "distribution": dist, "segment_scores": scores}


def storytelling_pattern(segments: list[dict[str, Any]]) -> dict[str, Any]:
    full = " ".join(seg.get("text", "") for seg in segments).lower()
    first_chunk = " ".join(seg.get("text", "") for seg in segments[:3]).lower()

    hooks = ["today", "in this video", "you won't believe", "let me show", "watch this"]
    problem_markers = ["problem", "issue", "struggle", "challenge"]
    solution_markers = ["solution", "fix", "how to", "steps", "here's what"]
    cta_markers = ["subscribe", "like this video", "comment below", "follow", "check out"]
    emotion_markers = ["amazing", "excited", "love", "frustrating", "shocking"]

    has_hook = any(k in first_chunk for k in hooks)
    has_problem = any(k in full for k in problem_markers)
    has_solution = any(k in full for k in solution_markers)
    has_cta = any(k in full for k in cta_markers)
    emotional_appeal = any(k in full for k in emotion_markers)

    if has_problem and has_solution:
        arc = "problem_solution"
    elif "story" in full or "happened" in full:
        arc = "testimonial"
    else:
        arc = "educational"

    climax_segment = None
    if segments:
        longest = max(segments, key=lambda s: len(s.get("text", "")))
        climax_segment = {"start": longest.get("start"), "end": longest.get("end")}

    return {
        "hook_presence": has_hook,
        "problem_solution_structure": has_problem and has_solution,
        "emotional_appeal": emotional_appeal,
        "call_to_action_detected": has_cta,
        "climax_position": climax_segment,
        "narrative_arc_type": arc,
    }


def detect_brands_from_text(text: str) -> list[str]:
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        brands = [ent.text.strip() for ent in doc.ents if ent.label_ in {"ORG", "PRODUCT"}]
    except Exception:
        # Fallback heuristic when spaCy model is not available.
        brands = re.findall(r"\b[A-Z][a-zA-Z0-9&]{2,}\b", text)

    cleaned = []
    seen = set()
    for b in brands:
        norm = b.strip().lower()
        if len(norm) < 3:
            continue
        if norm not in seen:
            seen.add(norm)
            cleaned.append(b.strip())
    return cleaned


def brand_mentions(transcript: str, segments: list[dict[str, Any]]) -> dict[str, Any]:
    brands = detect_brands_from_text(transcript)
    lower_transcript = transcript.lower()

    sentiment_by_brand = {b: {"positive": 0, "neutral": 0, "negative": 0} for b in brands}
    frequency = {}

    senti = transcript_sentiment(segments)
    segment_scores = senti.get("segment_scores", [])

    for brand in brands:
        k = brand.lower()
        frequency[brand] = lower_transcript.count(k)
        for seg in segment_scores:
            seg_text = next((s.get("text", "") for s in segments if s.get("start") == seg.get("start")), "")
            if k in seg_text.lower():
                sentiment_by_brand[brand][seg["label"]] += 1

    return {
        "brands": brands,
        "frequency": frequency,
        "sentiment": sentiment_by_brand,
    }
