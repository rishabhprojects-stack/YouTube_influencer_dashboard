from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


st.set_page_config(page_title="YouTube Analytics Dashboard", page_icon="📊", layout="wide")


def _safe_get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur


def _format_int(v: Any) -> str:
    try:
        return f"{int(v):,}"
    except Exception:
        return "0"


def _format_pct(v: Any) -> str:
    try:
        return f"{float(v) * 100:.2f}%"
    except Exception:
        return "0.00%"


def _format_date(v: Any) -> str:
    if not v:
        return "-"
    s = str(v)
    try:
        if len(s) == 8 and s.isdigit():
            return datetime.strptime(s, "%Y%m%d").strftime("%Y-%m-%d")
        return s
    except Exception:
        return s


def _load_report(path_text: str, uploaded) -> dict[str, Any] | None:
    if uploaded is not None:
        return json.loads(uploaded.getvalue().decode("utf-8"))
    path = Path(path_text)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _bar_df(data: dict[str, Any], col_a: str = "label", col_b: str = "value") -> pd.DataFrame:
    rows = [{col_a: k, col_b: v} for k, v in data.items()]
    return pd.DataFrame(rows)


def render(report: dict[str, Any]) -> None:
    meta = _safe_get(report, "video_metadata", default={})
    eng = _safe_get(report, "engagement_metrics", default={})
    content = _safe_get(report, "content_analysis", default={})
    visual = _safe_get(report, "visual_behavioral_analysis", default={})
    brands = _safe_get(report, "brand_analysis", default={})
    audience = _safe_get(report, "audience_analysis", default={})

    st.title("YouTube Video & Audience Analytics")
    st.caption("Local open-source analysis report viewer")

    st.subheader(_safe_get(meta, "title", default="Untitled"))
    c1, c2, c3 = st.columns(3)
    c1.write(f"**Channel:** {_safe_get(meta, 'channel_name', default='-')}")
    c2.write(f"**Published:** {_format_date(_safe_get(meta, 'publish_date', default='-'))}")
    c3.write(f"**Duration (sec):** {_safe_get(meta, 'duration_sec', default='-')}")
    if _safe_get(meta, "webpage_url"):
        st.markdown(f"[Open Video]({_safe_get(meta, 'webpage_url')})")

    st.markdown("### Executive KPIs")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Views", _format_int(_safe_get(meta, "views", default=0)))
    k2.metric("Likes", _format_int(_safe_get(meta, "likes", default=0)))
    k3.metric("Comments", _format_int(_safe_get(meta, "comment_count", default=0)))
    k4.metric("Like/View", _format_pct(_safe_get(eng, "like_to_view_ratio", default=0)))
    k5.metric("Comment/View", _format_pct(_safe_get(eng, "comment_to_view_ratio", default=0)))
    k6.metric("Engagement", _format_pct(_safe_get(eng, "engagement_rate", default=0)))

    st.markdown("### Sentiment Overview")
    col_l, col_r = st.columns(2)

    transcript_dist = _safe_get(content, "transcript_sentiment", "distribution", default={})
    if transcript_dist:
        tdf = _bar_df(transcript_dist, "sentiment", "count")
        col_l.write("**Transcript Sentiment (segment count)**")
        col_l.bar_chart(tdf.set_index("sentiment"))
    else:
        col_l.info("No transcript sentiment data.")

    audience_dist = _safe_get(audience, "general_sentiment", "sentiment_distribution_percent", default={})
    if audience_dist:
        adf = _bar_df(audience_dist, "sentiment", "percent")
        col_r.write("**Audience Comment Sentiment (%)**")
        col_r.bar_chart(adf.set_index("sentiment"))
    else:
        col_r.info("No audience sentiment data.")

    st.markdown("### Content Intelligence")
    ct1, ct2 = st.columns([2, 1])

    keywords = _safe_get(content, "keyword_extraction", "top_keywords", default=[])
    if keywords:
        kdf = pd.DataFrame(keywords).head(15)
        ct1.write("**Top Keywords**")
        ct1.bar_chart(kdf.set_index("keyword")["score"])
    else:
        ct1.info("No keyword data.")

    storytelling = _safe_get(content, "storytelling_pattern", default={})
    if storytelling:
        srows = pd.DataFrame(
            [{"signal": k, "value": v} for k, v in storytelling.items()]
        )
        ct2.write("**Storytelling Signals**")
        ct2.dataframe(srows, hide_index=True, use_container_width=True)

    topics = _safe_get(content, "topic_recognition", default=[])
    if topics:
        trows = pd.DataFrame(topics)
        st.write("**Detected Topics**")
        st.dataframe(trows, hide_index=True, use_container_width=True)

    st.markdown("### Influencer Behavior")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Smile Occurrences", str(_safe_get(visual, "smile_frequency", "total_smile_occurrences", default=0)))
    b2.metric("Smile / Min", str(_safe_get(visual, "smile_frequency", "smile_frequency_per_minute", default=0)))
    b3.metric("Energy Score", str(_safe_get(visual, "energy_estimation", "energy_score", default=0)))
    b4.metric("Speech Clarity", str(_safe_get(visual, "speech_clarity", "clarity_score", default=0)))

    vb_l, vb_r = st.columns(2)
    expr_dist = _safe_get(visual, "facial_expression", "distribution", default={})
    if expr_dist:
        edf = _bar_df(expr_dist, "expression", "count")
        vb_l.write("**Facial Expression Distribution**")
        vb_l.bar_chart(edf.set_index("expression"))
    else:
        vb_l.info("No facial expression data.")

    edit = _safe_get(visual, "editing_style", default={})
    if edit:
        vb_r.write("**Editing Style**")
        vb_r.dataframe(pd.DataFrame([edit]), hide_index=True, use_container_width=True)

    tone = _safe_get(visual, "speech_tone", default={})
    audio_q = _safe_get(visual, "audio_signal_quality", default={})
    tq1, tq2 = st.columns(2)
    if tone:
        tq1.write("**Speech Tone**")
        tq1.dataframe(pd.DataFrame([tone]), hide_index=True, use_container_width=True)
    if audio_q:
        tq2.write("**Audio Quality**")
        tq2.dataframe(pd.DataFrame([audio_q]), hide_index=True, use_container_width=True)

    st.markdown("### Brand Perception")
    brand_mentions = _safe_get(brands, "brand_mentions", default={})
    freq = _safe_get(brand_mentions, "frequency", default={})
    if freq:
        fdf = _bar_df(freq, "brand", "mentions").sort_values("mentions", ascending=False).head(20)
        st.write("**Top Brand/Product Mentions**")
        st.bar_chart(fdf.set_index("brand"))

    brand_comment_sent = _safe_get(brands, "brand_comment_sentiment", default={})
    if brand_comment_sent:
        rows: list[dict[str, Any]] = []
        for brand, data in brand_comment_sent.items():
            counts = data.get("counts", {})
            rows.append(
                {
                    "brand": brand,
                    "positive": counts.get("positive", 0),
                    "neutral": counts.get("neutral", 0),
                    "negative": counts.get("negative", 0),
                }
            )
        bdf = pd.DataFrame(rows)
        if not bdf.empty:
            st.write("**Brand Sentiment in Comments (counts)**")
            st.dataframe(bdf.sort_values(["positive", "negative"], ascending=[False, True]), hide_index=True, use_container_width=True)

    st.markdown("### Audience Breakdown")
    au1, au2 = st.columns(2)
    v_sent = _safe_get(audience, "sentiment_toward_video", default={})
    tcls = _safe_get(audience, "comment_topic_classification", default={})

    if v_sent:
        au1.write("**Sentiment Toward Video**")
        au1.dataframe(pd.DataFrame([v_sent.get("counts", {})]), hide_index=True, use_container_width=True)
    if tcls:
        au2.write("**Comment Topic Classification**")
        au2.dataframe(pd.DataFrame([tcls.get("counts", {})]), hide_index=True, use_container_width=True)

    top_comments = _safe_get(meta, "top_10_comments", default=[])
    if top_comments:
        st.markdown("### Top Comments")
        cdf = pd.DataFrame(top_comments)
        st.dataframe(cdf[[c for c in ["author", "likes", "text", "time"] if c in cdf.columns]], hide_index=True, use_container_width=True)

    warnings = report.get("warnings", [])
    if warnings:
        st.markdown("### Warnings")
        for w in warnings:
            st.warning(w)


with st.sidebar:
    st.header("Report Source")
    uploaded_file = st.file_uploader("Upload report.json", type=["json"])
    report_path = st.text_input("Or use local path", value="outputs/report.json")
    st.caption("Use either upload or local path")

report_obj = _load_report(report_path, uploaded_file)
if report_obj is None:
    st.error("Could not load report. Upload a JSON file or verify the local path.")
else:
    render(report_obj)
