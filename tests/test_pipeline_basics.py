from yta_pipeline.pipeline import _engagement_metrics


def test_engagement_metrics_zero_views():
    out = _engagement_metrics({"views": 0, "likes": 10, "comment_count": 3})
    assert out["engagement_rate"] == 0.0


def test_engagement_metrics_nonzero_views():
    out = _engagement_metrics({"views": 100, "likes": 10, "comment_count": 5})
    assert out["like_to_view_ratio"] == 0.1
    assert out["comment_to_view_ratio"] == 0.05
    assert out["engagement_rate"] == 0.15
