from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL
from youtube_comment_downloader import SORT_BY_POPULAR, YoutubeCommentDownloader

from yta_pipeline.utils.io import ensure_dir


@dataclass
class AcquisitionResult:
    video_path: Path
    audio_path: Path
    metadata: dict[str, Any]
    comments: list[dict[str, Any]]
    video_id: str


def extract_video_id(url: str) -> str:
    patterns = [r"v=([\w-]{11})", r"youtu\.be/([\w-]{11})", r"shorts/([\w-]{11})"]
    for pat in patterns:
        match = re.search(pat, url)
        if match:
            return match.group(1)
    return "unknown_video"


def download_video_and_metadata(url: str, work_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    work = ensure_dir(work_dir)
    out_template = str(work / "%(id)s.%(ext)s")
    ydl_opts = {
        "outtmpl": out_template,
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        file_path = Path(ydl.prepare_filename(info))
        if file_path.suffix.lower() != ".mp4":
            candidate = file_path.with_suffix(".mp4")
            if candidate.exists():
                file_path = candidate

    metadata = {
        "video_id": info.get("id"),
        "title": info.get("title"),
        "channel_name": info.get("channel"),
        "description": info.get("description", ""),
        "publish_date": info.get("upload_date"),
        "duration_sec": info.get("duration"),
        "views": info.get("view_count"),
        "likes": info.get("like_count"),
        "comment_count": info.get("comment_count"),
        "webpage_url": info.get("webpage_url"),
    }
    return file_path, metadata


def extract_audio(video_path: Path, output_audio_path: Path) -> Path:
    import ffmpeg

    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    (
        ffmpeg
        .input(str(video_path))
        .output(str(output_audio_path), ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )
    return output_audio_path


def fetch_comments(url: str, max_comments: int = 1000) -> list[dict[str, Any]]:
    downloader = YoutubeCommentDownloader()
    items = downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)
    comments: list[dict[str, Any]] = []
    for i, c in enumerate(items):
        if i >= max_comments:
            break
        comments.append({
            "text": c.get("text", ""),
            "author": c.get("author", ""),
            "likes": c.get("votes", 0),
            "time": c.get("time", ""),
            "cid": c.get("cid", ""),
        })
    return comments


def acquire(url: str, work_dir: str | Path, max_comments: int = 1000) -> AcquisitionResult:
    video_id = extract_video_id(url)
    data_dir = ensure_dir(Path(work_dir) / video_id)

    video_path, metadata = download_video_and_metadata(url, data_dir)
    audio_path = data_dir / f"{video_id}.wav"
    extract_audio(video_path, audio_path)
    comments = fetch_comments(url, max_comments=max_comments)

    return AcquisitionResult(
        video_path=video_path,
        audio_path=audio_path,
        metadata=metadata,
        comments=comments,
        video_id=video_id,
    )
