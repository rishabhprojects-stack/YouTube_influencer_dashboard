from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector


def extract_frames(video_path: str | Path, interval_sec: int = 2) -> list[dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps else 0

    frames: list[dict[str, Any]] = []
    frame_idx = 0
    step = max(1, int(interval_sec * fps))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            ts = frame_idx / fps
            frames.append({"timestamp": ts, "frame": frame})
        frame_idx += 1

    cap.release()
    return frames


def smile_and_expression_metrics(
    sampled_frames: list[dict[str, Any]],
    max_frames: int = 1200,
    smile_scale_factor: float = 1.7,
    smile_min_neighbors: int = 22,
) -> dict[str, Any]:
    if not sampled_frames:
        return {
            "smile_occurrences": 0,
            "smile_frequency_per_min": 0.0,
            "smile_duration_percentage": 0.0,
            "expression_distribution": {},
            "expression_timeline": [],
        }

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

    emotion_model = None
    try:
        from fer import FER

        emotion_model = FER(mtcnn=False)
    except Exception:
        emotion_model = None

    limited = sampled_frames[:max_frames]
    smile_hits = 0
    with_face = 0
    expression_count: dict[str, int] = {}
    timeline: list[dict[str, Any]] = []

    for item in limited:
        ts = item["timestamp"]
        frame = item["frame"]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            with_face += 1

        has_smile = False
        for (x, y, w, h) in faces:
            roi_gray = gray[y : y + h, x : x + w]
            smiles = smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=smile_scale_factor,
                minNeighbors=smile_min_neighbors,
            )
            if len(smiles) > 0:
                has_smile = True
                break

        if has_smile:
            smile_hits += 1

        expression = "unknown"
        if emotion_model is not None:
            try:
                preds = emotion_model.detect_emotions(frame)
                if preds and "emotions" in preds[0]:
                    expression = max(preds[0]["emotions"], key=preds[0]["emotions"].get)
            except Exception:
                expression = "unknown"

        expression_count[expression] = expression_count.get(expression, 0) + 1
        timeline.append({"timestamp": ts, "expression": expression, "smiling": has_smile})

    total = len(limited)
    duration_min = (limited[-1]["timestamp"] - limited[0]["timestamp"]) / 60.0 if total > 1 else 0.0
    smile_per_min = smile_hits / duration_min if duration_min > 0 else float(smile_hits)

    return {
        "smile_occurrences": smile_hits,
        "smile_frequency_per_min": round(smile_per_min, 3),
        "smile_duration_percentage": round((smile_hits / total) * 100, 2) if total else 0.0,
        "expression_distribution": expression_count,
        "expression_timeline": timeline,
        "frames_with_face": with_face,
    }


def energy_estimation(video_path: str | Path, sample_fps: int = 1) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(fps / sample_fps))

    prev_gray = None
    frame_idx = 0
    motion_scores: list[float] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion_scores.append(float(np.mean(diff)))
            prev_gray = gray
        frame_idx += 1

    cap.release()

    if not motion_scores:
        return {"energy_score": 0.0, "motion_magnitude": 0.0}

    motion = float(np.mean(motion_scores))
    norm = min(100.0, (motion / 40.0) * 100.0)
    return {"energy_score": round(norm, 2), "motion_magnitude": round(motion, 4)}


def editing_style(video_path: str | Path) -> dict[str, Any]:
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector())
    manager.detect_scenes(video)

    scenes = manager.get_scene_list()
    scene_count = len(scenes)

    if scene_count == 0:
        return {
            "jump_cuts_frequency": 0,
            "scene_transition_frequency": 0,
            "average_shot_length_sec": 0,
            "editing_intensity_score": 0,
            "editing_style_classification": "unknown",
        }

    durations = []
    for start, end in scenes:
        durations.append((end.get_seconds() - start.get_seconds()))

    avg_shot = float(np.mean(durations)) if durations else 0.0
    transitions_per_min = scene_count / max(1e-6, (video.duration.get_seconds() / 60.0))

    intensity = min(100.0, transitions_per_min * 8)
    if intensity > 65:
        label = "fast_paced_social"
    elif intensity > 35:
        label = "vlog_style"
    else:
        label = "cinematic_or_long_take"

    return {
        "jump_cuts_frequency": scene_count,
        "scene_transition_frequency": round(transitions_per_min, 3),
        "average_shot_length_sec": round(avg_shot, 3),
        "editing_intensity_score": round(intensity, 2),
        "editing_style_classification": label,
    }
