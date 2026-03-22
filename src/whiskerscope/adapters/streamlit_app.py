"""Whiskerscope — Streamlit live dashboard."""

from __future__ import annotations

import os
import time
from collections import deque
from datetime import datetime, timezone

import cv2
import pandas as pd
import streamlit as st

from whiskerscope.composition import (
    create_clip_recorder,
    create_detection_service,
    create_event_counter,
    init_app,
)
from whiskerscope.config import WhiskerscopeConfig

# --- Page config & dark theme ---
st.set_page_config(page_title="Whiskerscope", page_icon="🐱", layout="wide")

st.markdown(
    """<style>
    .stApp { background-color: #0e1117; }
    .status-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
    .green { background-color: #00c853; }
    .red { background-color: #ff1744; }
    </style>""",
    unsafe_allow_html=True,
)

st.title("🐱 Whiskerscope")

# --- Sidebar controls ---
st.sidebar.header("Controls")
confidence = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.5, 0.05)
record_clips = st.sidebar.checkbox("Record clips on detection", value=False)
model_name = st.sidebar.selectbox("Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)

# --- Init services ---
config = WhiskerscopeConfig(confidence=confidence, model_name=model_name)
init_app(config)

camera_ok = True
model_ok = True

try:
    service = create_detection_service(config)
except RuntimeError:
    camera_ok = False
    service = None

counter = create_event_counter(config)
clip_svc = create_clip_recorder(config) if record_clips else None

# --- Status indicators ---
st.sidebar.markdown("---")
st.sidebar.header("Status")
cam_dot = "green" if camera_ok else "red"
st.sidebar.markdown(f'<span class="status-dot {cam_dot}"></span> Camera', unsafe_allow_html=True)
mdl_dot = "green" if model_ok else "red"
st.sidebar.markdown(f'<span class="status-dot {mdl_dot}"></span> Model: {model_name}', unsafe_allow_html=True)

# --- Clip gallery ---
if record_clips:
    st.sidebar.markdown("---")
    st.sidebar.header("Saved Clips")
    clips_dir = config.clip_output_dir
    if os.path.isdir(clips_dir):
        clips = sorted(
            [f for f in os.listdir(clips_dir) if f.endswith(".avi")],
            reverse=True,
        )[:5]
        if clips:
            for c in clips:
                size_kb = os.path.getsize(os.path.join(clips_dir, c)) // 1024
                st.sidebar.text(f"📹 {c} ({size_kb} KB)")
        else:
            st.sidebar.caption("No clips yet")

# --- Error state: camera unavailable ---
if not camera_ok or service is None:
    st.error("Camera unavailable. Check your webcam connection and restart.")
    st.stop()

# --- Main layout ---
col_video, col_timeline = st.columns([7, 3])

with col_video:
    video_placeholder = st.empty()

with col_timeline:
    st.subheader("Detection Timeline")
    chart_placeholder = st.empty()
    st.subheader("Session History")
    history_placeholder = st.empty()

# Metrics row
m1, m2, m3, m4 = st.columns(4)
cats_now_ph = m1.empty()
total_det_ph = m2.empty()
max_sim_ph = m3.empty()
fps_ph = m4.empty()

stop = st.button("Stop")

# --- FPS tracking ---
fps_window: deque[float] = deque(maxlen=30)
last_time = time.monotonic()

# --- Timeline data ---
timeline_data: deque[dict] = deque(maxlen=60)
last_toast_time: float = 0.0

# --- Empty state ---
if not stop:
    cats_now_ph.metric("Cats now", 0)
    total_det_ph.metric("Total detections", 0)
    max_sim_ph.metric("Max simultaneous", 0)
    fps_ph.metric("FPS", "...")

try:
    while not stop:
        event = service.process_frame()
        if event is None:
            break

        # FPS
        now = time.monotonic()
        fps_window.append(now - last_time)
        last_time = now
        fps = 1.0 / (sum(fps_window) / len(fps_window)) if fps_window else 0

        # Stats
        max_conf = max((d.bbox.confidence for d in event.detections), default=0.0)
        counter.update(event.cat_count, max_confidence=max_conf)
        frame = event.frame.data

        # Draw bounding boxes
        for det in event.detections:
            b = det.bbox
            color = (0, 255, 0) if b.confidence >= 0.8 else (0, 255, 255) if b.confidence >= 0.5 else (0, 0, 255)
            cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
            label = f"cat {b.confidence:.0%}"
            cv2.putText(frame, label, (int(b.x1), int(b.y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Toast on first detection after gap
        if event.cat_count > 0 and (now - last_toast_time) > 5.0:
            st.toast(f"🐱 {event.cat_count} cat(s) detected!")
            last_toast_time = now

        # Clip recording
        if clip_svc:
            clip_path = clip_svc.update(event)
            if clip_path:
                st.sidebar.success(f"Saved: {os.path.basename(clip_path)}")

        # Display video
        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        # Timeline
        timeline_data.append({
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "cats": event.cat_count,
        })
        chart_placeholder.line_chart(pd.DataFrame(timeline_data).set_index("time"), height=200)

        # Session history from DB
        history = counter.session_history(limit=10)
        if history:
            history_placeholder.dataframe(
                pd.DataFrame(history),
                use_container_width=True,
                hide_index=True,
            )
        else:
            history_placeholder.caption("No cats detected yet")

        # Metrics
        stats = counter.stats()
        cats_now_ph.metric("Cats now", event.cat_count)
        total_det_ph.metric("Total detections", stats["total_detections"])
        max_sim_ph.metric("Max simultaneous", stats["max_simultaneous"])
        fps_ph.metric("FPS", f"{fps:.1f}")

finally:
    if service is not None:
        service.camera.release()
