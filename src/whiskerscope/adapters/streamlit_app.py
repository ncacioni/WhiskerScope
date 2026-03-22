"""Whiskerscope — Streamlit live dashboard."""

from __future__ import annotations

import cv2
import streamlit as st

from whiskerscope.composition import (
    create_clip_recorder,
    create_detection_service,
    create_event_counter,
)

st.set_page_config(page_title="Whiskerscope", page_icon="🐱", layout="wide")
st.title("Whiskerscope — Cat Detector")

# Sidebar controls
confidence = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.5, 0.05)
record_clips = st.sidebar.checkbox("Record clips on detection", value=False)

# Initialize services
service = create_detection_service(confidence=confidence)
counter = create_event_counter()
clip_svc = create_clip_recorder() if record_clips else None

# Placeholders
video_placeholder = st.empty()
col1, col2, col3 = st.columns(3)
cats_now = col1.empty()
total_det = col2.empty()
max_sim = col3.empty()
clip_log = st.sidebar.empty()

stop = st.button("Stop")

while not stop:
    event = service.process_frame()
    if event is None:
        break

    counter.update(event.cat_count)
    frame = event.frame.data

    # Draw bounding boxes
    for det in event.detections:
        b = det.bbox
        cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 255, 0), 2)
        label = f"cat {b.confidence:.0%}"
        cv2.putText(frame, label, (int(b.x1), int(b.y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Clip recording
    if clip_svc and event is not None:
        clip_path = clip_svc.update(event)
        if clip_path:
            clip_log.info(f"Saved: {clip_path}")

    # Display
    video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    stats = counter.stats()
    cats_now.metric("Cats now", event.cat_count)
    total_det.metric("Total detections", stats["total_detections"])
    max_sim.metric("Max simultaneous", stats["max_simultaneous"])

service.camera.release()
