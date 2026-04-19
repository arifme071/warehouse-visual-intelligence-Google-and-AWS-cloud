"""
Warehouse Visual Intelligence — Streamlit Cloud Dashboard
Cloud-safe version: only uses ultralytics + opencv-headless + streamlit
No dependency on agents/vision_pipeline local modules.
"""

import json
import tempfile
from pathlib import Path
import numpy as np
import cv2
import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="Warehouse Visual Intelligence",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Warehouse Visual Intelligence System")
st.caption("Multi-agent AI · YOLOv8 · Google Cloud + AWS · Real-time cost & safety analysis")

# ─── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox(
        "YOLOv8 Model",
        ["yolov8n.pt (fastest)", "yolov8s.pt (balanced)", "yolov8m.pt (accurate)"],
        index=0,
    )
    model_name = model_choice.split(" ")[0]
    conf_threshold = st.slider("Confidence threshold", 0.10, 1.0, 0.35, 0.05)
    frame_skip = st.slider("Video frame skip", 1, 10, 3,
                           help="Process every Nth frame. Higher = faster.")
    st.divider()
    st.markdown("**🔗 Project Links**")
    st.markdown("[📂 GitHub Repo](https://github.com/arifme071/warehouse-visual-intelligence)")
    st.markdown("[👤 LinkedIn](https://linkedin.com/in/marahman-gsu)")
    st.divider()
    st.info("Upload a warehouse image or video to run the full AI pipeline.")

# ─── Label mapping ──────────────────────────────────────────────
WAREHOUSE_LABELS = {
    "person":     ("worker",   (0, 200, 0)),
    "truck":      ("vehicle",  (0, 100, 255)),
    "car":        ("vehicle",  (0, 100, 255)),
    "motorcycle": ("vehicle",  (0, 100, 255)),
    "bicycle":    ("vehicle",  (0, 100, 255)),
    "suitcase":   ("parcel",   (255, 200, 0)),
    "backpack":   ("parcel",   (255, 200, 0)),
    "chair":      ("obstacle", (0, 0, 220)),
    "bottle":     ("item",     (200, 200, 200)),
    "handbag":    ("parcel",   (255, 200, 0)),
    "tie":        ("item",     (200, 200, 200)),
    "umbrella":   ("obstacle", (0, 0, 220)),
}

# Cost model
COST_MODEL = {
    "SAFETY_VIOLATION": 500.0,
    "MISSING_PPE":      200.0,
    "IDLE_EQUIPMENT":   120.0,
}

def draw_boxes(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        colour = det["colour"]
        label  = f"{det['label']} {det['confidence']:.0%}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

def run_detection(image, model):
    results = model(image, conf=conf_threshold, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            cls_name = result.names[int(box.cls)]
            info = WAREHOUSE_LABELS.get(cls_name, (cls_name, (180, 180, 180)))
            wlabel, colour = info
            conf_val = float(box.conf)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "label": wlabel, "original": cls_name,
                "confidence": round(conf_val, 3),
                "bbox": [x1, y1, x2, y2], "colour": colour,
            })
    return detections

def run_agent_analysis(detections):
    """Lightweight agent analysis — no external dependencies."""
    anomalies = []
    suggestions = []
    cost = 0.0

    workers  = [d for d in detections if d["label"] == "worker"]
    vehicles = [d for d in detections if d["label"] == "vehicle"]
    parcels  = [d for d in detections if d["label"] == "parcel"]

    # Safety violation check
    for v in vehicles:
        for w in workers:
            vx = (v["bbox"][0] + v["bbox"][2]) / 2
            wx = (w["bbox"][0] + w["bbox"][2]) / 2
            vy = (v["bbox"][1] + v["bbox"][3]) / 2
            wy = (w["bbox"][1] + w["bbox"][3]) / 2
            if abs(vx - wx) < 150 and abs(vy - wy) < 150:
                anomalies.append({
                    "type": "SAFETY_VIOLATION",
                    "severity": "critical",
                    "description": "Vehicle detected in close proximity to worker — collision risk.",
                    "location": f"Vehicle bbox: {[round(x) for x in v['bbox']]}"
                })
                cost += COST_MODEL["SAFETY_VIOLATION"]

    # Missing PPE check
    if workers:
        anomalies.append({
            "type": "MISSING_PPE",
            "severity": "warning",
            "description": f"{len(workers)} worker(s) detected — PPE compliance could not be verified.",
            "location": "General scene"
        })
        cost += COST_MODEL["MISSING_PPE"] * len(workers)

    # Layout suggestion
    if len(parcels) >= 2:
        suggestions.append({
            "category": "Pathway Clearance",
            "description": "Multiple parcels detected — check if pathways are clear.",
            "priority": "medium",
            "estimated_saving_pct": 8.0
        })
        cost += 8.0 * 50

    if len(detections) > 5:
        suggestions.append({
            "category": "Zone Density",
            "description": "High object density detected — consider redistributing across zones.",
            "priority": "medium",
            "estimated_saving_pct": 10.0
        })
        cost += 10.0 * 50

    return anomalies, suggestions, round(cost, 2)

def show_analysis(detections):
    """Show full analysis panel."""
    anomalies, suggestions, cost = run_agent_analysis(detections)

    st.divider()
    st.subheader("📊 Analysis")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Objects", len(detections))
    m2.metric("Anomalies", len(anomalies))
    m3.metric("Layout Issues", len(suggestions))
    m4.metric("Daily Impact", f"${cost:.0f}")

    import pandas as pd
    st.subheader("🔍 Detected Objects")
    st.dataframe(pd.DataFrame([
        {"Label": d["label"], "Class": d["original"], "Confidence": f"{d['confidence']:.1%}"}
        for d in detections
    ]), use_container_width=True, hide_index=True)

    if anomalies:
        st.subheader("⚠️ Anomalies")
        for a in anomalies:
            icon = "🔴" if a["severity"] == "critical" else "🟡"
            with st.expander(f"{icon} {a['type']} — {a['severity'].upper()}"):
                st.write(a["description"])
                st.caption(a["location"])
    else:
        st.success("✅ No anomalies detected")

    if suggestions:
        st.subheader("📐 Layout Suggestions")
        for s in suggestions:
            icon = "🔴" if s["priority"] == "high" else "🟡"
            with st.expander(f"{icon} {s['category']}"):
                st.write(s["description"])
                st.metric("Est. Saving", f"{s['estimated_saving_pct']}%")

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "summary": {
            "total_detections": len(detections),
            "total_anomalies": len(anomalies),
            "total_layout_suggestions": len(suggestions),
            "estimated_daily_cost_impact_usd": cost,
        },
        "anomalies": anomalies,
        "layout_suggestions": suggestions,
    }
    st.download_button("⬇ Download Report (JSON)",
                       data=json.dumps(report, indent=2),
                       file_name="warehouse_report.json",
                       mime="application/json")

# ─── Mode selector ──────────────────────────────────────────────
mode = st.radio("Select mode", ["🖼️ Image", "🎥 Video"], horizontal=True)
st.divider()

# ════════════════════════════════════════════════════════════════
# IMAGE MODE
# ════════════════════════════════════════════════════════════════
if mode == "🖼️ Image":
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📷 Input")
        uploaded = st.file_uploader("Upload a warehouse image", type=["jpg", "jpeg", "png"])
        image_array = None

        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB),
                     caption=uploaded.name, use_container_width=True)

    with col2:
        st.subheader("🤖 Detection Results")
        if image_array is not None:
            if st.button("▶ Run Full Pipeline", type="primary", use_container_width=True):
                with st.spinner("Loading YOLOv8 model..."):
                    try:
                        from ultralytics import YOLO
                        model = YOLO(model_name)
                    except Exception as e:
                        st.error(f"Model load failed: {e}")
                        st.stop()

                with st.spinner("Running detection..."):
                    detections = run_detection(image_array, model)
                    annotated  = draw_boxes(image_array.copy(), detections)

                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                         caption="Annotated Output", use_container_width=True)
                _, buf = cv2.imencode(".jpg", annotated)
                st.download_button("⬇ Download Annotated Image",
                                   data=buf.tobytes(),
                                   file_name="annotated_output.jpg",
                                   mime="image/jpeg")

                if detections:
                    show_analysis(detections)
                else:
                    st.info("No objects detected. Try lowering the confidence threshold or use yolov8s.pt.")
        else:
            st.info("👈 Upload a warehouse image to get started.")
            st.markdown("""
**Tips for best detections:**
- Images with people, forklifts or vehicles work best
- Try confidence threshold at 0.20 for challenging images
- Switch to `yolov8s.pt` for better accuracy
            """)

# ════════════════════════════════════════════════════════════════
# VIDEO MODE
# ════════════════════════════════════════════════════════════════
else:
    st.subheader("🎥 Video Processing")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("**Upload a warehouse video**")
        video_file = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"])
        if video_file:
            st.video(video_file)
            st.caption(f"{video_file.name} | {video_file.size / 1024 / 1024:.1f} MB")

    with col2:
        st.markdown("**Processing Controls**")
        if video_file:
            st.info(f"Model: `{model_name}` | Confidence: `{conf_threshold}` | Frame skip: `{frame_skip}`")

            if st.button("▶ Process Video", type="primary", use_container_width=True):
                progress_bar = st.progress(0, text="Initialising...")
                preview      = st.empty()
                metrics_box  = st.empty()

                try:
                    from ultralytics import YOLO
                    model = YOLO(model_name)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        tmp.write(video_file.read())
                        tmp_path = tmp.name

                    cap          = cv2.VideoCapture(tmp_path)
                    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
                    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    out_dir  = Path(tempfile.mkdtemp())
                    out_path = out_dir / f"annotated_{video_file.name}"
                    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
                    writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

                    frame_num        = 0
                    total_detections = 0
                    last_annotated   = None
                    all_detections   = []

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_num += 1
                        pct = frame_num / total_frames if total_frames > 0 else 0
                        progress_bar.progress(min(pct, 1.0),
                                              text=f"Frame {frame_num}/{total_frames}")

                        if frame_num % frame_skip == 0:
                            dets = run_detection(frame, model)
                            total_detections += len(dets)
                            all_detections.extend(dets)
                            annotated = draw_boxes(frame.copy(), dets)
                            overlay   = f"Frame {frame_num} | Detections: {len(dets)}"
                            cv2.putText(annotated, overlay, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
                            cv2.putText(annotated, overlay, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
                            writer.write(annotated)
                            last_annotated = annotated

                            if frame_num % 20 == 0:
                                preview.image(
                                    cv2.cvtColor(last_annotated, cv2.COLOR_BGR2RGB),
                                    caption=f"Live preview — Frame {frame_num}",
                                    use_container_width=True,
                                )
                                metrics_box.markdown(
                                    f"**Frames:** {frame_num}/{total_frames} &nbsp;|&nbsp; "
                                    f"**Detections:** {total_detections}"
                                )
                        else:
                            writer.write(frame)

                    cap.release()
                    writer.release()
                    progress_bar.progress(1.0, text="✅ Complete!")
                    st.success(f"🎉 Done! {frame_num} frames | {total_detections} detections")

                    with open(out_path, "rb") as f:
                        st.download_button(
                            "⬇ Download Annotated Video",
                            data=f.read(),
                            file_name=f"annotated_{video_file.name}",
                            mime="video/mp4",
                        )

                    if last_annotated is not None:
                        st.image(cv2.cvtColor(last_annotated, cv2.COLOR_BGR2RGB),
                                 caption="Final annotated frame", use_container_width=True)

                    if all_detections:
                        show_analysis(all_detections)

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("👈 Upload a video file to get started.")
            st.markdown("""
**Tips:**
- Videos with people or vehicles work best
- Keep under 2 minutes for faster processing
- Use `yolov8s.pt` for better accuracy
            """)
