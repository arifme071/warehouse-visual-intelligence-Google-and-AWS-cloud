"""
Warehouse Visual Intelligence — Streamlit Dashboard (Phase 2)
Live YOLOv8 detections with annotated image output and agent pipeline.

Run: streamlit run dashboard/app.py
"""

import json
import sys
from pathlib import Path
import numpy as np
import cv2
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Warehouse Visual Intelligence",
    page_icon="🏭",
    layout="wide",
)

# ─── Header ─────────────────────────────────────────────────────
st.title("🏭 Warehouse Visual Intelligence System")
st.caption("Multi-agent AI · YOLOv8 · Google Cloud + AWS · Real-time cost analysis")

# ─── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.selectbox(
        "YOLOv8 Model",
        ["yolov8n.pt (fastest)", "yolov8s.pt (balanced)", "yolov8m.pt (accurate)"],
        index=0,
        help="Nano is best for CPU. Use Medium on GPU."
    )
    model_name = model_choice.split(" ")[0]

    conf_threshold = st.slider("Confidence threshold", 0.1, 1.0, 0.35, 0.05)

    cloud_export = st.selectbox(
        "Export annotated image to",
        ["None", "Google Cloud Storage", "AWS S3"],
    )

    st.divider()
    st.markdown("**Quick Start**")
    st.code("python data/download_samples.py", language="bash")
    st.code("streamlit run dashboard/app.py", language="bash")

# ─── Image Input ────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📷 Input")
    tab_upload, tab_sample = st.tabs(["Upload Image", "Sample Images"])
    image_array = None

    with tab_upload:
        uploaded = st.file_uploader("Choose a warehouse image", type=["jpg", "jpeg", "png"])
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB),
                     caption=uploaded.name, use_container_width=True)

    with tab_sample:
        sample_dir = Path("data/sample_images")
        samples = sorted(sample_dir.glob("*.jpg")) + sorted(sample_dir.glob("*.png")) if sample_dir.exists() else []
        if samples:
            selected = st.selectbox("Select sample", [p.name for p in samples])
            image_array = cv2.imread(str(sample_dir / selected))
            st.image(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB),
                     caption=selected, use_container_width=True)
        else:
            st.warning("No sample images found.")
            st.info("Run this first:")
            st.code("python data/download_samples.py", language="bash")

# ─── Detection & Results ────────────────────────────────────────
with col2:
    st.subheader("🤖 Detection Results")

    if image_array is not None:
        run_btn = st.button("▶ Run Full Pipeline", type="primary", use_container_width=True)

        if run_btn:
            detections = []
            annotated = image_array.copy()

            with st.spinner("Running YOLOv8 detection..."):
                try:
                    from ultralytics import YOLO
                    model = YOLO(model_name)
                    results = model(image_array, conf=conf_threshold, verbose=False)

                    WAREHOUSE_LABELS = {
                        "person":    ("worker",   (0, 200, 0)),
                        "truck":     ("vehicle",  (0, 100, 255)),
                        "car":       ("vehicle",  (0, 100, 255)),
                        "suitcase":  ("parcel",   (255, 200, 0)),
                        "backpack":  ("parcel",   (255, 200, 0)),
                        "chair":     ("obstacle", (0, 0, 220)),
                        "bottle":    ("item",     (200, 200, 200)),
                    }

                    for result in results:
                        for box in result.boxes:
                            cls_name = result.names[int(box.cls)]
                            info = WAREHOUSE_LABELS.get(cls_name, (cls_name, (180, 180, 180)))
                            wlabel, colour = info
                            conf_val = float(box.conf)
                            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                            detections.append({"label": wlabel, "original": cls_name,
                                               "confidence": round(conf_val, 3), "bbox": [x1, y1, x2, y2]})
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                            label_text = f"{wlabel} {conf_val:.0%}"
                            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
                            cv2.putText(annotated, label_text, (x1 + 2, y1 - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                             caption="Annotated Output", use_container_width=True)
                    _, buf = cv2.imencode(".jpg", annotated)
                    st.download_button("⬇ Download Annotated Image", data=buf.tobytes(),
                                       file_name="annotated_output.jpg", mime="image/jpeg")

                except ImportError:
                    st.error("YOLOv8 not installed. Run: pip install ultralytics")

            # ─── Agent Pipeline ─────────────────────────────────
            st.divider()
            st.subheader("📊 Analysis")

            if detections:
                with st.spinner("Running agent pipeline..."):
                    from vision_pipeline.preprocess import preprocess_image
                    from agents.orchestrator import Orchestrator
                    preprocessed = preprocess_image(image_array)
                    orch = Orchestrator()
                    report = orch.run([preprocessed])
                    report_dict = report.to_dict()

                s = report_dict["summary"]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Objects", len(detections))
                m2.metric("Anomalies", s["total_anomalies"])
                m3.metric("Layout Issues", s["total_layout_suggestions"])
                m4.metric("Daily Impact", f"${s['estimated_daily_cost_impact_usd']:.0f}")

                # Detection table
                st.subheader("🔍 Detected Objects")
                import pandas as pd
                df = pd.DataFrame([
                    {"Label": d["label"], "Class": d["original"], "Confidence": f"{d['confidence']:.1%}"}
                    for d in detections
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Anomalies
                if report_dict["anomalies"]:
                    st.subheader("⚠️ Anomalies")
                    for a in report_dict["anomalies"]:
                        icon = "🔴" if a["severity"] == "critical" else "🟡"
                        with st.expander(f"{icon} {a['type']} — {a['severity'].upper()}"):
                            st.write(a["description"])
                            st.caption(a["location"])
                else:
                    st.success("✅ No anomalies detected")

                # Layout suggestions
                if report_dict["layout_suggestions"]:
                    st.subheader("📐 Layout Suggestions")
                    for ls in report_dict["layout_suggestions"]:
                        icon = "🔴" if ls["priority"] == "high" else "🟡"
                        with st.expander(f"{icon} {ls['category']}"):
                            st.write(ls["description"])
                            st.metric("Est. Saving", f"{ls['estimated_saving_pct']}%")

                # Downloads
                st.download_button("⬇ Download Full Report (JSON)",
                                   data=json.dumps(report_dict, indent=2),
                                   file_name="warehouse_report.json", mime="application/json")

                # Cloud export
                if cloud_export != "None":
                    if st.button(f"☁ Export to {cloud_export}"):
                        out_path = Path("output/annotated/dashboard_export.jpg")
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(out_path), annotated)
                        if cloud_export == "Google Cloud Storage":
                            from cloud_infra.setup_gcs import upload_image
                            upload_image(out_path, gcs_folder="output/phase2/")
                        else:
                            from cloud_infra.setup_aws import upload_image
                            upload_image(out_path, s3_folder="output/phase2/")
                        st.success(f"Exported to {cloud_export}!")
            else:
                st.info("No objects detected. Try lowering the confidence threshold in the sidebar.")
    else:
        st.info("👈 Upload an image or select a sample to get started.")
