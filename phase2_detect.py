"""
Phase 2 - YOLOv8 Detection + Bounding Box Visualisation
Runs real YOLOv8 detections on warehouse images, draws annotated
bounding boxes, saves results, and optionally uploads to GCS/S3.

Run:
    python phase2_detect.py --input data/sample_images/
    python phase2_detect.py --input data/sample_images/warehouse_01.jpg
    python phase2_detect.py --input data/sample_images/ --upload gcs
    python phase2_detect.py --input data/sample_images/ --upload aws
"""

import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from loguru import logger
from datetime import datetime


# ─── Config ───────────────────────────────────────────────────────
OUTPUT_DIR       = Path("output/annotated")
REPORT_DIR       = Path("output/reports")
MODEL_NAME       = "yolov8n.pt"        # nano = fastest on CPU; swap to yolov8s.pt for better accuracy
CONF_THRESHOLD   = 0.35
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Warehouse-relevant COCO classes (YOLOv8 default model)
WAREHOUSE_LABELS = {
    "person":    ("worker",   (0, 200, 0)),       # green
    "truck":     ("vehicle",  (0, 100, 255)),      # orange
    "car":       ("vehicle",  (0, 100, 255)),
    "motorcycle":("vehicle",  (0, 100, 255)),
    "bicycle":   ("vehicle",  (0, 100, 255)),
    "backpack":  ("parcel",   (255, 200, 0)),      # cyan
    "suitcase":  ("parcel",   (255, 200, 0)),
    "bottle":    ("item",     (200, 200, 200)),
    "chair":     ("obstacle", (0, 0, 220)),        # red
    "couch":     ("obstacle", (0, 0, 220)),
    "box":       ("pallet",   (255, 100, 0)),
    "laptop":    ("equipment",(180, 0, 180)),
    "tv":        ("equipment",(180, 0, 180)),
}


def load_model():
    """Load YOLOv8 model (downloads automatically on first run ~6MB)."""
    from ultralytics import YOLO
    logger.info(f"Loading YOLOv8 model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    logger.success("Model loaded ✓")
    return model


def detect_image(model, image_path: Path) -> dict:
    """
    Run YOLOv8 on a single image.

    Returns:
        dict with image array, detections list, and metadata
    """
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning(f"Could not load: {image_path}")
        return None

    h, w = image.shape[:2]
    results = model(image, conf=CONF_THRESHOLD, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            cls_name = result.names[int(box.cls)]
            label_info = WAREHOUSE_LABELS.get(cls_name)
            warehouse_label = label_info[0] if label_info else cls_name
            colour        = label_info[1] if label_info else (200, 200, 200)
            conf          = float(box.conf)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            detections.append({
                "original_class":  cls_name,
                "warehouse_label": warehouse_label,
                "confidence":      round(conf, 3),
                "bbox":            [x1, y1, x2, y2],
                "colour":          colour,
            })

    logger.info(f"  {image_path.name}: {len(detections)} object(s) detected")
    return {
        "image":      image,
        "path":       image_path,
        "detections": detections,
        "image_size": (w, h),
    }


def draw_annotations(result: dict) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image.
    Returns annotated image as np.ndarray.
    """
    image = result["image"].copy()

    for det in result["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        colour  = det["colour"]
        label   = f"{det['warehouse_label']} {det['confidence']:.0%}"

        # Bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)

        # Label text
        cv2.putText(
            image, label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

    # Summary overlay (top-left)
    summary = f"Objects: {len(result['detections'])}  |  Model: {MODEL_NAME}"
    cv2.putText(image, summary, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, summary, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    return image


def save_annotated(annotated: np.ndarray, original_path: Path) -> Path:
    """Save annotated image to output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"annotated_{original_path.name}"
    cv2.imwrite(str(out_path), annotated)
    logger.success(f"  Saved annotated: {out_path}")
    return out_path


def save_report(results: list, output_path: Path) -> Path:
    """Save JSON detection report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "model": MODEL_NAME,
        "confidence_threshold": CONF_THRESHOLD,
        "images_processed": len(results),
        "total_detections": sum(len(r["detections"]) for r in results),
        "results": [
            {
                "image": r["path"].name,
                "image_size": r["image_size"],
                "detections": [
                    {k: v for k, v in d.items() if k != "colour"}
                    for d in r["detections"]
                ],
            }
            for r in results
        ],
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.success(f"Report saved: {output_path}")
    return output_path


def upload_to_cloud(files: list[Path], provider: str):
    """Upload annotated images and report to GCS or S3."""
    if provider == "gcs":
        from cloud_infra.setup_gcs import upload_image
        for f in files:
            upload_image(f, gcs_folder="output/phase2/")
    elif provider == "aws":
        from cloud_infra.setup_aws import upload_image
        for f in files:
            upload_image(f, s3_folder="output/phase2/")
    logger.success(f"Uploaded {len(files)} file(s) to {provider.upper()}")


def run_pipeline(input_path: Path, upload: str = None):
    """Main Phase 2 pipeline."""
    # 1. Collect image paths
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted([
            p for p in input_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ])
    else:
        logger.error(f"Input not found: {input_path}")
        return

    if not image_paths:
        logger.error("No images found. Run: python data/download_samples.py first")
        return

    logger.info(f"Processing {len(image_paths)} image(s) from {input_path}")

    # 2. Load model
    model = load_model()

    # 3. Detect + annotate
    all_results = []
    annotated_paths = []

    for img_path in image_paths:
        result = detect_image(model, img_path)
        if result is None:
            continue

        annotated = draw_annotations(result)
        out_path = save_annotated(annotated, img_path)

        all_results.append(result)
        annotated_paths.append(out_path)

    # 4. Save report
    report_path = REPORT_DIR / f"phase2_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_report(all_results, report_path)
    annotated_paths.append(report_path)

    # 5. Summary
    total = sum(len(r["detections"]) for r in all_results)
    logger.success(f"\n{'='*50}")
    logger.success(f"  Phase 2 Complete!")
    logger.success(f"  Images processed : {len(all_results)}")
    logger.success(f"  Total detections : {total}")
    logger.success(f"  Annotated images : {OUTPUT_DIR}/")
    logger.success(f"  Report           : {report_path}")
    logger.success(f"{'='*50}")

    # 6. Optional cloud upload
    if upload:
        upload_to_cloud(annotated_paths, upload)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: YOLOv8 Warehouse Detection")
    parser.add_argument("--input",  type=str, default="data/sample_images/", help="Image or folder")
    parser.add_argument("--upload", type=str, choices=["gcs", "aws"], default=None, help="Upload to cloud")
    parser.add_argument("--conf",   type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    args = parser.parse_args()

    CONF_THRESHOLD = args.conf
    run_pipeline(Path(args.input), upload=args.upload)
