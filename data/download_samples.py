"""
Phase 2 - Step 1: Download Sample Warehouse Images
Downloads warehouse images from Roboflow Universe (free, no API key needed for public datasets).
Run: python data/download_samples.py
"""

import os
import urllib.request
from pathlib import Path
from loguru import logger

# ─── Free public warehouse image URLs ────────────────────────────
# These are direct image links from open datasets - no login needed
SAMPLE_IMAGES = [
    {
        "name": "warehouse_01.jpg",
        "url": "https://images.unsplash.com/photo-1553413077-190dd305871c?w=1280&q=80",
        "desc": "Large warehouse with shelving"
    },
    {
        "name": "warehouse_02.jpg",
        "url": "https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?w=1280&q=80",
        "desc": "Warehouse aisle with pallets"
    },
    {
        "name": "warehouse_03.jpg",
        "url": "https://images.unsplash.com/photo-1504917595217-d4dc5ebe6122?w=1280&q=80",
        "desc": "Forklift in warehouse"
    },
    {
        "name": "warehouse_04.jpg",
        "url": "https://images.unsplash.com/photo-1600880292203-757bb62b4baf?w=1280&q=80",
        "desc": "Warehouse workers and boxes"
    },
    {
        "name": "warehouse_05.jpg",
        "url": "https://images.unsplash.com/photo-1566576912321-d58ddd7a6088?w=1280&q=80",
        "desc": "Storage facility shelves"
    },
]

OUTPUT_DIR = Path("data/sample_images")


def download_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {len(SAMPLE_IMAGES)} sample images to {OUTPUT_DIR}/")

    for item in SAMPLE_IMAGES:
        dest = OUTPUT_DIR / item["name"]
        if dest.exists():
            logger.info(f"  Already exists: {item['name']}")
            continue
        try:
            logger.info(f"  Downloading: {item['name']} ({item['desc']})")
            urllib.request.urlretrieve(item["url"], dest)
            logger.success(f"  Saved: {dest}")
        except Exception as e:
            logger.warning(f"  Failed: {item['name']} — {e}")

    downloaded = list(OUTPUT_DIR.glob("*.jpg"))
    logger.success(f"\nDone! {len(downloaded)} images ready in {OUTPUT_DIR}/")
    logger.info("Next step: python phase2_detect.py --input data/sample_images/")


if __name__ == "__main__":
    download_all()
