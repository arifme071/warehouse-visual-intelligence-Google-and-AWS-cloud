"""
Cloud Infrastructure - AWS (S3 + Rekognition)
Mirrors the Google Cloud setup (setup_gcs.py) but uses:
  - AWS S3       → image storage (equivalent to GCS)
  - AWS Rekognition → object detection (equivalent to GCP Vision API)

Run once to set up: python -m cloud_infra.setup_aws
"""

import os
import json
from pathlib import Path
from loguru import logger


# ─── Config from .env ─────────────────────────────────────────────
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION            = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME        = os.getenv("S3_BUCKET_NAME", "warehouse-visual-intelligence")


def get_s3_client():
    """Return a boto3 S3 client using env credentials."""
    import boto3
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def get_rekognition_client():
    """Return a boto3 Rekognition client."""
    import boto3
    return boto3.client(
        "rekognition",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


# ─── S3 Bucket Setup ──────────────────────────────────────────────

def create_bucket(bucket_name: str = S3_BUCKET_NAME, region: str = AWS_REGION) -> bool:
    """
    Create an S3 bucket if it does not already exist.

    Args:
        bucket_name: Name of the S3 bucket
        region: AWS region to create bucket in

    Returns:
        True on success, False on failure
    """
    try:
        s3 = get_s3_client()

        # Check if bucket already exists
        existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
        if bucket_name in existing:
            logger.info(f"S3 bucket already exists: s3://{bucket_name}")
            return True

        # us-east-1 does NOT accept LocationConstraint
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region},
            )

        # Block all public access (security best practice)
        s3.put_public_access_block(
            Bucket=bucket_name,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )

        logger.success(f"S3 bucket created: s3://{bucket_name} in {region}")
        return True

    except Exception as e:
        logger.error(f"Failed to create S3 bucket: {e}")
        return False


# ─── Upload / Download ────────────────────────────────────────────

def upload_image(local_path: Path, s3_folder: str = "input/") -> str | None:
    """
    Upload a local image to S3.

    Args:
        local_path: Path to the local image file
        s3_folder: Destination folder prefix in S3

    Returns:
        S3 URI (s3://bucket/key) or None on failure
    """
    try:
        s3 = get_s3_client()
        key = s3_folder + local_path.name
        s3.upload_file(str(local_path), S3_BUCKET_NAME, key)
        uri = f"s3://{S3_BUCKET_NAME}/{key}"
        logger.success(f"Uploaded: {local_path.name} → {uri}")
        return uri
    except Exception as e:
        logger.error(f"S3 upload failed: {e}")
        return None


def upload_report(report_path: Path, s3_folder: str = "reports/") -> str | None:
    """Upload a JSON report to S3."""
    return upload_image(report_path, s3_folder=s3_folder)


def download_images(s3_folder: str = "input/", local_dir: Path = Path("data/downloaded/")) -> list[Path]:
    """
    Download all images from an S3 folder to a local directory.

    Args:
        s3_folder: S3 key prefix to list and download
        local_dir: Local directory to save files into

    Returns:
        List of downloaded local file paths
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    try:
        s3 = get_s3_client()
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=s3_folder)
        for obj in response.get("Contents", []):
            key = obj["Key"]
            local_path = local_dir / Path(key).name
            s3.download_file(S3_BUCKET_NAME, key, str(local_path))
            downloaded.append(local_path)
            logger.debug(f"Downloaded: {key} → {local_path}")
        logger.info(f"Downloaded {len(downloaded)} file(s) to {local_dir}")
    except Exception as e:
        logger.error(f"S3 download failed: {e}")
    return downloaded


# ─── AWS Rekognition Detection ────────────────────────────────────

def detect_labels_from_bytes(image_bytes: bytes, min_confidence: float = 40.0) -> list[dict]:
    """
    Run AWS Rekognition label detection on raw image bytes.
    Equivalent to Google Cloud Vision object localisation.

    Args:
        image_bytes: JPEG/PNG image as bytes
        min_confidence: Minimum confidence threshold (0-100)

    Returns:
        List of label dicts with name, confidence, and bounding box
    """
    try:
        rekognition = get_rekognition_client()
        response = rekognition.detect_labels(
            Image={"Bytes": image_bytes},
            MaxLabels=20,
            MinConfidence=min_confidence,
        )

        results = []
        for label in response.get("Labels", []):
            for instance in label.get("Instances", []):
                box = instance.get("BoundingBox", {})
                results.append({
                    "name": label["Name"],
                    "confidence": round(label["Confidence"], 2),
                    "bounding_box": {
                        "left":   box.get("Left", 0),
                        "top":    box.get("Top", 0),
                        "width":  box.get("Width", 0),
                        "height": box.get("Height", 0),
                    },
                })
            # Labels without instances (scene-level labels)
            if not label.get("Instances"):
                results.append({
                    "name": label["Name"],
                    "confidence": round(label["Confidence"], 2),
                    "bounding_box": None,
                })

        logger.debug(f"Rekognition: {len(results)} label(s) detected")
        return results

    except Exception as e:
        logger.error(f"Rekognition detection failed: {e}")
        return []


def detect_labels_from_s3(s3_key: str, min_confidence: float = 40.0) -> list[dict]:
    """
    Run Rekognition directly on an image already stored in S3.
    More efficient than downloading first — no data transfer cost.

    Args:
        s3_key: S3 object key (e.g. 'input/warehouse_01.jpg')
        min_confidence: Minimum confidence threshold

    Returns:
        List of label dicts
    """
    try:
        rekognition = get_rekognition_client()
        response = rekognition.detect_labels(
            Image={"S3Object": {"Bucket": S3_BUCKET_NAME, "Name": s3_key}},
            MaxLabels=20,
            MinConfidence=min_confidence,
        )
        labels = response.get("Labels", [])
        logger.debug(f"Rekognition (S3): {len(labels)} label(s) for {s3_key}")
        return labels

    except Exception as e:
        logger.error(f"Rekognition S3 detection failed: {e}")
        return []


def detect_ppe(image_bytes: bytes) -> list[dict]:
    """
    Detect PPE (hard hats, masks, hand covers) using Rekognition PPE API.
    Useful for warehouse safety compliance checks.

    Args:
        image_bytes: JPEG/PNG image as bytes

    Returns:
        List of person-level PPE detection results
    """
    try:
        rekognition = get_rekognition_client()
        response = rekognition.detect_protective_equipment(
            Image={"Bytes": image_bytes},
            SummarizationAttributes={
                "MinConfidence": 80.0,
                "RequiredEquipmentTypes": ["HEAD_COVER", "HAND_COVER", "FACE_COVER"],
            },
        )
        persons = response.get("Persons", [])
        logger.debug(f"PPE check: {len(persons)} person(s) analysed")
        return persons

    except Exception as e:
        logger.error(f"PPE detection failed: {e}")
        return []


# ─── Cost Estimation Helpers ──────────────────────────────────────

def estimate_rekognition_cost(num_images: int) -> dict:
    """
    Estimate AWS Rekognition API cost for a given number of images.
    Pricing as of 2024: $0.001 per image (first 1M images/month).

    Args:
        num_images: Number of images to process

    Returns:
        Cost breakdown dict
    """
    price_per_image = 0.001  # USD
    total = round(num_images * price_per_image, 4)
    return {
        "num_images": num_images,
        "price_per_image_usd": price_per_image,
        "estimated_total_usd": total,
        "note": "AWS Rekognition label detection pricing (first 1M images/month)",
    }


def estimate_s3_cost(storage_gb: float, requests: int = 1000) -> dict:
    """
    Estimate S3 storage and request costs.
    Pricing: ~$0.023/GB storage, $0.0004 per 1000 PUT requests.

    Args:
        storage_gb: Estimated data stored in GB
        requests: Number of PUT/GET requests

    Returns:
        Cost breakdown dict
    """
    storage_cost = round(storage_gb * 0.023, 4)
    request_cost = round((requests / 1000) * 0.0004, 4)
    return {
        "storage_gb": storage_gb,
        "storage_cost_usd": storage_cost,
        "requests": requests,
        "request_cost_usd": request_cost,
        "total_usd": round(storage_cost + request_cost, 4),
        "note": "AWS S3 Standard pricing (us-east-1)",
    }


# ─── CLI Entry Point ──────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Setting up AWS infrastructure...")
    logger.info(f"Region : {AWS_REGION}")
    logger.info(f"Bucket : s3://{S3_BUCKET_NAME}")

    success = create_bucket()

    if success:
        logger.success("AWS S3 bucket ready.")
        cost = estimate_rekognition_cost(100)
        logger.info(f"Cost estimate for 100 images: ${cost['estimated_total_usd']}")
    else:
        logger.error("Setup failed. Check your AWS credentials in .env")
