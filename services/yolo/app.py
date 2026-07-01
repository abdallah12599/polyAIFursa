from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import Response
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from sqlalchemy.orm import Session
from ultralytics import YOLO
from PIL import Image
from dotenv import load_dotenv
import boto3
import logging
import os
import uuid
import time
import signal
import tempfile

from db import get_db, init_db
from models import PredictionSession, DetectionObject


load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
def handle_shutdown(signum, frame):
    logging.info("Graceful shutdown requested. Cleaning up before exit...")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

# Disable GPU usage
import torch
torch.cuda.is_available = lambda: False
start_time = time.time()

# Confidence threshold for object detection (0.0 - 1.0).
# Detections below this score are discarded.
# Override with: export CONFIDENCE_THRESHOLD=0.7
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.5))
logging.info(f"CONFIDENCE_THRESHOLD set to {CONFIDENCE_THRESHOLD}")

# S3 is the intermediary store shared with the agent service.
# Bucket and region are never hard-coded; they come from the environment.
AWS_REGION = os.environ.get("AWS_REGION")
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET")
s3_client = boto3.client("s3", region_name=AWS_REGION) if AWS_REGION else None

# Download the AI model (tiny model ~6MB)
model = YOLO("yolov8n.pt")  


class PredictRequest(BaseModel):
    image_s3_key: str


app = FastAPI()

# Expose /metrics endpoints with default process metrics + FastAPI HTTP metrics
Instrumentator().instrument(app).expose(app)

init_db()


@app.post("/predict")
def predict(request: PredictRequest, db: Session = Depends(get_db)):
    """
    Predict objects in an image stored in S3.

    The agent uploads the original image to S3 and sends us only its object
    key. We download it, run YOLO, upload the annotated image back to S3, and
    store the S3 keys (not local paths) in the database.
    """
    if not AWS_REGION or not AWS_S3_BUCKET or s3_client is None:
        raise HTTPException(
            status_code=500,
            detail="S3 is not configured (set AWS_REGION and AWS_S3_BUCKET).",
        )

    start_time = time.time()
    uid = str(uuid.uuid4())
    ext = os.path.splitext(request.image_s3_key)[1] or ".jpg"

    with tempfile.TemporaryDirectory(prefix="yolo-") as temp_dir:
        original_path = os.path.join(temp_dir, "original" + ext)
        predicted_path = os.path.join(temp_dir, "predicted" + ext)

        # 1. Download the original image from S3 to a local temp file for YOLO.
        try:
            s3_client.download_file(AWS_S3_BUCKET, request.image_s3_key, original_path)
        except Exception:
            logging.exception("Failed to download original image from S3")
            raise HTTPException(status_code=502, detail="Failed to download image from S3")

        results = model(original_path, device="cpu", conf=CONFIDENCE_THRESHOLD)

        annotated_frame = results[0].plot()  # NumPy image with boxes
        annotated_image = Image.fromarray(annotated_frame)
        annotated_image.save(predicted_path)

        # 2. Upload the annotated image back to S3. Derive the predicted key from
        #    the original key when possible, otherwise fall back to a uid-based key.
        if "/original/" in request.image_s3_key:
            predicted_image_s3_key = request.image_s3_key.replace("/original/", "/predicted/")
        else:
            predicted_image_s3_key = f"yolo/{uid}/predicted/image.jpg"

        try:
            s3_client.upload_file(predicted_path, AWS_S3_BUCKET, predicted_image_s3_key)
        except Exception:
            logging.exception("Failed to upload predicted image to S3")
            raise HTTPException(status_code=502, detail="Failed to upload predicted image to S3")

        # Store S3 keys (not local paths) so images are still available after
        # the temporary local files are deleted.
        db.add(PredictionSession(
            uid=uid,
            original_image=request.image_s3_key,
            predicted_image=predicted_image_s3_key,
        ))

        detected_labels = []
        for box in results[0].boxes:
            label_idx = int(box.cls[0].item())
            label = model.names[label_idx]
            score = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            db.add(DetectionObject(
                prediction_uid=uid,
                label=label,
                score=score,
                box=str(bbox),
            ))
            detected_labels.append(label)

        db.commit()

        return {
            "prediction_uid": uid,
            "detection_count": len(results[0].boxes),
            "labels": detected_labels,
            "time_took": round(time.time() - start_time, 2),
            "original_image_s3_key": request.image_s3_key,
            "predicted_image_s3_key": predicted_image_s3_key,
        }

@app.get("/prediction/{uid}")
def get_prediction_by_uid(uid: str, db: Session = Depends(get_db)):
    """
    Get prediction session by uid with all detected objects
    """
    session = db.query(PredictionSession).filter(PredictionSession.uid == uid).first()
    if not session:
        raise HTTPException(status_code=404, detail="Prediction not found")

    objects = db.query(DetectionObject).filter(
        DetectionObject.prediction_uid == uid
    ).all()

    return {
        "uid": session.uid,
        "timestamp": session.timestamp,
        "original_image": session.original_image,
        "predicted_image": session.predicted_image,
        "detection_objects": [
            {
                "id": obj.id,
                "label": obj.label,
                "score": obj.score,
                "box": obj.box
            } for obj in objects
        ]
    }


@app.get("/prediction/{uid}/image")
def get_prediction_image(uid: str, db: Session = Depends(get_db)):
    """
    Return the annotated (bounding-box) image for a prediction.

    The annotated image lives in S3 (predicted_image holds the S3 key), so we
    fetch it from S3 and stream the bytes back to the caller.
    """
    if not AWS_REGION or not AWS_S3_BUCKET or s3_client is None:
        raise HTTPException(
            status_code=500,
            detail="S3 is not configured (set AWS_REGION and AWS_S3_BUCKET).",
        )

    session = db.query(PredictionSession).filter(PredictionSession.uid == uid).first()
    if not session or not session.predicted_image:
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        obj = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=session.predicted_image)
        image_bytes = obj["Body"].read()
    except Exception:
        logging.exception("Failed to fetch predicted image from S3")
        raise HTTPException(status_code=404, detail="Image not found")

    return Response(content=image_bytes, media_type="image/jpeg")


@app.get("/predictions/label/{label}")
def get_predictions_by_label(label: str, db: Session = Depends(get_db)):
    """
    Get all prediction sessions that contain at least one detected object
    with the given label (e.g. "person", "car").
    """
    # A label made up of only whitespace (or empty) is not a valid query.
    if not label.strip():
        raise HTTPException(status_code=400, detail="Label cannot be empty")

    # Find every session that has at least one object with this label.
    sessions = (
        db.query(PredictionSession)
        .join(DetectionObject, PredictionSession.uid == DetectionObject.prediction_uid)
        .filter(DetectionObject.label == label)
        .distinct()
        .all()
    )

    results = []
    for session in sessions:
        objects = db.query(DetectionObject).filter(
            DetectionObject.prediction_uid == session.uid
        ).all()
        results.append({
            "uid": session.uid,
            "timestamp": session.timestamp,
            "detection_objects": [
                {
                    "id": obj.id,
                    "label": obj.label,
                    "score": obj.score,
                    "box": obj.box
                } for obj in objects
            ]
        })
    return results


@app.get("/predictions/score/{min_score}")
def get_detections_by_score(min_score: float, db: Session = Depends(get_db)):
    """
    Get all detection objects whose confidence score is greater than or
    equal to min_score (a float between 0.0 and 1.0).
    """
    if min_score < 0.0 or min_score > 1.0:
        raise HTTPException(status_code=400, detail="min_score must be between 0.0 and 1.0")

    objects = db.query(DetectionObject).filter(
        DetectionObject.score >= min_score
    ).all()

    return [
        {
            "id": obj.id,
            "prediction_uid": obj.prediction_uid,
            "label": obj.label,
            "score": obj.score,
            "box": obj.box
        } for obj in objects
    ]


@app.get("/health")
def health():
    """
    Health check endpoint
    """
    return {"status": "ok"}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    init_db()
    
    uvicorn.run(app, host="0.0.0.0", port=8080)
