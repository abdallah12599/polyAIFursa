from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from ultralytics import YOLO
from PIL import Image
import logging
import os
import uuid
import shutil
import time
import signal

from db import (
    init_db,
    get_db,
    PredictionSession,
    DetectionObject,
)


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

UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# Download the AI model (tiny model ~6MB)
model = YOLO("yolov8n.pt")  


class PredictResponse(BaseModel):
    """Validated shape of the /predict response."""
    prediction_uid: str = Field(..., description="Unique id of this prediction session")
    detection_count: int = Field(..., ge=0, description="Number of objects detected")
    labels: list[str] = Field(default_factory=list, description="Detected object labels")
    time_took: float = Field(..., ge=0, description="Inference time in seconds")


app = FastAPI()

# Expose /metrics endpoints with default process metrics + FastAPI HTTP metrics
Instrumentator().instrument(app).expose(app)

init_db()


def save_prediction_session(db: Session, uid, original_image, predicted_image):
    """
    Save prediction session to database
    """
    db.add(PredictionSession(
        uid=uid,
        original_image=original_image,
        predicted_image=predicted_image,
    ))

def save_detection_object(db: Session, prediction_uid, label, score, box):
    """
    Save detection object to database
    """
    db.add(DetectionObject(
        prediction_uid=prediction_uid,
        label=label,
        score=score,
        box=str(box),
    ))

@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Predict objects in an image
    """
    start_time = time.time()
    ext = os.path.splitext(file.filename)[1]
    uid = str(uuid.uuid4())
    original_path = os.path.join(UPLOAD_DIR, uid + ext)
    predicted_path = os.path.join(PREDICTED_DIR, uid + ext)

    with open(original_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    results = model(original_path, device="cpu", conf=CONFIDENCE_THRESHOLD)

    annotated_frame = results[0].plot()  # NumPy image with boxes
    annotated_image = Image.fromarray(annotated_frame)
    annotated_image.save(predicted_path)

    save_prediction_session(db, uid, original_path, predicted_path)

    detected_labels = []
    for box in results[0].boxes:
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        save_detection_object(db, uid, label, score, bbox)
        detected_labels.append(label)

    db.commit()

    processing_time = round(time.time() - start_time, 2)

    return PredictResponse(
        prediction_uid=uid,
        detection_count=len(results[0].boxes),
        labels=detected_labels,
        time_took=processing_time,
    )

@app.get("/prediction/{uid}")
def get_prediction_by_uid(uid: str, db: Session = Depends(get_db)):
    """
    Get prediction session by uid with all detected objects
    """
    session = db.query(PredictionSession).filter(PredictionSession.uid == uid).first()
    if not session:
        raise HTTPException(status_code=404, detail="Prediction not found")

    objects = (
        db.query(DetectionObject)
        .filter(DetectionObject.prediction_uid == uid)
        .all()
    )

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
    Return the annotated (bounding-box) images r a prediction
    """
    session = db.query(PredictionSession).filter(PredictionSession.uid == uid).first()
    if not session or not session.predicted_image or not os.path.exists(session.predicted_image):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(session.predicted_image)


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
        objects = (
            db.query(DetectionObject)
            .filter(DetectionObject.prediction_uid == session.uid)
            .all()
        )
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

    objects = (
        db.query(DetectionObject)
        .filter(DetectionObject.score >= min_score)
        .all()
    )

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
