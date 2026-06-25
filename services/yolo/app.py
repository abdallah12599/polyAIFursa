from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from ultralytics import YOLO
from PIL import Image
import sqlite3
import logging
import os
import uuid
import shutil
import time
import signal


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
DB_PATH = "predictions.db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# Download the AI model (tiny model ~6MB)
model = YOLO("yolov8n.pt")  

# Initialize SQLite
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        # Create the predictions main table to store the prediction session
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_sessions (
                uid TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                original_image TEXT,
                predicted_image TEXT
            )
        """)
        
        # Create the objects table to store individual detected objects in a given image
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detection_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_uid TEXT,
                label TEXT,
                score REAL,
                box TEXT,
                FOREIGN KEY (prediction_uid) REFERENCES prediction_sessions (uid)
            )
        """)
        
        # Create index for faster queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_uid ON detection_objects (prediction_uid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON detection_objects (label)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_score ON detection_objects (score)")


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


def save_prediction_session(uid, original_image, predicted_image):
    """
    Save prediction session to database
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO prediction_sessions (uid, original_image, predicted_image)
            VALUES (?, ?, ?)
        """, (uid, original_image, predicted_image))

def save_detection_object(prediction_uid, label, score, box):
    """
    Save detection object to database
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO detection_objects (prediction_uid, label, score, box)
            VALUES (?, ?, ?, ?)
        """, (prediction_uid, label, score, str(box)))

@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...)):
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

    save_prediction_session(uid, original_path, predicted_path)
    
    detected_labels = []
    for box in results[0].boxes:
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        save_detection_object(uid, label, score, bbox)
        detected_labels.append(label)

    processing_time = round(time.time() - start_time, 2)

    return PredictResponse(
        prediction_uid=uid,
        detection_count=len(results[0].boxes),
        labels=detected_labels,
        time_took=processing_time,
    )

@app.get("/prediction/{uid}")
def get_prediction_by_uid(uid: str):
    """
    Get prediction session by uid with all detected objects
    """

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        # Get prediction session
        session = conn.execute("SELECT * FROM prediction_sessions WHERE uid = ?", (uid,)).fetchone()
        if not session:
            raise HTTPException(status_code=404, detail="Prediction not found")
            
        # Get all detection objects for this prediction
        objects = conn.execute(
            "SELECT * FROM detection_objects WHERE prediction_uid = ?", 
            (uid,)
        ).fetchall()
        
        return {
            "uid": session["uid"],
            "timestamp": session["timestamp"],
            "original_image": session["original_image"],
            "predicted_image": session["predicted_image"],
            "detection_objects": [
                {
                    "id": obj["id"],
                    "label": obj["label"],
                    "score": obj["score"],
                    "box": obj["box"]
                } for obj in objects
            ]
        }


@app.get("/prediction/{uid}/image")
def get_prediction_image(uid: str):
    """
    Return the annotated (bounding-box) images r a prediction
    """
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT predicted_image FROM prediction_sessions WHERE uid = ?", (uid,)
        ).fetchone()
    if not row or not os.path.exists(row[0]):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(row[0])


@app.get("/predictions/label/{label}")
def get_predictions_by_label(label: str):
    """
    Get all prediction sessions that contain at least one detected object
    with the given label (e.g. "person", "car").
    """
    # A label made up of only whitespace (or empty) is not a valid query.
    if not label.strip():
        raise HTTPException(status_code=400, detail="Label cannot be empty")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        # Find every session that has at least one object with this label.
        sessions = conn.execute("""
            SELECT DISTINCT prediction_sessions.uid, prediction_sessions.timestamp
            FROM prediction_sessions
            JOIN detection_objects
                ON prediction_sessions.uid = detection_objects.prediction_uid
            WHERE detection_objects.label = ?
        """, (label,)).fetchall()

        results = []
        for session in sessions:
            objects = conn.execute(
                "SELECT * FROM detection_objects WHERE prediction_uid = ?",
                (session["uid"],)
            ).fetchall()
            results.append({
                "uid": session["uid"],
                "timestamp": session["timestamp"],
                "detection_objects": [
                    {
                        "id": obj["id"],
                        "label": obj["label"],
                        "score": obj["score"],
                        "box": obj["box"]
                    } for obj in objects
                ]
            })
        return results


@app.get("/predictions/score/{min_score}")
def get_detections_by_score(min_score: float):
    """
    Get all detection objects whose confidence score is greater than or
    equal to min_score (a float between 0.0 and 1.0).
    """
    if min_score < 0.0 or min_score > 1.0:
        raise HTTPException(status_code=400, detail="min_score must be between 0.0 and 1.0")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        objects = conn.execute(
            "SELECT * FROM detection_objects WHERE score >= ?",
            (min_score,)
        ).fetchall()

        return [
            {
                "id": obj["id"],
                "prediction_uid": obj["prediction_uid"],
                "label": obj["label"],
                "score": obj["score"],
                "box": obj["box"]
            } for obj in objects
        ]


@app.get("/health")
def health():
    """
    Health check endpoint
    """
    return {"status": "ok"}

    @app.get("/health2")
def health():
    """
    Health check endpoint
    """
    return {"status": "ok"}

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    init_db()
    
    uvicorn.run(app, host="0.0.0.0", port=8080)
