---
name: yolo-api-data-layer
description: Refactor the YOLO FastAPI data layer from raw sqlite3 to SQLAlchemy ORM. Use when asked to migrate services/yolo to SQLAlchemy, add SQLAlchemy models/tables/endpoints, replace raw SQL or conn.execute calls, set up Base/engine/SessionLocal/get_db dependency injection, support SQLite and PostgreSQL via env vars, or write tests with temporary SQLite databases.
---

# YOLO API Data Layer: sqlite3 -> SQLAlchemy

Refactor the YOLO object-detection service in `services/yolo/app.py` so all
database access goes through **SQLAlchemy ORM** instead of raw `sqlite3`.

## Hard rules (do not violate)

1. Replace **all** raw `sqlite3` usage with the SQLAlchemy ORM.
2. Do **not** change any existing public API behavior.
3. Keep every existing endpoint, HTTP status code, and response body structure
   **exactly** the same (same JSON keys, same shapes, same error details).
4. Do **not** use raw SQL strings: no `CREATE TABLE`, `INSERT INTO`, `SELECT`,
   `DELETE`, and no `conn.execute(...)`. Use ORM models and `Session` queries.
   (Plain `text()` SQL is also disallowed.)
5. Tests must use **temporary SQLite databases** and must **never** touch the
   real database file or a real PostgreSQL server.
6. When API tests are added or updated, **mock the YOLO model** and assert
   **both** the status code **and** the response body structure.

## Current data layer (what you are replacing)

`services/yolo/app.py` currently uses `sqlite3` with `DB_PATH = "predictions.db"`:

- `init_db()` runs `CREATE TABLE IF NOT EXISTS` for two tables and three indexes.
- `save_prediction_session(uid, original_image, predicted_image)` -> `INSERT`.
- `save_detection_object(prediction_uid, label, score, box)` -> `INSERT`.
- Endpoints reading the DB: `GET /prediction/{uid}`, `GET /prediction/{uid}/image`,
  `GET /predictions/label/{label}`, `GET /predictions/score/{min_score}`.
- `POST /predict` writes one session + N detection objects.

### Existing tables (recreate these exactly as ORM models)

`prediction_sessions`:
- `uid TEXT PRIMARY KEY`
- `timestamp DATETIME DEFAULT CURRENT_TIMESTAMP`
- `original_image TEXT`
- `predicted_image TEXT`

`detection_objects`:
- `id INTEGER PRIMARY KEY AUTOINCREMENT`
- `prediction_uid TEXT` (FK -> `prediction_sessions.uid`)
- `label TEXT`
- `score REAL`
- `box TEXT`
- indexes on `prediction_uid`, `label`, `score`

## Target architecture

Create a dedicated module `services/yolo/db.py` so models and session wiring stay
separate from request handlers. Keep it explicit and readable.

```python
# services/yolo/db.py
import os
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, ForeignKey, func, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker

# SQLite for local dev; override with a full PostgreSQL URL in production.
# e.g. DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/dbname
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./predictions.db")

# check_same_thread is a SQLite-only connect arg; omit it for other backends.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class PredictionSession(Base):
    __tablename__ = "prediction_sessions"
    uid = Column(String, primary_key=True)
    timestamp = Column(DateTime, server_default=func.now())
    original_image = Column(String)
    predicted_image = Column(String)


class DetectionObject(Base):
    __tablename__ = "detection_objects"
    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_uid = Column(String, ForeignKey("prediction_sessions.uid"))
    label = Column(String)
    score = Column(Float)
    box = Column(String)


Index("idx_prediction_uid", DetectionObject.prediction_uid)
Index("idx_label", DetectionObject.label)
Index("idx_score", DetectionObject.score)


def init_db():
    """Create tables. Replaces the old CREATE TABLE IF NOT EXISTS block."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Environment variables (SQLite + PostgreSQL)

- Default (local dev): `DATABASE_URL=sqlite:///./predictions.db`.
- PostgreSQL: set `DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/db`.
- Do not hardcode the engine; always read `DATABASE_URL`.
- Add `sqlalchemy` to `services/yolo/requirements.txt`, and `psycopg2-binary`
  for PostgreSQL support.

## Refactor steps

Copy this checklist and track progress:

```
- [ ] Step 1: Create services/yolo/db.py (Base, engine, SessionLocal, models, get_db, init_db)
- [ ] Step 2: Remove `import sqlite3` and DB_PATH; import from db.py in app.py
- [ ] Step 3: Rewrite save/query helpers and endpoints to use Session + ORM
- [ ] Step 4: Add Depends(get_db) to every endpoint that touches the DB
- [ ] Step 5: Update requirements.txt (sqlalchemy, psycopg2-binary)
- [ ] Step 6: Update tests to use a temporary SQLite DB and mocked YOLO model
- [ ] Step 7: Run pytest and confirm behavior is unchanged
```

### Step 3 + 4: endpoints with dependency injection

Every endpoint that reads or writes the database must declare the session via
FastAPI dependency injection:

```python
from fastapi import Depends
from sqlalchemy.orm import Session
from db import get_db, PredictionSession, DetectionObject

@app.get("/prediction/{uid}")
def get_prediction_by_uid(uid: str, db: Session = Depends(get_db)):
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
            {"id": o.id, "label": o.label, "score": o.score, "box": o.box}
            for o in objects
        ],
    }
```

**Preserve exact response shapes.** The dict keys, nesting, list ordering
semantics, and error `detail` strings must match the current implementation:
- `/prediction/{uid}` -> 404 `"Prediction not found"` when missing.
- `/prediction/{uid}/image` -> 404 `"Image not found"` when session or file is missing.
- `/predictions/label/{label}` -> 400 `"Label cannot be empty"` for blank labels.
- `/predictions/score/{min_score}` -> 400 `"min_score must be between 0.0 and 1.0"`.

For writes in `/predict`, create model instances, `db.add(...)` them, and
`db.commit()`. `timestamp` is auto-populated by `server_default=func.now()`,
matching the old `DEFAULT CURRENT_TIMESTAMP`.

## Testing requirements

Tests must construct a **temporary SQLite database** (e.g. under pytest's
`tmp_path`) and override the `get_db` dependency so the real DB is never used.

```python
import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app as yolo_app
from db import Base, get_db


@pytest.fixture
def client(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path/'test.db'}"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    yolo_app.app.dependency_overrides[get_db] = override_get_db
    # Never load real weights: mock the YOLO model used by /predict.
    monkeypatch.setattr(yolo_app, "model", _FakeModel())
    yield TestClient(yolo_app.app)
    yolo_app.app.dependency_overrides.clear()
```

When testing `/predict` or any endpoint that runs inference, **mock the YOLO
model** (no real weights, no GPU/CPU inference) and assert **both** the status
code and the JSON body keys/structure.

## Verification

- No `sqlite3`, `conn.execute`, or raw SQL strings remain in `services/yolo/`.
- All endpoints touching the DB use `db: Session = Depends(get_db)`.
- `pytest services/yolo/tests/ -v` passes with unchanged response shapes.
- Switching `DATABASE_URL` to a PostgreSQL URL requires no code changes.

## Self-check against the evals

This skill ships with eval cases at [evals/evals.json](evals/evals.json). After
making changes, find the case whose `prompt` matches the task you performed and
confirm your work satisfies **every** entry in that case's `assertions` list.
Treat the assertions as acceptance criteria; do not consider the task done until
they all hold. Use these cases as a guide:

- Refactoring the existing API → `refactor-existing-api-to-sqlalchemy`
- Adding a SQLAlchemy-backed endpoint → `add-new-sqlalchemy-endpoint`
- Adding a new model/table → `add-new-sqlalchemy-model-table`
- Deleting prediction data → `delete-prediction-with-sqlalchemy`
- PostgreSQL via env vars → `support-postgres-via-env`
- Temp-SQLite + mocked-model tests → `tests-use-temp-sqlite-and-mock-model`
