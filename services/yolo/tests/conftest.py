import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.5")

import app as yolo_app
from app import app
from db import Base, get_db
from models import PredictionSession, DetectionObject


# --- Mocked YOLO model -------------------------------------------------------
# Tests must never load real weights or run real inference. These small fakes
# mimic just the parts of the Ultralytics result API that app.predict() uses.

class _Item:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _Coords:
    def __init__(self, coords):
        self._coords = coords

    def tolist(self):
        return self._coords


class _FakeBox:
    def __init__(self, label_idx, score, coords):
        self.cls = [_Item(label_idx)]
        self.conf = [score]
        self.xyxy = [_Coords(coords)]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes

    def plot(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)


class _FakeModel:
    names = {0: "person"}

    def __call__(self, path, device=None, conf=None):
        return [_FakeResult([_FakeBox(0, 0.95, [1.0, 2.0, 3.0, 4.0])])]


# --- Database / client fixtures ---------------------------------------------

@pytest.fixture
def session_factory(tmp_path):
    """A sessionmaker bound to a throwaway SQLite file under tmp_path."""
    url = f"sqlite:///{tmp_path / 'test.db'}"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


@pytest.fixture
def client(session_factory, monkeypatch):
    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(yolo_app, "model", _FakeModel())
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def seed_session(session_factory):
    """Insert a prediction session (and its detection objects) via the ORM so
    endpoint tests don't have to run the YOLO model. objects is a list of
    (label, score, box) tuples."""
    def _seed(uid, original=None, predicted=None, objects=None):
        db = session_factory()
        try:
            db.add(PredictionSession(
                uid=uid,
                original_image=original,
                predicted_image=predicted,
            ))
            for label, score, box in (objects or []):
                db.add(DetectionObject(
                    prediction_uid=uid,
                    label=label,
                    score=score,
                    box=box,
                ))
            db.commit()
        finally:
            db.close()

    return _seed
