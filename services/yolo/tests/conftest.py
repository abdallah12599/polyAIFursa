"""Shared pytest fixtures for the YOLO service tests.

Every test runs against a throwaway SQLite database created under pytest's
`tmp_path`, wired in via FastAPI's dependency override of `get_db`. The real
database is never touched. The YOLO model is replaced with a lightweight fake
so no weights are loaded and no inference runs.
"""
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.5")

import app as app_module
from app import app
from db import Base, get_db


# --- Fake YOLO model: mimics just the bits app.predict() touches ---------------
class _FakeScalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value

    def __float__(self):
        return float(self._value)


class _FakeArray:
    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


class _FakeBox:
    def __init__(self, cls_idx, conf, xyxy):
        self.cls = [_FakeScalar(cls_idx)]
        self.conf = [_FakeScalar(conf)]
        self.xyxy = [_FakeArray(xyxy)]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes

    def plot(self):
        # Small blank RGB image; Image.fromarray() accepts this.
        return np.zeros((10, 10, 3), dtype=np.uint8)


class FakeModel:
    """Deterministic stand-in for the Ultralytics YOLO model."""

    names = {0: "person"}

    def __call__(self, *args, **kwargs):
        return [_FakeResult([_FakeBox(0, 0.95, [1.0, 2.0, 3.0, 4.0])])]


@pytest.fixture
def session_factory(tmp_path):
    """A sessionmaker bound to a temporary SQLite database."""
    url = f"sqlite:///{tmp_path / 'test_predictions.db'}"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)
    return TestingSessionLocal


@pytest.fixture
def client(session_factory, monkeypatch):
    """TestClient with get_db overridden to the temp DB and the model mocked."""
    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(app_module, "model", FakeModel())
    yield TestClient(app)
    app.dependency_overrides.clear()
