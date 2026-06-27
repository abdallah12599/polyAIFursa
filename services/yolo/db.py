"""SQLAlchemy data layer for the YOLO service.

Holds the engine, session factory, ORM models, and FastAPI session dependency.
SQLite is used for local development; set DATABASE_URL to a PostgreSQL URL in
production. No raw SQL is used anywhere in this service.
"""
import os

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    create_engine,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker

# SQLite for local dev; override with a full PostgreSQL URL in production, e.g.
# DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/dbname
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
    """Create all tables. Replaces the old CREATE TABLE IF NOT EXISTS block."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yield a session and always close it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
