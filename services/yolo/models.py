from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, func, Index

from db import Base


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
