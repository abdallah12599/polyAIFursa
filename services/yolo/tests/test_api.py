import os
import shutil

from db import PredictionSession, DetectionObject

TEST_IMAGE = os.path.join(os.path.dirname(__file__), "data", "beatles.jpeg")


def _seed_session(session_factory, uid, original=None, predicted=None, objects=None):
    """
    Insert a prediction session (and its detection objects) directly into the
    test database so endpoint tests don't have to run the YOLO model.
    objects is a list of (label, score, box) tuples.
    """
    db = session_factory()
    db.add(PredictionSession(uid=uid, original_image=original, predicted_image=predicted))
    for label, score, box in (objects or []):
        db.add(DetectionObject(prediction_uid=uid, label=label, score=score, box=box))
    db.commit()
    db.close()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict(client):
    with open(TEST_IMAGE, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("beatles.jpeg", f, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "prediction_uid" in data
    assert "detection_count" in data
    assert "labels" in data
    assert "time_took" in data
    # FakeModel always returns a single "person" detection.
    assert data["detection_count"] == 1
    assert data["labels"] == ["person"]


def test_get_prediction_by_uid(client, session_factory):
    _seed_session(
        session_factory,
        "uid-1",
        original="orig.jpg",
        predicted="pred.jpg",
        objects=[("person", 0.91, "[10, 20, 100, 200]")],
    )

    response = client.get("/prediction/uid-1")
    assert response.status_code == 200
    data = response.json()
    assert data["uid"] == "uid-1"
    assert set(data.keys()) == {
        "uid",
        "timestamp",
        "original_image",
        "predicted_image",
        "detection_objects",
    }
    assert len(data["detection_objects"]) == 1
    assert data["detection_objects"][0]["label"] == "person"


def test_get_prediction_by_uid_not_found(client):
    response = client.get("/prediction/does-not-exist")
    assert response.status_code == 404
    assert response.json()["detail"] == "Prediction not found"


def test_get_prediction_image(client, session_factory, tmp_path):
    image_path = str(tmp_path / "pred.jpg")
    shutil.copy(TEST_IMAGE, image_path)
    _seed_session(session_factory, "uid-img", original="orig.jpg", predicted=image_path)

    response = client.get("/prediction/uid-img/image")
    assert response.status_code == 200


def test_get_prediction_image_session_not_found(client):
    response = client.get("/prediction/missing/image")
    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found"


def test_get_prediction_image_file_missing(client, session_factory):
    # Session exists, but the predicted image file is gone from disk.
    _seed_session(session_factory, "uid-nofile", original="orig.jpg", predicted="/no/such/pred.jpg")

    response = client.get("/prediction/uid-nofile/image")
    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found"


def test_get_predictions_by_label(client, session_factory):
    _seed_session(session_factory, "uid-a", objects=[("person", 0.91, "[10, 20, 100, 200]")])
    _seed_session(session_factory, "uid-b", objects=[("car", 0.70, "[0, 0, 1, 1]")])

    response = client.get("/predictions/label/person")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["uid"] == "uid-a"
    assert data[0]["detection_objects"][0]["label"] == "person"
    assert data[0]["detection_objects"][0]["score"] == 0.91


def test_get_predictions_by_label_no_matches(client, session_factory):
    _seed_session(session_factory, "uid-a", objects=[("car", 0.70, "[0, 0, 1, 1]")])

    response = client.get("/predictions/label/person")
    assert response.status_code == 200
    assert response.json() == []


def test_get_predictions_by_label_empty(client):
    # A whitespace-only label decodes to an empty value after stripping.
    response = client.get("/predictions/label/%20")
    assert response.status_code == 400
    assert response.json()["detail"] == "Label cannot be empty"


def test_get_detections_by_score(client, session_factory):
    _seed_session(
        session_factory,
        "uid-a",
        objects=[
            ("person", 0.91, "[10, 20, 100, 200]"),
            ("cat", 0.30, "[0, 0, 1, 1]"),
        ],
    )

    response = client.get("/predictions/score/0.5")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["label"] == "person"
    assert data[0]["prediction_uid"] == "uid-a"
    assert data[0]["score"] == 0.91


def test_get_detections_by_score_no_matches(client, session_factory):
    _seed_session(session_factory, "uid-a", objects=[("cat", 0.30, "[0, 0, 1, 1]")])

    response = client.get("/predictions/score/0.9")
    assert response.status_code == 200
    assert response.json() == []


def test_get_detections_by_score_too_high(client):
    response = client.get("/predictions/score/1.5")
    assert response.status_code == 400
    assert response.json()["detail"] == "min_score must be between 0.0 and 1.0"


def test_get_detections_by_score_negative(client):
    response = client.get("/predictions/score/-0.5")
    assert response.status_code == 400
    assert response.json()["detail"] == "min_score must be between 0.0 and 1.0"
