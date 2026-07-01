import os

TEST_IMAGE = os.path.join(os.path.dirname(__file__), "data", "beatles.jpeg")


def test_predict_includes_processing_time(client):
    response = client.post(
        "/predict",
        json={"image_s3_key": "test/original/beatles.jpeg"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "time_took" in data
    assert isinstance(data["time_took"], (int, float))
    assert data["time_took"] >= 0
