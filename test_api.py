import io
from PIL import Image
from fastapi.testclient import TestClient
from classification_service.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_token_generation_success():
    response = client.post(
        "/token",
        data={"username": "beejaai", "password": "sairam123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_token_invalid_password():
    response = client.post(
        "/token",
        data={"username": "beejaai", "password": "wrongpassword"}
    )
    assert response.status_code == 401

def test_classify_without_token():
    response = client.post("/classify")
    assert response.status_code == 401

def test_classify_with_valid_token():
    token_resp = client.post(
        "/token",
        data={"username": "beejaai", "password": "sairam123"}
    )
    token = token_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    response = client.post(
        "/classify",
        headers=headers,
        files={"file": ("test_pan.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    res_data = response.json()
    assert "classified" in res_data
    assert "all_scores" in res_data