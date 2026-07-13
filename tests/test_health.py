from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get('/api/v1/health')
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'
        assert 'engine' in data
