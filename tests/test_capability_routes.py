from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.capability_routes import get_capability_probe, router
from app.services.capability_probe import CapabilityProbe, Lever, LeverStatus
from app.config import Settings


class FakeCapabilityProbe:
    def __init__(self) -> None:
        self.rescan_called = False
        self.fix_called_with: str | None = None

    async def list_levers(self) -> list[Lever]:
        return [Lever("hags", "HAGS", LeverStatus.ok, "enabled", False)]

    async def rescan(self) -> list[Lever]:
        self.rescan_called = True
        return await self.list_levers()

    async def apply_fix(self, lever_id: str) -> Lever:
        self.fix_called_with = lever_id
        return Lever(lever_id, lever_id, LeverStatus.ok, "fixed", False)


def make_client(fake: FakeCapabilityProbe) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_capability_probe] = lambda: fake
    return TestClient(app)


def test_get_capabilities_returns_levers() -> None:
    fake = FakeCapabilityProbe()
    client = make_client(fake)

    response = client.get("/api/v1/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["levers"][0]["id"] == "hags"
    assert body["levers"][0]["fixable"] is False


def test_post_rescan_calls_rescan() -> None:
    fake = FakeCapabilityProbe()
    client = make_client(fake)

    response = client.post("/api/v1/capabilities/rescan")

    assert response.status_code == 200
    assert fake.rescan_called is True


def test_post_fix_calls_apply_fix_with_lever_id() -> None:
    fake = FakeCapabilityProbe()
    client = make_client(fake)

    response = client.post("/api/v1/capabilities/hags/fix")

    assert response.status_code == 200
    assert fake.fix_called_with == "hags"
    assert response.json()["lever"]["status"] == "ok"
