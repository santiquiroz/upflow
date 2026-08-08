from __future__ import annotations

from pathlib import Path

import pytest

from app.api.routes import capability_tree
from app.config import Settings
from app.services.capabilities import CATALOG, DOMAIN_ORDER
from app.services.model_registry import ModelEntry, ModelKind, ModelStatus


class FakeRegistry:
    def __init__(self, entries: list[ModelEntry] | None = None) -> None:
        self._entries = entries or []

    def list(self) -> list[ModelEntry]:
        return list(self._entries)


def entry(kind: ModelKind) -> ModelEntry:
    return ModelEntry(
        id=f"m-{kind.value}",
        name="m",
        kind=kind,
        source="test",
        size_bytes=1,
        status=ModelStatus.installed,
    )


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), _env_file=None, **overrides)


def flat(response) -> dict:
    return {
        item.id: item
        for group in response.domains
        for item in (*group.capabilities, *group.roadmap)
    }


@pytest.mark.asyncio
async def test_the_tree_exposes_the_four_domains_in_order(tmp_path: Path):
    response = await capability_tree(make_settings(tmp_path), FakeRegistry())
    assert [group.domain for group in response.domains] == list(DOMAIN_ORDER)


@pytest.mark.asyncio
async def test_the_tree_exposes_every_capability_of_the_catalog(tmp_path: Path):
    response = await capability_tree(make_settings(tmp_path), FakeRegistry())
    assert sorted(flat(response)) == sorted(item.id for item in CATALOG)


@pytest.mark.asyncio
async def test_the_roadmap_is_separate_from_the_live_capabilities(tmp_path: Path):
    # Van separadas para que el frontend les pueda dar el encabezado de mapa de
    # ruta sin filtrar por su cuenta.
    response = await capability_tree(make_settings(tmp_path), FakeRegistry())
    for group in response.domains:
        for item in group.capabilities:
            assert item.status != "not_implemented"
        for item in group.roadmap:
            assert item.status == "not_implemented"
            assert item.unavailable_reason_key


@pytest.mark.asyncio
async def test_the_status_follows_the_disk(tmp_path: Path):
    binary = tmp_path / "rife.exe"
    settings = make_settings(tmp_path, RIFE_BINARY=str(binary))

    before = flat(await capability_tree(settings, FakeRegistry()))["video.interpolate"]
    binary.write_bytes(b"x")
    after = flat(await capability_tree(settings, FakeRegistry()))["video.interpolate"]

    assert before.status == "needs_setup"
    assert before.missing_packs == ["rife"]
    assert after.status == "available"
    assert after.missing_packs == []


@pytest.mark.asyncio
async def test_the_registry_decides_the_generation_capability(tmp_path: Path):
    settings = make_settings(tmp_path)

    empty = flat(await capability_tree(settings, FakeRegistry()))["generate.textToImage"]
    filled = flat(
        await capability_tree(settings, FakeRegistry([entry(ModelKind.diffusion_onnx)]))
    )["generate.textToImage"]

    assert empty.status == "needs_setup"
    assert filled.status == "available"


@pytest.mark.asyncio
async def test_the_response_serializes_camel_case_aliases(tmp_path: Path):
    response = await capability_tree(make_settings(tmp_path), FakeRegistry())
    dumped = response.model_dump(by_alias=True)

    group = dumped["domains"][0]
    assert "labelKey" in group
    item = group["roadmap"][0] if group["roadmap"] else group["capabilities"][0]
    assert "labelKey" in item
    assert "jobKind" in item
    assert "missingPacks" in item
    assert "unavailableReasonKey" in item
    assert "setupReasonKey" in item


@pytest.mark.asyncio
async def test_the_copy_travels_as_translation_keys(tmp_path: Path):
    response = await capability_tree(make_settings(tmp_path), FakeRegistry())
    for group in response.domains:
        assert group.label_key.startswith("capability.domain.")
        for item in (*group.capabilities, *group.roadmap):
            assert item.label_key.startswith("capability.")
            for key in (item.unavailable_reason_key, item.setup_reason_key):
                if key is not None:
                    assert key.startswith("capability.")


@pytest.mark.asyncio
async def test_every_capability_reports_its_strategies(tmp_path: Path):
    response = await capability_tree(make_settings(tmp_path), FakeRegistry())
    for item in flat(response).values():
        assert item.strategies
        assert set(item.strategies) <= {"dsp", "model"}


# ---------------------------------------------------------------------------
# Provision de paquetes
# ---------------------------------------------------------------------------


class FakeProvisioner:
    def __init__(self) -> None:
        self.requested: list[str] = []
        self._jobs: dict[str, object] = {}

    async def provision(self, pack: str) -> str:
        from app.services.pack_provisioner import ProvisionJob

        self.requested.append(pack)
        job = ProvisionJob(id=f"job-{len(self.requested)}", pack=pack)
        self._jobs[job.id] = job
        return job.id

    def status(self, job_id: str):
        return self._jobs.get(job_id)


class FakeApp:
    def __init__(self, provisioner: FakeProvisioner) -> None:
        self.state = type("S", (), {"pack_provisioner": provisioner})()


class FakeRequest:
    def __init__(self, provisioner: FakeProvisioner) -> None:
        self.app = FakeApp(provisioner)


@pytest.mark.asyncio
async def test_provisioning_a_capability_starts_its_missing_pack(tmp_path: Path):
    from app.api.routes import provision_capability

    settings = make_settings(tmp_path, RIFE_BINARY=str(tmp_path / "nope.exe"))
    provisioner = FakeProvisioner()

    response = await provision_capability(
        "video.interpolate", FakeRequest(provisioner), settings, FakeRegistry()
    )

    assert provisioner.requested == ["rife"]
    assert response.pack == "rife"
    assert response.status == "queued"
    assert response.status_url.endswith(response.job_id)


@pytest.mark.asyncio
async def test_provisioning_an_unknown_capability_is_404(tmp_path: Path):
    from fastapi import HTTPException

    from app.api.routes import provision_capability

    with pytest.raises(HTTPException) as exc_info:
        await provision_capability(
            "no.existe", FakeRequest(FakeProvisioner()), make_settings(tmp_path), FakeRegistry()
        )
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_provisioning_an_already_available_capability_is_409_with_reload_hint(tmp_path: Path):
    # Caso real: el primer click instaló el pack, la tarjeta quedó vieja y el
    # segundo click recibía un 400 críptico. Ahora es 409 con la cura.
    from fastapi import HTTPException

    from app.api.routes import provision_capability

    binary = tmp_path / "rife.exe"
    binary.write_bytes(b"x")
    settings = make_settings(tmp_path, RIFE_BINARY=str(binary))

    with pytest.raises(HTTPException) as exc_info:
        await provision_capability(
            "video.interpolate", FakeRequest(FakeProvisioner()), settings, FakeRegistry()
        )
    assert exc_info.value.status_code == 409
    assert "Recargá" in exc_info.value.detail


@pytest.mark.asyncio
async def test_a_capability_missing_only_a_model_cannot_be_provisioned(tmp_path: Path):
    # No hay paquete que bajar: lo que falta es instalar un modelo, que es otro
    # camino. Prometer una descarga aca seria un boton que no arregla nada.
    from fastapi import HTTPException

    from app.api.routes import provision_capability

    with pytest.raises(HTTPException) as exc_info:
        await provision_capability(
            "generate.textToImage",
            FakeRequest(FakeProvisioner()),
            make_settings(tmp_path),
            FakeRegistry(),
        )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_a_registry_capability_can_still_need_a_pack(tmp_path: Path):
    # video.upscale es de provisioning "registry" y AUN ASI necesita el binario
    # del motor: lo que decide es el paquete faltante, no la etiqueta.
    from app.api.routes import provision_capability

    settings = make_settings(tmp_path, ENGINE_BINARY=str(tmp_path / "nope.exe"))
    provisioner = FakeProvisioner()

    response = await provision_capability(
        "video.upscale", FakeRequest(provisioner), settings, FakeRegistry()
    )
    assert response.pack == "realesrgan"


@pytest.mark.asyncio
async def test_provision_status_reports_the_job(tmp_path: Path):
    from app.api.routes import provision_capability, provision_status

    settings = make_settings(tmp_path, RIFE_BINARY=str(tmp_path / "nope.exe"))
    provisioner = FakeProvisioner()
    request = FakeRequest(provisioner)
    started = await provision_capability("video.interpolate", request, settings, FakeRegistry())

    reported = await provision_status(started.job_id, request)
    assert reported.job_id == started.job_id
    assert reported.pack == "rife"


@pytest.mark.asyncio
async def test_provision_status_of_an_unknown_job_is_404():
    from fastapi import HTTPException

    from app.api.routes import provision_status

    with pytest.raises(HTTPException) as exc_info:
        await provision_status("nope", FakeRequest(FakeProvisioner()))
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_the_provision_response_serializes_camel_case_aliases(tmp_path: Path):
    from app.api.routes import provision_capability

    settings = make_settings(tmp_path, RIFE_BINARY=str(tmp_path / "nope.exe"))
    response = await provision_capability(
        "video.interpolate", FakeRequest(FakeProvisioner()), settings, FakeRegistry()
    )
    dumped = response.model_dump(by_alias=True)
    assert "jobId" in dumped
    assert "statusUrl" in dumped


# ---------------------------------------------------------------------------
# A traves de la app real
#
# Todo lo de arriba llama las corrutinas DIRECTO, que es el patron del repo pero
# no ejercita ni el ruteo ni la inyeccion de dependencias: un prefijo mal escrito
# o un Depends mal cableado pasaria entero. Estos van por TestClient.
# ---------------------------------------------------------------------------


def test_the_tree_answers_through_the_real_app():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        response = client.get("/api/v1/capabilities/tree")

    assert response.status_code == 200
    payload = response.json()
    assert [group["domain"] for group in payload["domains"]] == list(DOMAIN_ORDER)
    # camelCase en el cable, no los nombres snake_case de Python.
    first = payload["domains"][0]
    assert "labelKey" in first
    for item in (*first["capabilities"], *first["roadmap"]):
        assert "unavailableReasonKey" in item
        assert "missingPacks" in item


def test_the_provisioner_is_wired_into_the_app_state():
    # Sin esto el endpoint de provision tiraria AttributeError en la primera
    # llamada real, que es exactamente como se rompio /audio/voice-catalog.
    from fastapi.testclient import TestClient

    from app.main import app
    from app.services.pack_provisioner import PackProvisioner

    with TestClient(app):
        assert isinstance(app.state.pack_provisioner, PackProvisioner)


def test_provisioning_an_unknown_capability_answers_404_through_the_app():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        response = client.post("/api/v1/capabilities/no.existe/provision")

    assert response.status_code == 404


def test_the_provision_status_of_an_unknown_job_answers_404_through_the_app():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        response = client.get("/api/v1/capabilities/provision/nope")

    assert response.status_code == 404
