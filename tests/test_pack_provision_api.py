from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.api.routes import provision_pack
from app.services.license_gate import LICENSE_REQUIRED_KEY, LicenseNotAcceptedError
from app.services.pack_provisioner import UnknownPackError

# ---------------------------------------------------------------------------
# Provisionar POR PAQUETE, no solo por capacidad.
#
# Casi todas las pantallas saben QUE les falta ("el modelo de voz") pero no a
# que capacidad pertenece, y varios paquetes ni siquiera tienen una: sd.cpp,
# magpie, gmfss y audiosr existen sin figurar en el catalogo. Sin esta ruta esas
# pantallas no podian ofrecer el boton y caian en decirle al usuario que abriera
# una terminal.
# ---------------------------------------------------------------------------


class ProvisionerFalso:
    def __init__(self, *, falla: bool = False, exige_licencia: bool = False) -> None:
        self.falla = falla
        self.exige_licencia = exige_licencia
        self.pedidos: list[tuple[str, str | None]] = []
        self.aceptadas: list[bool] = []

    async def provision(
        self, pack: str, variant: str | None = None, *, accept_license: bool = False
    ) -> str:
        if self.falla:
            raise UnknownPackError(pack)
        if self.exige_licencia and not accept_license:
            raise LicenseNotAcceptedError(pack, LICENSE_REQUIRED_KEY, "Accept the license first.")
        self.pedidos.append((pack, variant))
        self.aceptadas.append(accept_license)
        return "job-1"

    def status(self, job_id: str):
        from app.services.pack_provisioner import ProvisionJob, ProvisionStatus

        return ProvisionJob(id=job_id, pack="kokoro", status=ProvisionStatus.queued)


class PeticionFalsa:
    def __init__(self, provisioner) -> None:
        self.app = type("App", (), {"state": type("S", (), {"pack_provisioner": provisioner})()})()


@pytest.mark.asyncio
async def test_bajar_un_paquete_conocido_encola_el_trabajo():
    provisioner = ProvisionerFalso()

    respuesta = await provision_pack(pack="kokoro", request=PeticionFalsa(provisioner))

    assert respuesta.job_id == "job-1"
    assert provisioner.pedidos == [("kokoro", None)]


@pytest.mark.asyncio
async def test_la_variante_llega_al_provisioner():
    # La traduccion es un modelo por par de idiomas: sin pasar cual, el boton
    # bajaria siempre el mismo y el usuario quedaria otra vez sin salida.
    provisioner = ProvisionerFalso()

    await provision_pack(
        pack="translation", request=PeticionFalsa(provisioner), variant="es-en"
    )

    assert provisioner.pedidos == [("translation", "es-en")]


@pytest.mark.asyncio
async def test_un_paquete_desconocido_es_un_400_y_no_un_500():
    provisioner = ProvisionerFalso(falla=True)

    with pytest.raises(HTTPException) as capturado:
        await provision_pack(pack="no-existe", request=PeticionFalsa(provisioner))

    assert capturado.value.status_code == 400


def test_la_ruta_recibe_la_variante_por_el_nombre_que_manda_el_frontend() -> None:
    """El frontend arma `?variant=es-en`. Si la ruta lo llamara distinto, el par
    llegaria en None y se bajaria siempre el mismo idioma, en silencio.

    Se lee la firma real en vez de confiar: es el tipo de desajuste que ningun
    test de una sola punta encuentra.
    """
    import inspect

    from app.api.routes import provision_pack

    parametros = inspect.signature(provision_pack).parameters

    assert "variant" in parametros
    assert parametros["variant"].default is None


@pytest.mark.asyncio
async def test_a_license_gated_pack_without_acceptance_is_a_keyed_403():
    provisioner = ProvisionerFalso(exige_licencia=True)

    with pytest.raises(HTTPException) as capturado:
        await provision_pack(pack="restore-faces-nc", request=PeticionFalsa(provisioner))

    assert capturado.value.status_code == 403
    assert capturado.value.detail == {"key": LICENSE_REQUIRED_KEY, "reason": "Accept the license first."}
    assert provisioner.pedidos == []


@pytest.mark.asyncio
async def test_accepting_the_license_reaches_the_provisioner():
    provisioner = ProvisionerFalso(exige_licencia=True)

    await provision_pack(
        pack="restore-faces-nc", request=PeticionFalsa(provisioner), accept_license=True
    )

    assert provisioner.aceptadas == [True]


@pytest.mark.asyncio
async def test_called_without_the_flag_the_license_is_not_accepted():
    # El default tiene que ser False de verdad, no el objeto Query (que es truthy).
    provisioner = ProvisionerFalso()

    await provision_pack(pack="kokoro", request=PeticionFalsa(provisioner))

    assert provisioner.aceptadas == [False]


def test_the_license_flag_travels_by_the_name_the_frontend_sends() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.routes import router

    app = FastAPI()
    app.include_router(router)
    provisioner = ProvisionerFalso(exige_licencia=True)
    app.state.pack_provisioner = provisioner

    refused = TestClient(app).post("/api/v1/packs/restore-faces-nc/provision")
    accepted = TestClient(app).post("/api/v1/packs/restore-faces-nc/provision?acceptLicense=true")

    assert refused.status_code == 403
    assert refused.json()["detail"]["key"] == LICENSE_REQUIRED_KEY
    assert accepted.status_code == 202
    assert provisioner.aceptadas == [True]
