from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.api.routes import provision_pack
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
    def __init__(self, *, falla: bool = False) -> None:
        self.falla = falla
        self.pedidos: list[tuple[str, str | None]] = []

    async def provision(self, pack: str, variant: str | None = None) -> str:
        if self.falla:
            raise UnknownPackError(pack)
        self.pedidos.append((pack, variant))
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
