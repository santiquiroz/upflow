from __future__ import annotations

import pytest

from app.services.missing_pack import (
    PACK_LABELS,
    MissingPack,
    UnknownPackLabel,
    label_for,
    missing_pack_message,
)
from app.services.pack_provisioner import PACK_SCRIPTS


def test_todo_paquete_descargable_tiene_como_nombrarse() -> None:
    # Sin esto, un paquete nuevo llega al usuario como una excepcion cruda o
    # como el nombre interno del pack, que no le dice nada.
    sin_nombre = sorted(set(PACK_SCRIPTS) - set(PACK_LABELS))
    assert sin_nombre == [], f"Paquetes sin descripcion para el usuario: {sin_nombre}"


def test_ningun_nombre_menciona_un_script_ni_una_ruta() -> None:
    for pack, etiqueta in PACK_LABELS.items():
        assert ".ps1" not in etiqueta, pack
        assert "scripts/" not in etiqueta, pack


def test_el_mensaje_dice_donde_actuar_y_no_un_comando() -> None:
    mensaje = missing_pack_message("shap-e")

    assert "boton" in mensaje
    assert ".ps1" not in mensaje


def test_el_detalle_se_suma_sin_pisar_la_frase() -> None:
    mensaje = missing_pack_message("openscad", detail="Se esperaba en C:/vendor.")

    assert "OpenSCAD" in mensaje
    assert "C:/vendor" in mensaje


def test_la_excepcion_lleva_el_pack_para_que_la_pantalla_sepa_que_boton_dar() -> None:
    # Una frase sola no alcanza: sin el pack, el frontend puede explicar el
    # problema pero no ofrecer la descarga, que es todo el punto.
    with pytest.raises(MissingPack) as capturado:
        raise MissingPack("kokoro")

    assert capturado.value.pack == "kokoro"


def test_un_paquete_desconocido_falla_fuerte_en_vez_de_inventar_texto() -> None:
    with pytest.raises(UnknownPackLabel):
        label_for("no-existe")
