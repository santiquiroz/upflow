"""El pack de AudioSR viene en dos precisiones y hay que elegir la que corre.

fp16 pesa la mitad (2.51 -> 1.26 GiB) y corre 9% mas rapido con la salida a
59.4 dB SI-SDR de la fp32 — inaudible — pero SOLO en GPU: el EP de CPU tiene
muchos menos kernels fp16. Como la precision se decide al instalar y el
dispositivo se elige por trabajo, las dos decisiones pueden contradecirse mucho
despues, y el unico momento util para avisar es antes de crear la sesion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.engines.audiosr.assets import AudioSrAssets
from app.services.engines.audiosr_restore import _require_precision_matches_device
from app.services.pack_provisioner import build_command, default_variant


@pytest.mark.parametrize(
    "device,esperada",
    [
        ("dml:0", "fp16"),
        ("dml:1", "fp16"),
        ("cuda:0", "fp16"),
        ("cpu", "fp32"),
        ("CPU", "fp32"),
        (" cpu ", "fp32"),
    ],
)
def test_la_precision_sigue_al_dispositivo_por_defecto(device: str, esperada: str) -> None:
    assert default_variant("audiosr", device) == esperada


@pytest.mark.parametrize("pack", ["karaoke", "translation", "gmfss", "ffmpeg"])
def test_los_demas_packs_no_tienen_variante_por_defecto(pack: str) -> None:
    # Devolver algo aca haria que el boton pasara un -Precision a scripts que no
    # lo aceptan, y la descarga fallaria en la linea de comandos.
    assert default_variant(pack, "dml:0") is None


def test_la_variante_llega_al_script_como_parametro() -> None:
    comando = build_command("audiosr", "fp16")

    assert comando[-2:] == ["-Precision", "fp16"]


def test_una_precision_inventada_no_llega_a_la_linea_de_comandos() -> None:
    with pytest.raises(ValueError):
        build_command("audiosr", "fp8")


def _assets(tmp_path: Path, precision: str | None) -> AudioSrAssets:
    manifest = {"scale_factor": 1.0, "cfg": {"guidance_scale": 1.0, "unconditional_value": 0.0}}
    if precision is not None:
        manifest["precision"] = precision
    return AudioSrAssets(
        model_dir=tmp_path, manifest=manifest, alphas_cumprod=None, mel_basis=None
    )


def test_fp16_en_cpu_falla_con_una_salida_concreta(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError) as error:
        _require_precision_matches_device(_assets(tmp_path, "fp16"), "cpu")

    # El mensaje tiene que decir QUE hacer: el usuario no puede deducir que el
    # pack se reinstala en otra precision a partir de un fallo de kernel.
    assert "fp32" in str(error.value)


def test_fp16_en_gpu_pasa(tmp_path: Path) -> None:
    _require_precision_matches_device(_assets(tmp_path, "fp16"), "dml:0")


def test_fp32_en_cpu_pasa(tmp_path: Path) -> None:
    _require_precision_matches_device(_assets(tmp_path, "fp32"), "cpu")


def test_un_manifest_viejo_sin_precision_se_trata_como_fp32(tmp_path: Path) -> None:
    # Los packs instalados antes de que existiera fp16 no declaran nada, y son
    # fp32: tratarlos como sospechosos romperia instalaciones que funcionan.
    _require_precision_matches_device(_assets(tmp_path, None), "cpu")
