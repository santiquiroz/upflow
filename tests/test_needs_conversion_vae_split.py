from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.services.generation_installer import _needs_conversion

# ---------------------------------------------------------------------------
# Un repo que YA trae ONNX completo no se convierte.
#
# Al exportar, la carpeta torch `vae` se parte en `vae_encoder` y `vae_decoder`,
# asi que comparando carpeta a carpeta `vae` NUNCA encuentra su gemela y el repo
# entero se da por no-convertido. Medido el 2026-08-06 contra los repos reales:
# sdxl-turbo, stable-diffusion-xl-base-1.0 y LCM_Dreamshaper_v7 traen ONNX
# completo y se convertian igual — cerca de cuarenta minutos tirados por
# instalacion, en tres de los modelos mas usados.
#
# El mapeo ya existia (`_COMPONENT_ALIASES`) pero solo lo usaba la validacion
# posterior, no la decision. Esa inconsistencia era el bug.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Archivo:
    path: str
    size: int = 0


def repo(*rutas: str) -> list[Archivo]:
    return [Archivo(r) for r in rutas]


# Tomado del listado real de stabilityai/sdxl-turbo el 2026-08-06.
SDXL_TURBO = repo(
    "model_index.json",
    "unet/diffusion_pytorch_model.safetensors",
    "unet/model.onnx",
    "vae/diffusion_pytorch_model.safetensors",
    "vae_encoder/model.onnx",
    "vae_decoder/model.onnx",
    "text_encoder/model.safetensors",
    "text_encoder/model.onnx",
)


class TestUnRepoConOnnxCompletoNoSeConvierte:
    def test_el_vae_partido_cuenta_como_presente(self) -> None:
        assert _needs_conversion(SDXL_TURBO) is False

    def test_el_vae_alternativo_de_sdxl_tambien(self) -> None:
        # SDXL base publica `vae` y `vae_1_0`; el export produce un solo par.
        archivos = SDXL_TURBO + repo("vae_1_0/diffusion_pytorch_model.safetensors")

        assert _needs_conversion(archivos) is False

    def test_medio_vae_exportado_no_alcanza(self) -> None:
        # Media exportacion del VAE es una exportacion rota: darla por buena
        # dejaria un pipeline que falla recien al generar.
        archivos = [a for a in SDXL_TURBO if a.path != "vae_decoder/model.onnx"]

        assert _needs_conversion(archivos) is True


class TestLoQueSiTieneQueConvertirse:
    def test_un_repo_solo_pytorch_se_convierte(self) -> None:
        assert _needs_conversion(
            repo(
                "model_index.json",
                "unet/diffusion_pytorch_model.safetensors",
                "vae/diffusion_pytorch_model.safetensors",
            )
        ) is True

    def test_un_componente_sin_onnx_sigue_pidiendo_conversion(self) -> None:
        # Lo que este arreglo NO puede tapar: un componente que de verdad falta.
        archivos = SDXL_TURBO + repo("safety_checker/model.safetensors")

        assert _needs_conversion(archivos) is True

    @pytest.mark.parametrize("faltante", ["unet/model.onnx", "text_encoder/model.onnx"])
    def test_si_falta_el_onnx_de_un_componente_se_convierte(self, faltante: str) -> None:
        archivos = [a for a in SDXL_TURBO if a.path != faltante]

        assert _needs_conversion(archivos) is True


class TestLaEtiquetaDelBuscadorDiceLoMismo:
    """La etiqueta y la decisión tienen que coincidir.

    Son dos funciones distintas con la misma regla escrita dos veces, y por eso
    el bug del VAE partido estaba duplicado. Si divergen, el usuario ve
    "requiere conversión" en un modelo que se baja directo — o al revés, que es
    peor.
    """

    def test_un_repo_con_onnx_completo_sale_listo_para_usar(self) -> None:
        from app.services.generation_compat import classify

        veredicto, _razon = classify(
            tuple(a.path for a in SDXL_TURBO) + ("text_encoder/model.onnx",), None
        )

        assert veredicto == "ready_onnx"

    def test_un_repo_solo_pytorch_sale_como_que_requiere_conversion(self) -> None:
        from app.services.generation_compat import classify

        veredicto, _razon = classify(
            ("model_index.json", "unet/diffusion_pytorch_model.safetensors"), None
        )

        assert veredicto == "needs_conversion"
