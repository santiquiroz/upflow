from __future__ import annotations

import pytest

from app.services.generation_converter import (
    _allocation_message_for_export,
    _is_allocation_runtime_error,
    silence_export_noise,
)

# ---------------------------------------------------------------------------
# Dos cosas distintas que hacen ver el export ONNX como una falla:
#
# 1. El ruido: `torch.onnx.export` escupe decenas de TracerWarning por modelo.
#    Son esperadas y no significan nada, pero en la consola del instalador se
#    leen como un volcado de errores. (Reporte real 2026-08-05: un usuario dio
#    la instalacion por fallida viendo solo esas lineas.)
# 2. El fallo de verdad: exportar la UNet de un SDXL en CPU pide mucha RAM. Si
#    no alcanza, muere con un error de asignacion crudo que no dice que hacer.
#    El guard existia para materializar el checkpoint, no para exportarlo.
# ---------------------------------------------------------------------------


class TestSilenceExportNoise:
    def test_the_tracer_warnings_do_not_reach_the_console(self) -> None:
        import warnings

        with warnings.catch_warnings(record=True) as capturadas:
            warnings.simplefilter("always")
            with silence_export_noise():
                warnings.warn(
                    "Converting a tensor to a Python boolean might cause the trace"
                    " to be incorrect",
                    UserWarning,
                )

        assert capturadas == []

    def test_a_warning_that_is_not_noise_still_gets_through(self) -> None:
        # Silenciar todo escondería el aviso que sí importa.
        import warnings

        with warnings.catch_warnings(record=True) as capturadas:
            warnings.simplefilter("always")
            with silence_export_noise():
                warnings.warn("el modelo no cabe en memoria", UserWarning)

        assert len(capturadas) == 1

    def test_it_restores_the_previous_filters(self) -> None:
        import warnings

        antes = list(warnings.filters)
        with silence_export_noise():
            pass
        assert list(warnings.filters) == antes


class TestAllocationMessageForExport:
    def test_it_names_the_step_that_ran_out_of_memory(self) -> None:
        mensaje = _allocation_message_for_export("unet")

        assert "unet" in mensaje
        assert "RAM" in mensaje

    def test_it_tells_the_user_what_to_do(self) -> None:
        mensaje = _allocation_message_for_export(None)

        assert "cerr" in mensaje.lower() or "cierr" in mensaje.lower()


class TestAllocationDetection:
    @pytest.mark.parametrize(
        "texto",
        [
            "DefaultCPUAllocator: not enough memory: you tried to allocate 8589934592 bytes",
            "[enforce fail at alloc_cpu.cpp:117] . DefaultCPUAllocator: can't allocate memory",
            "CUDA out of memory",
            "std::bad_alloc",
        ],
    )
    def test_recognises_the_ways_windows_and_torch_say_it(self, texto: str) -> None:
        assert _is_allocation_runtime_error(RuntimeError(texto))

    def test_an_unrelated_error_is_not_confused_with_it(self) -> None:
        # Confundir cualquier fallo con "falta RAM" mandaria al usuario a cerrar
        # programas por un problema que no es ese.
        assert not _is_allocation_runtime_error(RuntimeError("model_index.json not found"))


class TestExportTranslatesMemoryFailures:
    def test_a_c_plus_plus_allocation_death_becomes_a_message_the_user_can_act_on(
        self, tmp_path, monkeypatch
    ) -> None:
        """Antes moria con `std::bad_alloc`, que no le dice nada a nadie."""
        from app.services import generation_converter as modulo

        class FakeExportModule:
            @staticmethod
            def main_export(*_args, **_kwargs):
                raise RuntimeError("[enforce fail at alloc_cpu.cpp:117] std::bad_alloc")

        class FakeLogging:
            @staticmethod
            def get_verbosity() -> int:
                return 0

            @staticmethod
            def set_verbosity_info() -> None:
                return None

            @staticmethod
            def set_verbosity(_level: int) -> None:
                return None

        import sys
        import types

        paquete = types.ModuleType("optimum.exporters.onnx")
        paquete.main_export = FakeExportModule.main_export
        paquete.base = types.SimpleNamespace(is_onnxruntime_available=lambda: True)
        utils = types.ModuleType("optimum.utils")
        utils.logging = FakeLogging
        monkeypatch.setitem(sys.modules, "optimum.exporters.onnx", paquete)
        monkeypatch.setitem(sys.modules, "optimum.utils", utils)

        with pytest.raises(RuntimeError, match="RAM"):
            modulo._export_with_optimum(tmp_path, tmp_path / "out", lambda _c: None)
