from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.services.capabilities import (
    CATALOG,
    DOMAIN_ORDER,
    PathRequirement,
    RegistryRequirement,
    group_by_domain,
    resolve_capabilities,
)
from app.services.model_registry import ModelEntry, ModelKind, ModelStatus


class FakeRegistry:
    def __init__(self, entries: list[ModelEntry] | None = None) -> None:
        self._entries = entries or []

    def list(self) -> list[ModelEntry]:
        return list(self._entries)


def entry(kind: ModelKind, status: ModelStatus = ModelStatus.installed) -> ModelEntry:
    return ModelEntry(
        id=f"m-{kind.value}-{status.value}",
        name="m",
        kind=kind,
        source="test",
        size_bytes=1,
        status=status,
    )


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), _env_file=None, **overrides)


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def by_id(resolved: list, capability_id: str):
    return next(item for item in resolved if item.id == capability_id)


# ---------------------------------------------------------------------------
# El catalogo
# ---------------------------------------------------------------------------


def test_the_catalog_covers_the_four_domains():
    assert {capability.domain for capability in CATALOG} == set(DOMAIN_ORDER)


def test_capability_ids_are_unique():
    ids = [capability.id for capability in CATALOG]
    assert len(ids) == len(set(ids))


def test_capability_ids_are_namespaced_by_their_domain():
    # El id se usa en rutas y en claves de traduccion: si el prefijo no coincide
    # con el dominio, el arbol y la copia se desincronizan sin que nada avise.
    for capability in CATALOG:
        assert capability.id.startswith(f"{capability.domain}.")


def test_every_unimplemented_capability_explains_why():
    for capability in CATALOG:
        if capability.provisioning == "none":
            assert capability.unavailable_reason_key, capability.id


def test_no_unimplemented_capability_declares_a_job_kind():
    # Declarar un job_kind seria decir que hay a donde mandar el trabajo.
    for capability in CATALOG:
        if capability.provisioning == "none":
            assert capability.job_kind is None, capability.id


def test_every_queued_capability_declares_a_job_kind():
    # `builtin` queda afuera a proposito: es codigo sincronico en proceso, no
    # trabajo que se encole, y exigirle un job_kind seria pedirle una cola que no
    # tiene.
    for capability in CATALOG:
        if capability.provisioning in ("registry", "vendored_pack"):
            assert capability.job_kind, capability.id


def test_every_capability_declares_at_least_one_strategy():
    for capability in CATALOG:
        assert capability.strategies, capability.id


def test_capabilities_that_depend_on_disk_declare_what_they_need():
    # Una capacidad que depende del disco y no declara requisitos seria
    # `available` siempre, que es justamente la clase de mentira que este modulo
    # existe para evitar.
    #
    # `builtin` es la unica excepcion, y no es una mentira: es codigo en proceso,
    # asi que "siempre disponible" es la verdad. Por eso se exige lo contrario —
    # que NO declare requisitos — en vez de dejar el hueco abierto.
    for capability in CATALOG:
        if capability.provisioning in ("registry", "vendored_pack"):
            assert capability.requirements, capability.id


def test_builtin_capabilities_declare_nothing_to_install():
    # El hueco que abre la excepcion de arriba, cerrado: si una capacidad builtin
    # necesitara algo del disco, esta mal clasificada.
    for capability in CATALOG:
        if capability.provisioning == "builtin":
            assert capability.requirements == (), capability.id


def test_path_requirements_point_at_real_settings(tmp_path: Path):
    settings = make_settings(tmp_path)
    for capability in CATALOG:
        for requirement in capability.requirements:
            if isinstance(requirement, PathRequirement):
                assert hasattr(settings, requirement.setting_attr), requirement


def test_registry_requirements_use_real_kinds():
    for capability in CATALOG:
        for requirement in capability.requirements:
            if isinstance(requirement, RegistryRequirement):
                assert requirement.kinds
                for kind in requirement.kinds:
                    assert isinstance(kind, ModelKind)


# ---------------------------------------------------------------------------
# Resolucion contra el estado real
# ---------------------------------------------------------------------------


def test_a_missing_pack_leaves_the_capability_needing_setup(tmp_path: Path):
    settings = make_settings(tmp_path, RIFE_BINARY=str(tmp_path / "nope.exe"))
    resolved = resolve_capabilities(settings, FakeRegistry())

    interpolate = by_id(resolved, "video.interpolate")
    assert interpolate.status == "needs_setup"
    assert interpolate.missing_packs == ("rife",)


def test_the_status_follows_the_disk_when_the_pack_appears(tmp_path: Path):
    binary = tmp_path / "rife.exe"
    settings = make_settings(tmp_path, RIFE_BINARY=str(binary))

    before = by_id(resolve_capabilities(settings, FakeRegistry()), "video.interpolate")
    touch(binary)
    after = by_id(resolve_capabilities(settings, FakeRegistry()), "video.interpolate")

    assert before.status == "needs_setup"
    assert after.status == "available"


def test_deleting_the_pack_by_hand_takes_the_capability_back_to_needs_setup(tmp_path: Path):
    # El status se deriva de disco y nunca de un flag persistido, justamente para
    # que borrar la carpeta a mano no deje la UI diciendo que esta listo.
    binary = touch(tmp_path / "rife.exe")
    settings = make_settings(tmp_path, RIFE_BINARY=str(binary))
    assert by_id(resolve_capabilities(settings, FakeRegistry()), "video.interpolate").status == (
        "available"
    )

    binary.unlink()
    assert by_id(resolve_capabilities(settings, FakeRegistry()), "video.interpolate").status == (
        "needs_setup"
    )


def test_an_empty_registry_leaves_generation_needing_setup(tmp_path: Path):
    settings = make_settings(tmp_path)
    resolved = resolve_capabilities(settings, FakeRegistry())

    generation = by_id(resolved, "generate.textToImage")
    assert generation.status == "needs_setup"
    # No falta un paquete, falta un modelo: el aviso tiene que decir eso.
    assert generation.missing_packs == ()
    assert generation.setup_reason_key == "capability.setup.missingModel"


def test_an_installed_diffusion_model_makes_generation_available(tmp_path: Path):
    settings = make_settings(tmp_path)
    registry = FakeRegistry([entry(ModelKind.diffusion_onnx)])

    assert by_id(resolve_capabilities(settings, registry), "generate.textToImage").status == (
        "available"
    )


@pytest.mark.parametrize("status", [ModelStatus.converting, ModelStatus.error])
def test_a_model_that_is_not_usable_yet_does_not_count(tmp_path: Path, status: ModelStatus):
    # Una entrada convirtiendose o en error existe en el registro pero no se
    # puede usar: contarla mostraria la capacidad como lista antes de tiempo.
    settings = make_settings(tmp_path)
    registry = FakeRegistry([entry(ModelKind.diffusion_onnx, status)])

    assert by_id(resolve_capabilities(settings, registry), "generate.textToImage").status == (
        "needs_setup"
    )


def test_upscaling_needs_the_engine_binary_and_not_just_a_model(tmp_path: Path):
    # Las entradas builtin-ncnn se siembran solas en el registro, asi que mirar
    # solo el registro daria `available` en una instalacion sin el binario.
    settings = make_settings(tmp_path, ENGINE_BINARY=str(tmp_path / "nope.exe"))
    registry = FakeRegistry([entry(ModelKind.builtin_ncnn)])

    upscale = by_id(resolve_capabilities(settings, registry), "video.upscale")
    assert upscale.status == "needs_setup"
    assert upscale.missing_packs == ("realesrgan",)


def test_upscaling_needs_a_model_and_not_just_the_binary(tmp_path: Path):
    binary = touch(tmp_path / "engine.exe")
    settings = make_settings(tmp_path, ENGINE_BINARY=str(binary))

    upscale = by_id(resolve_capabilities(settings, FakeRegistry()), "video.upscale")
    assert upscale.status == "needs_setup"
    assert upscale.setup_reason_key == "capability.setup.missingModel"


def test_upscaling_is_available_with_both(tmp_path: Path):
    binary = touch(tmp_path / "engine.exe")
    settings = make_settings(tmp_path, ENGINE_BINARY=str(binary))
    registry = FakeRegistry([entry(ModelKind.onnx)])

    assert by_id(resolve_capabilities(settings, registry), "video.upscale").status == "available"


def test_a_missing_pack_is_reported_before_a_missing_model(tmp_path: Path):
    # Sin el binario no sirve de nada instalar un modelo, asi que el paquete es
    # lo primero que hay que decir.
    settings = make_settings(tmp_path, ENGINE_BINARY=str(tmp_path / "nope.exe"))
    upscale = by_id(resolve_capabilities(settings, FakeRegistry()), "image.upscale")

    assert upscale.setup_reason_key == "capability.setup.missingPack"


def test_an_unimplemented_capability_never_becomes_available(tmp_path: Path):
    # Ni con el registro lleno ni con archivos en disco.
    settings = make_settings(tmp_path)
    registry = FakeRegistry(
        [entry(ModelKind.onnx), entry(ModelKind.diffusion_onnx), entry(ModelKind.builtin_ncnn)]
    )
    resolved = resolve_capabilities(settings, registry)

    for capability in CATALOG:
        if capability.provisioning == "none":
            assert by_id(resolved, capability.id).status == "not_implemented"


def test_resolving_returns_every_capability_of_the_catalog(tmp_path: Path):
    resolved = resolve_capabilities(make_settings(tmp_path), FakeRegistry())
    assert [item.id for item in resolved] == [item.id for item in CATALOG]


# ---------------------------------------------------------------------------
# Agrupado por dominio
# ---------------------------------------------------------------------------


def test_grouping_keeps_the_domain_order(tmp_path: Path):
    grouped = group_by_domain(resolve_capabilities(make_settings(tmp_path), FakeRegistry()))
    assert [group.domain for group in grouped] == list(DOMAIN_ORDER)


def test_grouping_separates_the_roadmap_from_the_live_capabilities(tmp_path: Path):
    grouped = group_by_domain(resolve_capabilities(make_settings(tmp_path), FakeRegistry()))
    generate = next(group for group in grouped if group.domain == "generate")

    # textToVideo salio del roadmap al enviarse en la v0.27.0; dejarlo aca
    # congelaba una mentira en la pantalla de entrada.
    assert {item.id for item in generate.roadmap} >= {
        "generate.imageTo3d",
    }
    assert "generate.textToVideo" not in {item.id for item in generate.roadmap}
    for item in generate.capabilities:
        assert item.status != "not_implemented"
    for item in generate.roadmap:
        assert item.status == "not_implemented"


def test_grouping_loses_no_capability(tmp_path: Path):
    resolved = resolve_capabilities(make_settings(tmp_path), FakeRegistry())
    grouped = group_by_domain(resolved)
    seen = [
        item.id for group in grouped for item in (*group.capabilities, *group.roadmap)
    ]
    assert sorted(seen) == sorted(item.id for item in resolved)


def test_every_domain_carries_its_own_label_key(tmp_path: Path):
    grouped = group_by_domain(resolve_capabilities(make_settings(tmp_path), FakeRegistry()))
    keys = [group.label_key for group in grouped]
    assert keys == [f"capability.domain.{domain}" for domain in DOMAIN_ORDER]
    assert len(set(keys)) == len(keys)


def test_text_to_video_is_not_advertised_as_impossible(tmp_path) -> None:
    """La generación de video por Vulkan está en produccion desde la v0.27.0:
    motor, endpoint, componente del instalador y modo en la pantalla de Generate.
    El arbol seguia diciendo que no habia camino, o sea que la pantalla de
    entrada le mentia al usuario sobre una feature ya enviada."""
    from app.services.capabilities import CATALOG

    cap = next(c for c in CATALOG if c.id == "generate.textToVideo")
    assert cap.job_kind == "generation"
    assert cap.provisioning != "none"
    assert cap.unavailable_reason_key is None
    assert cap.requirements, "sin requisitos, el estado no se puede derivar del disco"


# ---------------------------------------------------------------------------
# El chequeo de impresion no baja nada ni instala nada: es numpy en proceso.
# El catalogo no tenia como decir eso — `provisioning="none"` significa NO
# IMPLEMENTADA, no "funciona sin bajar nada". Sin una forma de expresarlo, una
# capacidad que anda perfecto aparecia en el mapa de ruta.
# ---------------------------------------------------------------------------


class TestPrintDomain:
    def test_the_print_check_is_available_out_of_the_box(self, tmp_path) -> None:
        from app.config import Settings
        from app.services.capabilities import resolve_capabilities
        from app.services.model_registry import ModelRegistry

        settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
        resueltas = resolve_capabilities(settings, ModelRegistry(settings))

        por_id = {c.id: c for c in resueltas}
        assert por_id["print.check"].status == "available"
        assert por_id["print.check"].missing_packs == ()
        # La reparacion tambien: es numpy en proceso, no baja nada.
        assert por_id["print.repair"].status == "available"

    def test_the_print_domain_is_in_the_order(self) -> None:
        from app.services.capabilities import DOMAIN_ORDER

        assert "print" in DOMAIN_ORDER

    def test_the_lanes_that_do_not_exist_yet_say_so(self) -> None:
        # Prometer lo que no esta hecho es peor que no listarlo.
        from app.config import Settings
        from app.services.capabilities import resolve_capabilities
        from app.services.model_registry import ModelRegistry
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(RUNTIME_DIR=tmp, _env_file=None)
            resueltas = resolve_capabilities(settings, ModelRegistry(settings))

        por_id = {c.id: c for c in resueltas}
        assert por_id["print.slice"].status == "not_implemented"
        assert por_id["print.generate"].status == "not_implemented"
