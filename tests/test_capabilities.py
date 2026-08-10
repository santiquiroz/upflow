from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings, resolve_against_project_root
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


def _find(resueltas, capacidad_id: str):
    return next(c for c in resueltas if c.id == capacidad_id)


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


# Capacidades que necesitan un modelo bajado pero NO encolan: responden en la
# misma peticion porque son lo bastante rapidas, y encolarlas costaria mas que
# hacerlas. Medido: 1,90 s de habla salen en menos de medio segundo.
SINCRONICAS = {"audio.speak", "audio.voiceConvert"}


def test_every_queued_capability_declares_a_job_kind():
    # `builtin` queda afuera a proposito: es codigo sincronico en proceso, no
    # trabajo que se encole, y exigirle un job_kind seria pedirle una cola que no
    # tiene. Las de SINCRONICAS quedan afuera por lo mismo, aunque si necesiten
    # bajar un modelo: el provisioning dice de donde sale el modelo, no si el
    # trabajo se encola.
    for capability in CATALOG:
        if capability.id in SINCRONICAS:
            continue
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


def rife_settings(tmp_path: Path) -> Settings:
    # Las dos rutas apuntadas al tmp: la interpolacion necesita el binario Y la
    # carpeta del modelo, y dejar los defaults haria que la prueba dependiera de
    # lo que este bajado en el checkout.
    return make_settings(
        tmp_path,
        RIFE_BINARY=str(tmp_path / "rife.exe"),
        RIFE_MODELS_DIR=str(tmp_path / "rife-models"),
    )


def test_a_missing_pack_leaves_the_capability_needing_setup(tmp_path: Path):
    settings = rife_settings(tmp_path)
    resolved = resolve_capabilities(settings, FakeRegistry())

    interpolate = by_id(resolved, "video.interpolate")
    assert interpolate.status == "needs_setup"
    assert interpolate.missing_packs == ("rife",)


def test_the_status_follows_the_disk_when_the_pack_appears(tmp_path: Path):
    settings = rife_settings(tmp_path)

    before = by_id(resolve_capabilities(settings, FakeRegistry()), "video.interpolate")
    touch(Path(settings.rife_binary))
    settings.rife_default_model_path.mkdir(parents=True, exist_ok=True)
    after = by_id(resolve_capabilities(settings, FakeRegistry()), "video.interpolate")

    assert before.status == "needs_setup"
    assert after.status == "available"


def test_the_binary_without_the_model_folder_is_not_enough(tmp_path: Path):
    # interpolation_available() exige la carpeta del modelo configurado ademas
    # del binario: sin declararla, un install a medias dejaba la tarjeta en verde
    # y el job manager rechazaba el trabajo despues.
    settings = rife_settings(tmp_path)
    touch(Path(settings.rife_binary))

    assert by_id(resolve_capabilities(settings, FakeRegistry()), "video.interpolate").status == (
        "needs_setup"
    )


def test_deleting_the_pack_by_hand_takes_the_capability_back_to_needs_setup(tmp_path: Path):
    # El status se deriva de disco y nunca de un flag persistido, justamente para
    # que borrar la carpeta a mano no deje la UI diciendo que esta listo.
    settings = rife_settings(tmp_path)
    binary = touch(Path(settings.rife_binary))
    settings.rife_default_model_path.mkdir(parents=True, exist_ok=True)
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
        # La generacion SI existe, pero necesita bajar el modelo: `needs_setup`
        # y no `not_implemented`. La diferencia importa — una dice "falta bajar
        # algo" y la otra "no esta hecho", y confundirlas manda al usuario a
        # buscar un boton que no existe.
        assert por_id["print.generate"].status == "needs_setup"
        assert "shap-e" in por_id["print.generate"].missing_packs


# ---------------------------------------------------------------------------
# Foto a 3D: es OTRO repo de pesos que el de texto (openai/shap-e-img2img), asi
# que tiene su propia tarjeta con su propio pack. Confundirlos haria que el
# boton de descargar baje el modelo equivocado.
# ---------------------------------------------------------------------------


class TestFotoA3dEnElArbol:
    def test_sin_el_pack_pide_bajarlo(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path / "runtime")

        foto = _find(resolve_capabilities(settings, FakeRegistry()), "print.generatePhoto")

        assert foto.status == "needs_setup"
        assert "shap-e-img2img" in foto.missing_packs
        assert foto.setup_reason_key == "capability.setup.missingPack"

    def test_el_pack_de_texto_no_habilita_el_de_foto(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path / "runtime")
        touch(tmp_path / "vendor" / "shap-e" / "model_index.json")

        foto = _find(resolve_capabilities(settings, FakeRegistry()), "print.generatePhoto")

        assert foto.status == "needs_setup"

    def test_la_carpeta_sola_sin_indice_no_alcanza(self, tmp_path: Path) -> None:
        # Una descarga a medias deja la carpeta existiendo: el requisito apunta
        # al indice, que es lo que la tuberia necesita para cargar.
        settings = make_settings(tmp_path / "runtime")
        (tmp_path / "vendor" / "shap-e-img2img").mkdir(parents=True)

        foto = _find(resolve_capabilities(settings, FakeRegistry()), "print.generatePhoto")

        assert foto.status == "needs_setup"

    def test_con_el_indice_queda_disponible(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path / "runtime")
        touch(tmp_path / "vendor" / "shap-e-img2img" / "model_index.json")

        foto = _find(resolve_capabilities(settings, FakeRegistry()), "print.generatePhoto")

        assert foto.status == "available"


# ---------------------------------------------------------------------------
# El carril CAD necesita DOS cosas que no son un pack: OpenSCAD instalado y un
# servidor de modelo local apuntado. Ninguna se baja, las dos las pone el
# usuario, y decir "disponible" sin ellas manda a la pantalla a fallar en el
# medio en vez de avisar antes.
# ---------------------------------------------------------------------------


class TestElCarrilCad:
    def test_sin_servidor_de_modelo_no_esta_disponible(self, tmp_path) -> None:
        settings = Settings(RUNTIME_DIR=str(tmp_path), CAD_LLM_BASE_URL="", _env_file=None)

        cad = _find(resolve_capabilities(settings, FakeRegistry()), "print.cad")

        assert cad.status != "available"

    def test_con_las_dos_cosas_puestas_esta_disponible(self, tmp_path) -> None:
        binario = tmp_path / "openscad.exe"
        binario.write_bytes(b"x")
        settings = Settings(
            RUNTIME_DIR=str(tmp_path),
            CAD_LLM_BASE_URL="http://localhost:11434/v1",
            OPENSCAD_BINARY_PATH=str(binario),
            _env_file=None,
        )

        cad = _find(resolve_capabilities(settings, FakeRegistry()), "print.cad")

        assert cad.status == "available"

    def test_dice_que_falta_configurar_no_que_falta_bajar(self, tmp_path) -> None:
        # Mandar al usuario a bajar un pack que no existe es peor que no decir
        # nada: lo que falta lo configura el, no un boton de descarga.
        binario = tmp_path / "openscad.exe"
        binario.write_bytes(b"x")
        settings = Settings(
            RUNTIME_DIR=str(tmp_path),
            CAD_LLM_BASE_URL="",
            OPENSCAD_BINARY_PATH=str(binario),
            _env_file=None,
        )

        cad = _find(resolve_capabilities(settings, FakeRegistry()), "print.cad")

        assert cad.setup_reason_key == "capability.setup.missingSetting"
        assert cad.missing_packs == ()


# ---------------------------------------------------------------------------
# La voz no figuraba en el arbol. La pantalla de Tareas es donde el usuario elige
# que hacer, y "convertir texto en habla" simplemente no estaba: para descubrirlo
# habia que entrar a la seccion Voz y encontrarse con un texto muerto.
# ---------------------------------------------------------------------------


class TestLaVozEstaEnElArbol:
    def test_la_sintesis_de_voz_es_una_capacidad(self, tmp_path) -> None:
        settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)

        assert _find(resolve_capabilities(settings, FakeRegistry()), "audio.speak")

    def test_la_conversion_de_voz_es_una_capacidad(self, tmp_path) -> None:
        settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)

        assert _find(resolve_capabilities(settings, FakeRegistry()), "audio.voiceConvert")

    def test_sin_el_modelo_pide_bajar_el_paquete_de_voz(self, tmp_path) -> None:
        # Que diga que falta un pack es lo que le da el boton a la pantalla de
        # Tareas; sin eso vuelve a ser un cartel sin salida.
        settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)

        voz = _find(resolve_capabilities(settings, FakeRegistry()), "audio.speak")

        assert voz.status != "available"
        assert "kokoro" in voz.missing_packs

    def test_con_el_modelo_puesto_queda_disponible(self, tmp_path) -> None:
        modelo = tmp_path / "vendor" / "kokoro"
        touch(modelo / "model.onnx")
        touch(modelo / "tokenizer.json")
        settings = Settings(RUNTIME_DIR=str(tmp_path / "runtime"), _env_file=None)

        voz = _find(resolve_capabilities(settings, FakeRegistry()), "audio.speak")

        assert voz.status == "available"

    def test_la_carpeta_vacia_no_alcanza(self, tmp_path) -> None:
        # El script crea vendor\kokoro ANTES de bajar nada, asi que hasta una
        # descarga fallida dejaba la tarjeta en verde con el directorio vacio.
        (tmp_path / "vendor" / "kokoro").mkdir(parents=True, exist_ok=True)
        settings = Settings(RUNTIME_DIR=str(tmp_path / "runtime"), _env_file=None)

        voz = _find(resolve_capabilities(settings, FakeRegistry()), "audio.speak")

        assert voz.status == "needs_setup"
        assert "kokoro" in voz.missing_packs

    def test_el_modelo_sin_el_tokenizador_no_alcanza(self, tmp_path) -> None:
        # KokoroTtsEngine.available() exige los dos, y tokenizer.json solo se
        # bajaba dentro del mismo if que el modelo: una instalacion con uno y sin
        # el otro no se arreglaba sola y la tarjeta decia que si.
        touch(tmp_path / "vendor" / "kokoro" / "model.onnx")
        settings = Settings(RUNTIME_DIR=str(tmp_path / "runtime"), _env_file=None)

        voz = _find(resolve_capabilities(settings, FakeRegistry()), "audio.speak")

        assert voz.status == "needs_setup"


# ---------------------------------------------------------------------------
# La conversion de voz necesita TRES piezas y el catalogo declaraba UNA. El
# usuario bajaba el pack, la tarjeta decia "disponible" y la pantalla de Voz le
# contestaba "falta el modelo de conversion de voz": el script no bajaba el
# encoder de x-vector y aun asi terminaba en 0.
# ---------------------------------------------------------------------------


def _vendor_de(settings: Settings) -> Path:
    return Path(settings.runtime_dir).parent / "vendor"


def _poner_piezas(settings: Settings, *, vc=True, vocoder=True, xvector=True) -> None:
    vendor = _vendor_de(settings)
    if vc:
        (vendor / "speecht5-vc").mkdir(parents=True, exist_ok=True)
    if vocoder:
        (vendor / "speecht5-hifigan").mkdir(parents=True, exist_ok=True)
    if xvector:
        touch(vendor / "xvector" / "tdnn.onnx")


class TestLaConversionDeVozPideLasTresPiezas:
    def test_el_catalogo_declara_todo_lo_que_el_motor_exige(self, tmp_path: Path) -> None:
        # La prueba de coherencia: si el motor empieza a necesitar otra pieza y
        # nadie la declara en el CATALOG, la tarjeta vuelve a mentir. Comparar
        # rutas resueltas y no nombres es lo que hace que no se pueda fingir.
        from app.services.engines.voice_convert import VoiceConversionEngine

        settings = make_settings(tmp_path / "runtime")
        capacidad = next(c for c in CATALOG if c.id == "audio.voiceConvert")
        declaradas = {
            resolve_against_project_root(str(getattr(settings, r.setting_attr)))
            for r in capacidad.requirements
            if isinstance(r, PathRequirement)
        }

        exigidas = set(VoiceConversionEngine(_vendor_de(settings)).required_paths())

        assert exigidas <= declaradas, (
            "El motor exige rutas que el catalogo no declara, asi que la tarjeta "
            f"puede decir 'disponible' sin ellas: {sorted(exigidas - declaradas)}"
        )

    @pytest.mark.parametrize("falta", ["vc", "vocoder", "xvector"])
    def test_sin_cualquiera_de_las_tres_no_esta_disponible(
        self, tmp_path: Path, falta: str
    ) -> None:
        settings = make_settings(tmp_path / "runtime")
        _poner_piezas(settings, **{falta: False})

        conversion = _find(resolve_capabilities(settings, FakeRegistry()), "audio.voiceConvert")

        assert conversion.status == "needs_setup"
        assert conversion.missing_packs == ("voice-conversion",)

    def test_con_las_tres_queda_disponible(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path / "runtime")
        _poner_piezas(settings)

        conversion = _find(resolve_capabilities(settings, FakeRegistry()), "audio.voiceConvert")

        assert conversion.status == "available"

    def test_la_carpeta_del_xvector_sin_el_modelo_no_alcanza(self, tmp_path: Path) -> None:
        # Una descarga cortada deja el directorio existiendo: el requisito apunta
        # al archivo, que es lo que el encoder necesita para cargar.
        settings = make_settings(tmp_path / "runtime")
        _poner_piezas(settings, xvector=False)
        (_vendor_de(settings) / "xvector").mkdir(parents=True, exist_ok=True)

        conversion = _find(resolve_capabilities(settings, FakeRegistry()), "audio.voiceConvert")

        assert conversion.status == "needs_setup"

    def test_la_tarjeta_y_el_motor_dicen_lo_mismo(self, tmp_path: Path) -> None:
        # Lo que el usuario reporto: la tarjeta decia disponible y el endpoint
        # decia que no. Las dos respuestas salen ahora del mismo criterio.
        from app.services.engines.voice_convert import VoiceConversionEngine

        settings = make_settings(tmp_path / "runtime")
        motor = VoiceConversionEngine(_vendor_de(settings))

        for faltante in (None, "vc", "vocoder", "xvector"):
            for sobrante in _vendor_de(settings).glob("*"):
                if sobrante.is_dir():
                    for hijo in sorted(sobrante.rglob("*"), reverse=True):
                        hijo.unlink() if hijo.is_file() else hijo.rmdir()
                    sobrante.rmdir()
            _poner_piezas(settings, **({faltante: False} if faltante else {}))

            tarjeta = _find(
                resolve_capabilities(settings, FakeRegistry()), "audio.voiceConvert"
            )
            assert (tarjeta.status == "available") is motor.available(), faltante


# ---------------------------------------------------------------------------
# El mismo barrido encontro dos tarjetas mas que se derivaban de menos de lo que
# su motor exige, y una que apuntaba directamente al pack equivocado.
# ---------------------------------------------------------------------------


class TestOtrasTarjetasQueSeDerivabanDeMenos:
    def test_la_generacion_3d_pide_el_indice_y_no_la_carpeta(self, tmp_path: Path) -> None:
        # Una descarga a medias deja vendor\shap-e existiendo: Shape3dEngine
        # carga model_index.json, asi que es eso lo que hay que exigir.
        settings = make_settings(tmp_path / "runtime")
        (tmp_path / "vendor" / "shap-e").mkdir(parents=True)

        pieza = _find(resolve_capabilities(settings, FakeRegistry()), "print.generate")

        assert pieza.status == "needs_setup"

        touch(tmp_path / "vendor" / "shap-e" / "model_index.json")
        pieza = _find(resolve_capabilities(settings, FakeRegistry()), "print.generate")
        assert pieza.status == "available"

    def test_el_video_generado_necesita_el_motor_Y_los_pesos(self, tmp_path: Path) -> None:
        # Son dos packs distintos: `sdcpp` trae sd-cli.exe y `wan-video` los
        # pesos. La tarjeta pedia el binario contra el pack de pesos, asi que
        # apretar su boton bajaba 16 GB y la dejaba igual de roja.
        binario = tmp_path / "sd-cli.exe"
        settings = make_settings(
            tmp_path / "runtime",
            SDCPP_BINARY=str(binario),
            SDCPP_MODELS_DIR=str(tmp_path / "sdcpp-models"),
        )

        solo_nada = _find(resolve_capabilities(settings, FakeRegistry()), "generate.textToVideo")
        assert solo_nada.status == "needs_setup"
        assert set(solo_nada.missing_packs) == {"sdcpp", "wan-video"}

        touch(binario)
        solo_motor = _find(resolve_capabilities(settings, FakeRegistry()), "generate.textToVideo")
        assert solo_motor.status == "needs_setup"
        assert solo_motor.missing_packs == ("wan-video",)

        settings.sdcpp_video_models_dir_path.mkdir(parents=True, exist_ok=True)
        completo = _find(resolve_capabilities(settings, FakeRegistry()), "generate.textToVideo")
        assert completo.status == "available"


def test_every_path_requirement_names_a_pack_that_can_produce_it() -> None:
    # La invariante que faltaba: no alcanza con que el pack tenga script, tiene
    # que ser el script que produce ESA ruta. `generate.textToVideo` pedia el
    # binario de sd.cpp contra el pack de pesos de video, que nunca lo baja.
    from app.services.pack_provisioner import PACK_SCRIPTS

    esperado = {
        "sdcpp_binary": "sdcpp",
        "sdcpp_video_models_dir_path": "wan-video",
        "voice_conversion_xvector_path": "voice-conversion",
        "kokoro_model_file": "kokoro",
        "rife_default_model_path": "rife",
    }
    declarado = {
        requirement.setting_attr: requirement.pack
        for capability in CATALOG
        for requirement in capability.requirements
        if isinstance(requirement, PathRequirement)
    }

    for atributo, pack in esperado.items():
        assert declarado.get(atributo) == pack, atributo
        assert pack in PACK_SCRIPTS, pack


def test_a_pack_that_covers_several_pieces_is_offered_once(tmp_path: Path) -> None:
    # Tres requisitos del mismo pack no son tres botones de descarga.
    settings = make_settings(tmp_path / "runtime")

    conversion = _find(resolve_capabilities(settings, FakeRegistry()), "audio.voiceConvert")

    assert conversion.missing_packs == ("voice-conversion",)


# ---------------------------------------------------------------------------
# AudioSR estaba escondido detras de ENABLE_AUDIOSR + un script manual: no
# aparecia en el arbol y habia que saber que existia. Ahora es una tarjeta como
# las demas: el boton baja el pack, y el flag pendiente se dice como
# configuracion faltante en vez de mentir "disponible" con jobs que fallarian.
# ---------------------------------------------------------------------------


class TestAudioSrEnElArbol:
    def test_sin_los_modelos_pide_bajar_el_paquete(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path, AUDIOSR_MODEL_DIR=str(tmp_path / "audiosr"))

        restore_sr = _find(resolve_capabilities(settings, FakeRegistry()), "audio.restoreSr")

        assert restore_sr.status == "needs_setup"
        assert "audiosr" in restore_sr.missing_packs
        # El paquete manda sobre el flag: sin los modelos no sirve de nada
        # prender ENABLE_AUDIOSR, asi que lo primero que se ofrece es descargar.
        assert restore_sr.setup_reason_key == "capability.setup.missingPack"

    def test_con_los_modelos_pero_sin_el_flag_dice_que_falta_configurar(
        self, tmp_path: Path
    ) -> None:
        model_dir = tmp_path / "audiosr"
        model_dir.mkdir()
        settings = make_settings(tmp_path, AUDIOSR_MODEL_DIR=str(model_dir))

        restore_sr = _find(resolve_capabilities(settings, FakeRegistry()), "audio.restoreSr")

        assert restore_sr.status == "needs_setup"
        assert restore_sr.missing_packs == ()
        assert restore_sr.setup_reason_key == "capability.setup.missingSetting"

    def test_con_los_modelos_y_el_flag_queda_disponible(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "audiosr"
        model_dir.mkdir()
        settings = make_settings(
            tmp_path, AUDIOSR_MODEL_DIR=str(model_dir), ENABLE_AUDIOSR=True
        )

        restore_sr = _find(resolve_capabilities(settings, FakeRegistry()), "audio.restoreSr")

        assert restore_sr.status == "available"
