from __future__ import annotations

from pathlib import Path

import pytest

import app.services.pack_provisioner as provisioner_module
from app.config import Settings
from app.services.pack_provisioner import (
    PACK_FIXED_ARGUMENTS,
    PACK_SCRIPTS,
    PROVISION_TIMEOUT_SECONDS,
    PackProvisioner,
    ProvisionStatus,
    UnknownPackError,
    build_command,
    packs_required_by_catalog,
    provisioning_supported,
    script_for,
)
from app.services.license_gate import (
    LICENSE_REQUIRED_KEY,
    LICENSE_UNAVAILABLE_KEY,
    LicenseNotAcceptedError,
    gated_license_path,
    gated_license_text,
    license_gate,
    license_texts_on_disk,
)
from app.services.restore_models import BUNDLE_NAMES, LICENSE_GATED_PACKS


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)


class FakeScript:
    """Sustituye run_guarded_process: registra el comando y elige el resultado."""

    def __init__(self, returncode: int = 0, stderr: bytes = b"") -> None:
        self.commands: list[list[str]] = []
        self.timeouts: list[float] = []
        self.envs: list[dict[str, str] | None] = []
        self.returncode = returncode
        self.stderr = stderr

    async def __call__(self, command, timeout, *, env=None):
        self.commands.append(list(command))
        self.timeouts.append(timeout)
        self.envs.append(env)
        return b"", self.stderr, self.returncode


@pytest.fixture
def script(monkeypatch) -> FakeScript:
    fake = FakeScript()
    monkeypatch.setattr(provisioner_module, "run_guarded_process", fake)
    monkeypatch.setattr(provisioner_module, "provisioning_supported", lambda: True)
    return fake


@pytest.fixture(autouse=True)
def existing_script(monkeypatch, tmp_path: Path):
    # Los scripts reales viven en scripts/ del repo; en los tests se apunta a uno
    # de mentira para no depender del checkout.
    fake = tmp_path / "download-fake.ps1"
    fake.write_text("# noop", encoding="utf-8")
    monkeypatch.setattr(provisioner_module, "script_path", lambda pack: fake)
    return fake


# ---------------------------------------------------------------------------
# El mapeo de paquetes a scripts
# ---------------------------------------------------------------------------


def test_every_pack_the_catalog_requires_has_a_download_script():
    # Sin esto una capacidad podria quedar en needs_setup para siempre, con un
    # boton de descargar que no sabe que correr.
    missing = sorted(packs_required_by_catalog() - set(PACK_SCRIPTS))
    assert missing == []


def test_every_mapped_script_exists_in_the_repo():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    for pack, name in PACK_SCRIPTS.items():
        assert (scripts_dir / name).exists(), f"{pack} -> {name}"


def test_audiosr_is_a_catalog_pack_with_its_script():
    # AudioSR tenia script y label pero ninguna capacidad lo pedia: el boton de
    # descargar existia sin ninguna tarjeta desde donde apretarlo.
    assert "audiosr" in packs_required_by_catalog()
    assert PACK_SCRIPTS["audiosr"] == "download-audiosr-onnx.ps1"


def test_shap_e_img2img_is_a_catalog_pack_with_its_script():
    # Foto a 3D es OTRO repo de pesos que el de texto: sin su propio pack, el
    # boton de descargar bajaria el modelo equivocado.
    assert "shap-e-img2img" in packs_required_by_catalog()
    assert PACK_SCRIPTS["shap-e-img2img"] == "download-shap-e-img2img.ps1"


def test_an_unknown_pack_is_rejected():
    with pytest.raises(UnknownPackError):
        script_for("no-existe")


def test_the_command_runs_the_script_under_powershell(existing_script: Path):
    command = build_command("rife")
    assert command[0] == "powershell"
    # Sin -NoProfile el perfil del usuario puede cambiar el comportamiento, y sin
    # Bypass la politica de ejecucion por defecto rechaza el script.
    assert "-NoProfile" in command
    assert "Bypass" in command
    assert command[-1] == str(existing_script)


# ---------------------------------------------------------------------------
# El job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_successful_run_marks_the_job_done(tmp_path: Path, script: FakeScript):
    provisioner = PackProvisioner(make_settings(tmp_path))
    job_id = await provisioner.provision("rife")

    assert provisioner.status(job_id).status is ProvisionStatus.queued
    await provisioner._process_next()

    assert provisioner.status(job_id).status is ProvisionStatus.done
    assert provisioner.status(job_id).error is None
    assert len(script.commands) == 1


@pytest.mark.asyncio
async def test_the_run_uses_a_generous_timeout(tmp_path: Path, script: FakeScript):
    # Los scripts bajan cientos de MB: el techo es un guard contra un proceso
    # colgado, no una expectativa de duracion.
    provisioner = PackProvisioner(make_settings(tmp_path))
    await provisioner.provision("rife")
    await provisioner._process_next()

    assert script.timeouts == [PROVISION_TIMEOUT_SECONDS]


@pytest.mark.asyncio
async def test_a_nonzero_exit_reports_the_code_and_the_stderr(tmp_path: Path, monkeypatch):
    fake = FakeScript(returncode=3, stderr=b"404 Not Found")
    monkeypatch.setattr(provisioner_module, "run_guarded_process", fake)
    monkeypatch.setattr(provisioner_module, "provisioning_supported", lambda: True)

    provisioner = PackProvisioner(make_settings(tmp_path))
    job_id = await provisioner.provision("rife")
    await provisioner._process_next()

    job = provisioner.status(job_id)
    assert job.status is ProvisionStatus.error
    assert "3" in job.error
    assert "404 Not Found" in job.error


@pytest.mark.asyncio
async def test_a_missing_script_fails_the_job_without_running_anything(
    tmp_path: Path, script: FakeScript, monkeypatch
):
    monkeypatch.setattr(
        provisioner_module, "script_path", lambda pack: tmp_path / "no-existe.ps1"
    )
    provisioner = PackProvisioner(make_settings(tmp_path))
    job_id = await provisioner.provision("rife")
    await provisioner._process_next()

    assert provisioner.status(job_id).status is ProvisionStatus.error
    assert script.commands == []


@pytest.mark.asyncio
async def test_an_unknown_pack_never_becomes_a_job(tmp_path: Path, script: FakeScript):
    # Se valida antes de encolar: un pack desconocido falla la request en vez de
    # aparecer como un job que despues se muere solo.
    provisioner = PackProvisioner(make_settings(tmp_path))
    with pytest.raises(UnknownPackError):
        await provisioner.provision("no-existe")
    assert script.commands == []


@pytest.mark.asyncio
async def test_an_unsupported_platform_says_so_instead_of_failing_oddly(
    tmp_path: Path, script: FakeScript, monkeypatch
):
    monkeypatch.setattr(provisioner_module, "provisioning_supported", lambda: False)
    provisioner = PackProvisioner(make_settings(tmp_path))
    job_id = await provisioner.provision("rife")
    await provisioner._process_next()

    job = provisioner.status(job_id)
    assert job.status is ProvisionStatus.error
    assert "Windows" in job.error
    assert script.commands == []


def test_provisioning_is_only_supported_on_windows():
    assert provisioning_supported("win32") is True
    assert provisioning_supported("linux") is False
    assert provisioning_supported("darwin") is False


@pytest.mark.asyncio
async def test_an_unexpected_exception_lands_in_the_job_and_not_in_the_worker(
    tmp_path: Path, monkeypatch
):
    async def explode(command, timeout, *, env=None):
        raise OSError("disco lleno")

    monkeypatch.setattr(provisioner_module, "run_guarded_process", explode)
    monkeypatch.setattr(provisioner_module, "provisioning_supported", lambda: True)

    provisioner = PackProvisioner(make_settings(tmp_path))
    job_id = await provisioner.provision("rife")
    await provisioner._process_next()

    job = provisioner.status(job_id)
    assert job.status is ProvisionStatus.error
    assert "disco lleno" in job.error


@pytest.mark.asyncio
async def test_status_of_an_unknown_job_is_none(tmp_path: Path):
    assert PackProvisioner(make_settings(tmp_path)).status("nope") is None


@pytest.mark.asyncio
async def test_two_packs_queue_and_both_run(tmp_path: Path, script: FakeScript):
    provisioner = PackProvisioner(make_settings(tmp_path))
    first = await provisioner.provision("rife")
    second = await provisioner.provision("apollo")

    assert await provisioner._process_next() is True
    assert await provisioner._process_next() is True
    assert await provisioner._process_next() is False

    assert provisioner.status(first).status is ProvisionStatus.done
    assert provisioner.status(second).status is ProvisionStatus.done


# La invariante de arriba va en UN solo sentido: que todo pack pedido por el
# catalogo tenga script. Faltaba la inversa, y por ese hueco se colaron tres
# descargas que solo se podian hacer desde la terminal: el modelo de voz, la
# conversion de voz y la traduccion. El usuario las veia como texto crudo
# diciendole que corriera un .ps1, sin boton.
#
# Los aceleradores por hardware quedan afuera A PROPOSITO y por escrito: no son
# un pack de una capacidad, son un proveedor de ejecucion atado al chip.
SCRIPTS_SIN_PACK_A_PROPOSITO = {
    "download-openvino-ep.ps1",
    "download-tensorrt-rtx-ep.ps1",
}


def test_every_download_script_is_reachable_from_a_button():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    existentes = {p.name for p in scripts_dir.glob("download-*.ps1")}
    huerfanos = sorted(existentes - set(PACK_SCRIPTS.values()) - SCRIPTS_SIN_PACK_A_PROPOSITO)
    assert huerfanos == [], (
        "Estos scripts no se pueden correr desde la app, asi que el usuario "
        f"queda obligado a abrir una terminal: {huerfanos}"
    )


class TestPacksConVariante:
    """La traduccion no es UN modelo: es uno por par de idiomas.

    El script lo recibe por `-Pair`. Sin poder pasarle ese valor, el boton solo
    podria bajar el par por defecto y el usuario quedaria otra vez obligado a
    abrir una terminal para cualquier otro idioma — que es exactamente lo que
    este mecanismo existe para evitar.
    """

    def test_la_variante_viaja_al_script(self):
        comando = build_command("translation", variant="es-en")

        assert comando[-2:] == ["-Pair", "es-en"]

    def test_sin_variante_el_comando_queda_como_siempre(self):
        assert "-Pair" not in build_command("translation")

    def test_un_pack_sin_parametro_rechaza_la_variante(self):
        # Pasarle una variante a un pack que no la entiende seria mandarle un
        # argumento que el script no declara: PowerShell lo tomaria como ruta.
        with pytest.raises(ValueError):
            build_command("rife", variant="es-en")

    def test_una_variante_con_forma_rara_se_rechaza_antes_de_correr_nada(self):
        # El valor llega desde una peticion HTTP y termina en una linea de
        # comandos.
        with pytest.raises(ValueError):
            build_command("translation", variant="es-en; rm -rf /")


async def test_el_script_corre_sin_psmodulepath_heredado(tmp_path, script, monkeypatch):
    # Pasado real (2026-08-09): server relanzado desde pwsh 7 -> el powershell
    # 5.1 hijo heredaba un PSModulePath ajeno y Get-FileHash dejaba de resolver.
    # Sin la variable, 5.1 reconstruye su default completo.
    monkeypatch.setenv("PSModulePath", "C:\\ruta\\de\\pwsh7\\rota")
    provisioner = PackProvisioner(make_settings(tmp_path))
    job_id = await provisioner.provision("rife")
    await provisioner._process_next()

    assert script.envs, "run_guarded_process no recibio env"
    env = script.envs[0]
    assert env is not None
    assert "PSModulePath" not in env
    assert provisioner.status(job_id).status is ProvisionStatus.done


class TestLosPacksDeRestauracion:
    """Tres packs, un solo script: el bundle viaja como argumento fijo.

    No es una variante que elige el usuario: cada pack ES un bundle, asi que el
    argumento sale del pack y nunca de la peticion HTTP.
    """

    @pytest.mark.parametrize("bundle", BUNDLE_NAMES)
    def test_cada_bundle_es_un_pack_que_corre_download_restore(self, bundle):
        assert PACK_SCRIPTS[f"restore-{bundle}"] == "download-restore.ps1"

    @pytest.mark.parametrize("bundle", BUNDLE_NAMES)
    def test_el_bundle_viaja_como_argumento_fijo(self, bundle, existing_script: Path):
        comando = build_command(f"restore-{bundle}")

        assert comando[-3:] == [str(existing_script), "-Bundle", bundle]

    def test_los_argumentos_fijos_son_de_packs_conocidos(self):
        assert set(PACK_FIXED_ARGUMENTS) <= set(PACK_SCRIPTS)
        assert set(PACK_FIXED_ARGUMENTS) == {f"restore-{b}" for b in BUNDLE_NAMES}

    def test_un_pack_de_restauracion_no_acepta_variantes(self):
        with pytest.raises(ValueError):
            build_command("restore-core", variant="faces")

    def test_los_demas_packs_no_reciben_argumentos_fijos(self, existing_script: Path):
        assert build_command("rife")[-1] == str(existing_script)

    async def test_el_job_corre_el_script_con_su_bundle(self, tmp_path, script):
        provisioner = PackProvisioner(make_settings(tmp_path))
        job_id = await provisioner.provision("restore-faces")
        await provisioner._process_next()

        assert provisioner.status(job_id).status is ProvisionStatus.done
        assert script.commands[0][-2:] == ["-Bundle", "faces"]


class TestLicenseGate:
    """Un pack no comercial no se baja sin que el usuario lea y acepte su licencia.

    La compuerta vive en el provisioner y no solo en la UI: cualquier cliente de la
    API (o un boton viejo) choca contra ella.
    """

    GATED = frozenset({"rife"})

    def gated_provisioner(self, tmp_path: Path, text: str | None = "NON-COMMERCIAL LICENSE\n") -> PackProvisioner:
        licenses = tmp_path / "gated"
        licenses.mkdir(exist_ok=True)
        if text is not None:
            (licenses / "rife.txt").write_text(text, encoding="utf-8")
        return PackProvisioner(make_settings(tmp_path), gated_packs=self.GATED, gated_licenses_dir=licenses)

    async def test_a_gated_pack_without_accepting_the_license_never_becomes_a_job(self, tmp_path, script):
        provisioner = self.gated_provisioner(tmp_path)

        with pytest.raises(LicenseNotAcceptedError) as caught:
            await provisioner.provision("rife")

        assert caught.value.key == LICENSE_REQUIRED_KEY
        assert await provisioner._process_next() is False
        assert script.commands == []

    async def test_a_gated_pack_with_the_license_accepted_runs(self, tmp_path, script):
        provisioner = self.gated_provisioner(tmp_path)

        job_id = await provisioner.provision("rife", accept_license=True)
        await provisioner._process_next()

        assert provisioner.status(job_id).status is ProvisionStatus.done
        assert len(script.commands) == 1

    async def test_a_gated_pack_whose_license_text_is_missing_is_refused_even_if_accepted(self, tmp_path, script):
        # Aceptar un texto que nadie pudo leer no es consentimiento: falla cerrado.
        provisioner = self.gated_provisioner(tmp_path, text=None)

        with pytest.raises(LicenseNotAcceptedError) as caught:
            await provisioner.provision("rife", accept_license=True)

        assert caught.value.key == LICENSE_UNAVAILABLE_KEY
        assert script.commands == []

    async def test_a_blank_license_text_counts_as_missing(self, tmp_path, script):
        provisioner = self.gated_provisioner(tmp_path, text="  \n")

        with pytest.raises(LicenseNotAcceptedError) as caught:
            await provisioner.provision("rife", accept_license=True)

        assert caught.value.key == LICENSE_UNAVAILABLE_KEY

    async def test_a_pack_outside_the_gate_needs_no_license(self, tmp_path, script):
        provisioner = self.gated_provisioner(tmp_path)

        job_id = await provisioner.provision("apollo")
        await provisioner._process_next()

        assert provisioner.status(job_id).status is ProvisionStatus.done

    def test_the_license_error_is_not_a_value_error(self):
        # La ruta traduce ValueError a 400: la compuerta tiene que llegar como 403.
        assert not issubclass(LicenseNotAcceptedError, ValueError)

    def test_the_default_provisioner_gates_the_catalog_license_gated_packs(self, tmp_path):
        provisioner = PackProvisioner(make_settings(tmp_path))

        assert provisioner._gated_packs == LICENSE_GATED_PACKS

    def test_every_license_gated_pack_ships_its_full_license_text(self):
        missing = sorted(pack for pack in LICENSE_GATED_PACKS if gated_license_text(pack) is None)
        assert missing == [], f"Gated packs without app/licenses/gated/<pack>.txt: {missing}"

    def test_every_gated_license_text_belongs_to_a_gated_pack_with_a_script(self):
        orphans = sorted(license_texts_on_disk() - LICENSE_GATED_PACKS)
        assert orphans == []
        assert set(LICENSE_GATED_PACKS) <= set(PACK_SCRIPTS)


class TestLicenseGateView:
    def test_an_ungated_pack_reports_no_gate_and_no_text(self, tmp_path):
        assert license_gate("rife", frozenset(), tmp_path) == {"pack": "rife", "gated": False, "licenseText": None}

    def test_a_gated_pack_reports_its_full_license_text(self, tmp_path):
        (tmp_path / "rife.txt").write_text("S-Lab License 1.0\n\nFull text.\n", encoding="utf-8")

        gate = license_gate("rife", frozenset({"rife"}), tmp_path)

        assert gate == {"pack": "rife", "gated": True, "licenseText": "S-Lab License 1.0\n\nFull text.\n"}

    def test_a_gated_pack_without_its_text_reports_none(self, tmp_path):
        assert license_gate("rife", frozenset({"rife"}), tmp_path)["licenseText"] is None

    def test_the_license_text_path_is_named_after_the_pack(self, tmp_path):
        assert gated_license_path("restore-faces-nc", tmp_path) == tmp_path / "restore-faces-nc.txt"

    def test_no_texts_on_disk_when_the_directory_does_not_exist(self, tmp_path):
        assert license_texts_on_disk(tmp_path / "missing") == frozenset()
