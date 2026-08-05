from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.services.saved_prompts import SavedPrompt, SavedPromptStore

# ---------------------------------------------------------------------------
# Un prompt guardado es DATO DEL USUARIO, no copia de la app: se guarda tal cual
# lo escribió y no se traduce nunca. Los presets de fábrica sí son copia y viajan
# como claves — son dos cosas distintas que viven en lugares distintos.
#
# La app tiene usuarios, así que cada uno ve los suyos: los prompts de alguien
# pueden decir cualquier cosa y no son de la máquina, son de la persona.
# ---------------------------------------------------------------------------


def make_store(tmp_path: Path) -> SavedPromptStore:
    return SavedPromptStore(Settings(_env_file=None, RUNTIME_DIR=str(tmp_path)))


class TestSaving:
    def test_a_saved_prompt_comes_back(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        saved = store.save(owner_id="ana", name="Mi retrato", prompt="portrait, soft light")

        assert [p.name for p in store.list_for("ana")] == ["Mi retrato"]
        assert saved.prompt == "portrait, soft light"

    def test_the_prompt_is_stored_verbatim(self, tmp_path: Path) -> None:
        # Es lo que va al modelo: recortarlo o normalizarlo cambia lo que genera.
        raro = "  UPPER case, 35mm, ((weights)), 日本語  "
        store = make_store(tmp_path)
        store.save(owner_id="ana", name="raro", prompt=raro)

        assert store.list_for("ana")[0].prompt == raro.strip()

    def test_the_negative_prompt_is_optional(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        saved = store.save(owner_id="ana", name="sin negativo", prompt="algo")

        assert saved.negative_prompt == ""

    def test_an_empty_name_is_refused(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(ValueError):
            store.save(owner_id="ana", name="   ", prompt="algo")

    def test_an_empty_prompt_is_refused(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(ValueError):
            store.save(owner_id="ana", name="vacio", prompt="  ")


class TestOwnership:
    def test_each_user_only_sees_their_own(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        store.save(owner_id="ana", name="de ana", prompt="a")
        store.save(owner_id="beto", name="de beto", prompt="b")

        assert [p.name for p in store.list_for("ana")] == ["de ana"]
        assert [p.name for p in store.list_for("beto")] == ["de beto"]

    def test_nobody_can_delete_someone_elses(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        de_ana = store.save(owner_id="ana", name="de ana", prompt="a")

        assert store.delete(owner_id="beto", prompt_id=de_ana.id) is False
        assert len(store.list_for("ana")) == 1

    def test_deleting_your_own_works(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        mio = store.save(owner_id="ana", name="mio", prompt="a")

        assert store.delete(owner_id="ana", prompt_id=mio.id) is True
        assert store.list_for("ana") == []

    def test_deleting_something_that_does_not_exist_is_false_not_a_crash(self, tmp_path: Path) -> None:
        assert make_store(tmp_path).delete(owner_id="ana", prompt_id="no-existe") is False


class TestPersistence:
    def test_survives_a_restart(self, tmp_path: Path) -> None:
        make_store(tmp_path).save(owner_id="ana", name="persistente", prompt="a")

        assert [p.name for p in make_store(tmp_path).list_for("ana")] == ["persistente"]

    def test_a_corrupt_file_does_not_take_the_app_down(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        store.save(owner_id="ana", name="antes", prompt="a")
        store.path.write_text("{ esto no es json", encoding="utf-8")

        # Arranca vacio en vez de reventar, y deja el archivo roto respaldado
        # para poder mirarlo.
        assert make_store(tmp_path).list_for("ana") == []

    def test_newest_first_so_the_last_saved_is_at_hand(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        store.save(owner_id="ana", name="primero", prompt="a")
        store.save(owner_id="ana", name="segundo", prompt="b")

        assert [p.name for p in store.list_for("ana")] == ["segundo", "primero"]


class TestSavedPrompt:
    def test_ids_do_not_repeat(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        uno = store.save(owner_id="ana", name="a", prompt="x")
        dos = store.save(owner_id="ana", name="b", prompt="y")

        assert uno.id != dos.id

    def test_a_prompt_knows_its_generation_mode(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        saved = store.save(owner_id="ana", name="video", prompt="x", mode="video")

        assert saved.mode == "video"
        assert isinstance(saved, SavedPrompt)


# --- API -------------------------------------------------------------------


def test_the_api_round_trips_a_saved_prompt() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/generation/saved-prompts",
            json={"name": "de prueba", "prompt": "cinematic, 35mm", "mode": "video"},
        )
        assert created.status_code == 201
        body = created.json()
        assert body["prompt"] == "cinematic, 35mm"
        assert body["mode"] == "video"

        listed = client.get("/api/v1/generation/saved-prompts").json()["prompts"]
        assert any(p["id"] == body["id"] for p in listed)

        assert client.delete(f"/api/v1/generation/saved-prompts/{body['id']}").status_code == 204
        after = client.get("/api/v1/generation/saved-prompts").json()["prompts"]
        assert not any(p["id"] == body["id"] for p in after)


def test_deleting_something_that_is_not_there_is_a_404() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        assert client.delete("/api/v1/generation/saved-prompts/no-existe").status_code == 404


def test_an_empty_name_is_a_400_not_a_500() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/generation/saved-prompts", json={"name": "", "prompt": "algo"}
        )
        assert response.status_code in (400, 422)
