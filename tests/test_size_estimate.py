from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from app.api.routes import create_shape3d_job, estimate_print_size, shape3d_job_to_response
from app.config import Settings
from app.schemas import Shape3dJobRequest, SizeEstimateRequest
from app.services.openscad_llm import LlmUnavailable, OpenAiCompatibleClient
from app.services.shape3d_job_manager import (
    DEFAULT_MESH_LONGEST_MM,
    Shape3dJobManager,
)
from app.services.size_estimate import (
    ESTIMATE_TIMEOUT_S,
    MAX_ESTIMATE_MM,
    MIN_ESTIMATE_MM,
    SizeEstimateUnavailable,
    estimate_longest_mm,
)
from app.services.storage import StorageService

# ---------------------------------------------------------------------------
# La malla generada no tiene cotas y no las va a tener. Lo unico que se puede
# ofrecer honestamente es CUANTO MIDE el objeto de verdad, y eso es una
# sugerencia: se muestra, el usuario la confirma, y el trabajo registra de donde
# salio. Lo que estos tests cuidan es que nunca se convierta en una cota
# inventada — ni por una respuesta rara del modelo, ni por un servidor caido.
# ---------------------------------------------------------------------------


class ClienteFalso:
    def __init__(self, respuesta: str) -> None:
        self.respuesta = respuesta
        self.pedidos: list[str] = []

    def complete(self, system: str, user: str) -> str:
        self.pedidos.append(user)
        return self.respuesta


class ClienteQueFalla:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def complete(self, system: str, user: str) -> str:
        raise self.error


class TestWhatTheModelAnswers:
    def test_a_clean_json_answer_becomes_an_estimate(self) -> None:
        cliente = ClienteFalso('{"longest_mm": 95, "reference": "taza de cafe"}')

        estimacion = estimate_longest_mm("una taza de cafe", client=cliente)

        assert estimacion.longest_mm == pytest.approx(95.0)
        assert estimacion.reference == "taza de cafe"

    def test_the_object_reaches_the_model(self) -> None:
        cliente = ClienteFalso('{"longest_mm": 20, "reference": "tornillo M3"}')

        estimate_longest_mm("un tornillo M3", client=cliente)

        assert "un tornillo M3" in cliente.pedidos[0]

    def test_json_wrapped_in_markdown_still_counts(self) -> None:
        # Los modelos chicos envuelven en ```json casi siempre. Tirar una
        # respuesta correcta por el formato seria perder la estimacion por nada.
        cliente = ClienteFalso('Claro:\n```json\n{"longest_mm": 40, "reference": "pila AA"}\n```')

        assert estimate_longest_mm("una pila AA", client=cliente).longest_mm == pytest.approx(40.0)

    def test_a_bare_number_counts_too(self) -> None:
        cliente = ClienteFalso("95 mm")

        estimacion = estimate_longest_mm("una taza", client=cliente)

        assert estimacion.longest_mm == pytest.approx(95.0)
        # Sin referencia no se inventa una: el numero solo ya sirve.
        assert estimacion.reference == ""

    def test_a_number_written_as_text_inside_the_json_counts(self) -> None:
        cliente = ClienteFalso('{"longest_mm": "95 mm", "reference": "taza"}')

        assert estimate_longest_mm("una taza", client=cliente).longest_mm == pytest.approx(95.0)

    def test_the_reference_is_flattened_and_cut(self) -> None:
        # Viaja al trabajo y a la pantalla: un parrafo con saltos de linea no.
        cliente = ClienteFalso(
            '{"longest_mm": 95, "reference": "' + "taza\\n de cafe " * 20 + '"}'
        )

        referencia = estimate_longest_mm("una taza", client=cliente).reference

        assert "\n" not in referencia
        assert len(referencia) <= 80


class TestWhenThereIsNoHonestAnswer:
    """Preferir "no se" a un numero inventado. Cada uno de estos casos deja al
    usuario con el default de siempre, que es lo que pasaba antes."""

    def test_a_model_that_says_it_does_not_know_gives_no_estimate(self) -> None:
        cliente = ClienteFalso('{"longest_mm": null, "reference": null}')

        with pytest.raises(SizeEstimateUnavailable):
            estimate_longest_mm("un cosito", client=cliente)

    def test_garbage_gives_no_estimate(self) -> None:
        cliente = ClienteFalso("Como andas! No tengo forma de saber eso.")

        with pytest.raises(SizeEstimateUnavailable):
            estimate_longest_mm("algo", client=cliente)

    def test_an_empty_answer_gives_no_estimate(self) -> None:
        with pytest.raises(SizeEstimateUnavailable):
            estimate_longest_mm("algo", client=ClienteFalso(""))

    @pytest.mark.parametrize("medida", [0, 1, MIN_ESTIMATE_MM - 0.1, MAX_ESTIMATE_MM + 1, 4000])
    def test_a_measurement_out_of_printable_range_is_refused(self, medida: float) -> None:
        cliente = ClienteFalso(f'{{"longest_mm": {medida}, "reference": "x"}}')

        with pytest.raises(SizeEstimateUnavailable):
            estimate_longest_mm("algo", client=cliente)

    def test_a_negative_measurement_is_refused(self) -> None:
        cliente = ClienteFalso('{"longest_mm": -95, "reference": "x"}')

        with pytest.raises(SizeEstimateUnavailable):
            estimate_longest_mm("algo", client=cliente)

    def test_a_measurement_in_other_units_is_refused_instead_of_misread(self) -> None:
        # Leer "9.5 cm" como 9.5 mm es exactamente el error caro que hay que
        # evitar: la pieza sale diez veces mas chica y el plastico se tira.
        cliente = ClienteFalso("9.5 cm")

        with pytest.raises(SizeEstimateUnavailable):
            estimate_longest_mm("una taza", client=cliente)

    def test_a_true_is_not_a_measurement(self) -> None:
        cliente = ClienteFalso('{"longest_mm": true, "reference": "x"}')

        with pytest.raises(SizeEstimateUnavailable):
            estimate_longest_mm("algo", client=cliente)

    def test_a_dead_server_gives_no_estimate_instead_of_exploding(self) -> None:
        cliente = ClienteQueFalla(LlmUnavailable("No se pudo hablar con el modelo"))

        with pytest.raises(SizeEstimateUnavailable):
            estimate_longest_mm("una taza", client=cliente)

    def test_a_timeout_gives_no_estimate_instead_of_exploding(self) -> None:
        # urllib tira TimeoutError en la lectura, que NO es un LlmUnavailable:
        # sin atraparlo, el timeout se escapaba como error 500 de la pantalla.
        cliente = ClienteQueFalla(TimeoutError("timed out"))

        with pytest.raises(SizeEstimateUnavailable):
            estimate_longest_mm("una taza", client=cliente)

    def test_an_empty_description_never_reaches_the_model(self) -> None:
        cliente = ClienteFalso('{"longest_mm": 95, "reference": "x"}')

        with pytest.raises(SizeEstimateUnavailable):
            estimate_longest_mm("   ", client=cliente)

        assert cliente.pedidos == []


VISTO: dict = {}


class ClienteEspia(OpenAiCompatibleClient):
    """El cliente REAL del carril CAD, con la llamada HTTP interceptada."""

    def complete(self, system: str, user: str) -> str:
        VISTO["timeout"] = self.timeout_s
        VISTO["base_url"] = self.base_url
        VISTO["model"] = self.model
        return '{"longest_mm": 95, "reference": "taza"}'


class TestTheClientIsTheOneAlreadyConfigured:
    def test_it_reuses_the_configured_client_with_a_shorter_patience(self) -> None:
        # Es el mismo cliente del carril CAD (misma URL, mismo modelo), pero ese
        # espera hasta diez minutos por una pieza entera: la pantalla no puede
        # esperar eso por un numero de tres digitos.
        VISTO.clear()

        estimate_longest_mm(
            "una taza",
            client=ClienteEspia(base_url="http://x/v1", model="devstral-32k", timeout_s=600),
        )

        assert VISTO["base_url"] == "http://x/v1"
        assert VISTO["model"] == "devstral-32k"
        assert VISTO["timeout"] == ESTIMATE_TIMEOUT_S


# ---------------------------------------------------------------------------
# La procedencia de la medida. Un STL que dice "80 x 32 x 92 mm" sin decir de
# donde salio ese 100 del eje mayor esta mintiendo por omision: puede ser una
# decision del usuario o el relleno del programa, y no son lo mismo.
# ---------------------------------------------------------------------------


def tetrahedron() -> np.ndarray:
    a, b, c, d = (0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)
    return np.array([[a, c, b], [a, b, d], [a, d, c], [b, c, d]], dtype=np.float64)


class FakeEngine:
    def available(self) -> bool:
        return True

    def generate_from_text(self, prompt: str, **_kwargs) -> np.ndarray:
        return tetrahedron()


def make_manager(tmp_path: Path, cad_client=None) -> Shape3dJobManager:
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    return Shape3dJobManager(settings, FakeEngine(), cad_client=cad_client)


class TestWhereTheSizeCameFrom:
    @pytest.mark.asyncio
    async def test_without_a_size_the_job_says_it_was_the_default(self, tmp_path: Path) -> None:
        manager = make_manager(tmp_path)

        job = await manager.create_job(prompt="una taza")

        assert job.target_mm == pytest.approx(DEFAULT_MESH_LONGEST_MM)
        assert job.target_mm_source == "default"

    @pytest.mark.asyncio
    async def test_a_size_with_no_declared_origin_counts_as_the_users(
        self, tmp_path: Path
    ) -> None:
        manager = make_manager(tmp_path)

        job = await manager.create_job(prompt="una taza", target_mm=42.0)

        assert job.target_mm_source == "user"

    @pytest.mark.asyncio
    async def test_an_accepted_estimate_is_recorded_as_such_with_its_reference(
        self, tmp_path: Path
    ) -> None:
        manager = make_manager(tmp_path)

        job = await manager.create_job(
            prompt="una taza",
            target_mm=95.0,
            target_mm_source="estimate",
            target_mm_reference="taza de cafe",
        )

        assert job.target_mm_source == "estimate"
        assert job.target_mm_reference == "taza de cafe"

    @pytest.mark.asyncio
    async def test_a_reference_does_not_stick_to_a_size_the_user_wrote(
        self, tmp_path: Path
    ) -> None:
        # Seria atribuirle al modelo una medida que no dijo.
        manager = make_manager(tmp_path)

        job = await manager.create_job(
            prompt="una taza", target_mm=42.0, target_mm_reference="taza de cafe"
        )

        assert job.target_mm_reference is None

    @pytest.mark.asyncio
    async def test_nobody_can_declare_a_size_as_the_default(self, tmp_path: Path) -> None:
        # "Lo puso el programa" es algo que solo el programa puede decir de si
        # mismo; aceptarlo de afuera vuelve inutil el campo.
        manager = make_manager(tmp_path)

        with pytest.raises(ValueError):
            await manager.create_job(prompt="una taza", target_mm=95.0, target_mm_source="default")

    @pytest.mark.asyncio
    async def test_an_unknown_origin_is_refused(self, tmp_path: Path) -> None:
        manager = make_manager(tmp_path)

        with pytest.raises(ValueError):
            await manager.create_job(
                prompt="una taza", target_mm=95.0, target_mm_source="internet"
            )

    @pytest.mark.asyncio
    async def test_the_origin_travels_in_the_response(self, tmp_path: Path) -> None:
        manager = make_manager(tmp_path)
        job = await manager.create_job(
            prompt="una taza",
            target_mm=95.0,
            target_mm_source="estimate",
            target_mm_reference="taza de cafe",
        )

        respuesta = shape3d_job_to_response(job)

        assert respuesta.target_mm == pytest.approx(95.0)
        assert respuesta.target_mm_source == "estimate"
        assert respuesta.target_mm_reference == "taza de cafe"

    @pytest.mark.asyncio
    async def test_the_api_carries_the_declared_origin_into_the_job(
        self, tmp_path: Path
    ) -> None:
        manager = make_manager(tmp_path)

        respuesta = await create_shape3d_job(
            payload=Shape3dJobRequest(
                prompt="una taza",
                target_mm=95.0,
                target_mm_source="estimate",
                target_mm_reference="taza de cafe",
            ),
            request=None,
            jobs=manager,
        )

        assert respuesta.target_mm_source == "estimate"

    @pytest.mark.asyncio
    async def test_a_made_up_origin_from_the_api_is_a_400(self, tmp_path: Path) -> None:
        manager = make_manager(tmp_path)

        with pytest.raises(HTTPException) as exc_info:
            await create_shape3d_job(
                payload=Shape3dJobRequest(
                    prompt="una taza", target_mm=95.0, target_mm_source="internet"
                ),
                request=None,
                jobs=manager,
            )

        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# El endpoint. Sugiere y nada mas: no encola, no escala, y sin servidor de
# modelo configurado dice que no hay en vez de fingir un numero.
# ---------------------------------------------------------------------------


class TestTheEndpoint:
    @pytest.mark.asyncio
    async def test_it_answers_the_estimate_and_its_reference(self, tmp_path: Path) -> None:
        manager = make_manager(
            tmp_path, cad_client=ClienteFalso('{"longest_mm": 95, "reference": "taza de cafe"}')
        )

        respuesta = await estimate_print_size(
            payload=SizeEstimateRequest(prompt="una taza de cafe"), jobs=manager
        )

        assert respuesta.longest_mm == pytest.approx(95.0)
        assert respuesta.reference == "taza de cafe"

    @pytest.mark.asyncio
    async def test_it_queues_nothing(self, tmp_path: Path) -> None:
        # Estimar no es generar: si esto encolara, un usuario tanteando medidas
        # se comeria la cola de generacion entera.
        manager = make_manager(
            tmp_path, cad_client=ClienteFalso('{"longest_mm": 95, "reference": "taza"}')
        )

        await estimate_print_size(payload=SizeEstimateRequest(prompt="una taza"), jobs=manager)

        assert manager.queue_depth() == 0

    @pytest.mark.asyncio
    async def test_without_a_model_server_it_is_a_409_that_says_so(
        self, tmp_path: Path
    ) -> None:
        manager = make_manager(tmp_path, cad_client=None)

        with pytest.raises(HTTPException) as exc_info:
            await estimate_print_size(
                payload=SizeEstimateRequest(prompt="una taza"), jobs=manager
            )

        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_a_useless_answer_is_a_502_not_a_crash(self, tmp_path: Path) -> None:
        manager = make_manager(tmp_path, cad_client=ClienteFalso("ni idea, che"))

        with pytest.raises(HTTPException) as exc_info:
            await estimate_print_size(
                payload=SizeEstimateRequest(prompt="una taza"), jobs=manager
            )

        assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_an_empty_description_is_a_400(self, tmp_path: Path) -> None:
        manager = make_manager(
            tmp_path, cad_client=ClienteFalso('{"longest_mm": 95, "reference": "x"}')
        )

        with pytest.raises(HTTPException) as exc_info:
            await estimate_print_size(payload=SizeEstimateRequest(prompt="  "), jobs=manager)

        assert exc_info.value.status_code == 400


class TestTheMeshLaneKeepsWorkingWithoutAModelServer:
    @pytest.mark.asyncio
    async def test_a_job_without_a_size_still_completes_on_the_default(
        self, tmp_path: Path
    ) -> None:
        # Requisito duro: sin servidor de modelo, el carril de malla anda
        # exactamente como antes.
        manager = make_manager(tmp_path, cad_client=None)
        job = await manager.create_job(prompt="una taza")

        await manager._process_next()

        assert job.status.value == "completed"
        assert job.target_mm == pytest.approx(DEFAULT_MESH_LONGEST_MM)
        assert job.target_mm_source == "default"
