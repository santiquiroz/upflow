from __future__ import annotations

import asyncio
import io
import json
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

import cv2
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from app.api import restore_routes
from app.api import routes as api_routes
from app.api.auth_deps import get_user_store
from app.config import Settings, get_settings
from app.models import JobStatus
from app.services.device_semaphores import DeviceSemaphores
from app.services.engines.face_detect import FaceDetection, priors
from app.services.engines.face_restore import FaceRestoreResult, RestoredFace, blended_patch, paste_faces
from app.services.face_geometry import TEMPLATE_FFHQ_512, align_face, align_matrix
from app.services.job_manager import JobManager
from app.services.photo_restore_chain import RESTORE_CHAIN, step_ids
from app.services.photo_restore_job import PhotoRestoreJobRunner
from app.services.photo_restore_pipeline import ModelUse, StepCall, StepOutcome
from app.services.photo_restore_presets import PHOTO_PRESETS
from app.services.restore_session import AnalysisDetectors, RestoreSessionStore, default_detectors
from app.services.storage import StorageService
from app.services.xmp_packet import DIGITAL_SOURCE_COMPOSITE, DIGITAL_SOURCE_ENHANCED
from test_job_manager_restore import FakeDevices, FakeRestoreEngine, NeverCalledEngine, always_ready, brighten

SIZE = 256
FACE_SCALE = 0.18
FACE_OFFSET = (60.0, 70.0)


def make_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))


def smooth_rgb(width: int = SIZE, height: int = SIZE, seed: int = 3) -> np.ndarray:
    coarse = np.random.default_rng(seed).random((6, 6, 3), dtype=np.float32)
    return np.clip(cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)


def png_bytes(width: int = SIZE, height: int = SIZE) -> bytes:
    buffer = io.BytesIO()
    pixels = np.round(smooth_rgb(width, height) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def download_name(response) -> str:
    # Starlette manda filename*=utf-8'' cuando el nombre tiene espacios o no es ASCII.
    disposition = response.headers["content-disposition"]
    encoded = re.search(r"filename\*=utf-8''(?P<name>[^;]+)", disposition)
    return unquote(encoded["name"]) if encoded else re.search(r'filename="(?P<name>[^"]+)"', disposition)["name"]


def face_landmarks() -> np.ndarray:
    return TEMPLATE_FFHQ_512 * FACE_SCALE + np.asarray(FACE_OFFSET)


def line_damage(rgb: np.ndarray) -> np.ndarray:
    probability = np.zeros(rgb.shape[:2], dtype=np.float32)
    probability[rgb.shape[0] // 2 - 1 : rgb.shape[0] // 2 + 1, :] = 0.9
    return probability


def one_face(rgb: np.ndarray) -> tuple[FaceDetection, ...]:
    points = face_landmarks()
    box = (float(FACE_OFFSET[0]), float(FACE_OFFSET[1]), FACE_OFFSET[0] + 92.0, FACE_OFFSET[1] + 92.0)
    return (FaceDetection(box=box, score=0.99, landmarks=tuple(map(tuple, points.tolist()))),)


def fake_faces(image: np.ndarray, call: StepCall) -> StepOutcome:
    matrix = align_matrix(face_landmarks())
    aligned = align_face(call.source, matrix)
    restored = np.clip(aligned + np.float32(0.25), 0.0, 1.0).astype(np.float32)
    face = RestoredFace(0, matrix, 0.6, aligned, restored)
    pasted = paste_faces(image, [blended_patch(face)], 1.0)
    result = FaceRestoreResult(image=pasted, faces=(face,), scale=1.0)
    return StepOutcome(pasted, details={"restored": [0]}, faces=result)


@dataclass
class RepairSpy:
    calls: list[StepCall] = field(default_factory=list)

    def __call__(self, image: np.ndarray, call: StepCall) -> StepOutcome:
        self.calls.append(call)
        return StepOutcome(image, details={"finalCoverage": 0.0})


@dataclass
class Harness:
    client: TestClient
    settings: Settings
    manager: JobManager
    sessions: RestoreSessionStore

    def wait(self, job_id: str, timeout: float = 30.0) -> None:
        job = self.manager.jobs[job_id]
        deadline = time.monotonic() + timeout
        while job.status not in (JobStatus.completed, JobStatus.failed, JobStatus.cancelled):
            assert time.monotonic() < deadline, f"job did not finish: {job.status}"
            time.sleep(0.02)
        assert job.status == JobStatus.completed, job.error

    def analyze(self, content: bytes | None = None, name: str = "Grandma 1952.png") -> dict:
        files = {"file": (name, content or png_bytes(), "image/png")}
        response = self.client.post("/api/v1/restore/analyze", files=files)
        assert response.status_code == 200, response.text
        return response.json()

    def restore(self, **form) -> dict:
        response = self.post_job(**form)
        assert response.status_code == 202, response.text
        body = response.json()
        self.wait(body["jobId"])
        return body

    def post_job(self, file: bytes | None = None, **form):
        data = {"device": "cpu", "restore_steps": "tone", **form}
        files = None if file is None else {"file": ("Grandma 1952.png", file, "image/png")}
        return self.client.post("/api/v1/restore/jobs", data=data, files=files)


def build_app(settings: Settings, manager: JobManager, sessions: RestoreSessionStore) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await manager.start()
        yield
        await manager.stop()

    app = FastAPI(lifespan=lifespan)
    app.include_router(api_routes.router)
    app.include_router(restore_routes.router)
    app.state.job_manager = manager
    app.state.restore_sessions = sessions
    app.state.storage = StorageService(settings)
    app.state.devices_service = FakeDevices()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_user_store] = lambda: None
    return app


@pytest.fixture
def harness_factory(tmp_path: Path):
    clients: list[TestClient] = []

    def make(step_runners=None, detectors: AnalysisDetectors | None = None) -> Harness:
        settings = make_settings(tmp_path)
        sessions = RestoreSessionStore(settings, lambda: detectors or AnalysisDetectors())
        runner = PhotoRestoreJobRunner(
            settings,
            FakeRestoreEngine(),
            step_runners=step_runners or {"tone": brighten},
            app_version="9.9.9",
            check_ready=always_ready,
            sessions=sessions,
        )
        manager = JobManager(
            settings, NeverCalledEngine(), DeviceSemaphores(settings), devices=FakeDevices(), restore_runner=runner
        )
        client = TestClient(build_app(settings, manager, sessions))
        client.__enter__()
        clients.append(client)
        return Harness(client, settings, manager, sessions)

    yield make
    for client in clients:
        client.__exit__(None, None, None)


def damage_and_face() -> AnalysisDetectors:
    return AnalysisDetectors(damage=line_damage, faces=one_face)


# ---------------------------------------------------------------------------
# Analisis
# ---------------------------------------------------------------------------


def test_analyze_opens_a_session_with_preview_diagnosis_damage_and_faces(harness_factory) -> None:
    harness = harness_factory(detectors=damage_and_face())

    body = harness.analyze()

    assert len(body["token"]) == 32
    assert (body["width"], body["height"], body["bitDepth"], body["hasIcc"]) == (SIZE, SIZE, 8, False)
    assert body["originalName"] == "Grandma 1952.png"
    assert body["geometry"] == {"rotate90": 0, "crop": None, "angle": 0.0}
    assert body["diagnosis"]["toneKind"] in {"mono", "toned", "hand_tinted", "color"}
    assert body["proposedPreset"] and isinstance(body["proposedSteps"], list)
    assert body["damage"]["coverage"] > 0
    assert body["damage"]["probUrl"].endswith("/damage_prob.png")
    assert [face["index"] for face in body["faces"]] == [0]
    assert body["faces"][0]["eyePx"] > 0
    assert body["damageOverFaces"] is True
    assert body["eta"]["cpuSeconds"] >= body["eta"]["gpuSeconds"] >= 0
    for url in (body["previewUrl"], body["damage"]["probUrl"], body["faces"][0]["thumbnailUrl"]):
        assert harness.client.get(url).status_code == 200


def test_the_session_remembers_which_detector_made_the_damage_map(harness_factory) -> None:
    detector = ModelUse("bopbtl-scratch-detector", "cpu", "fp32")
    harness = harness_factory(detectors=AnalysisDetectors(damage=line_damage, damage_model=detector))

    token = harness.analyze()["token"]

    assert harness.sessions.job_inputs(token).damage_detector == detector


def test_a_session_without_a_damage_map_names_no_detector(harness_factory) -> None:
    detector = ModelUse("bopbtl-scratch-detector", "cpu", "fp32")
    harness = harness_factory(detectors=AnalysisDetectors(damage=None, damage_model=detector))

    token = harness.analyze()["token"]

    assert harness.sessions.job_inputs(token).damage_detector is None


def test_analyze_resolves_every_preset_for_this_photo(harness_factory) -> None:
    harness = harness_factory(detectors=damage_and_face())

    body = harness.analyze()

    selections = body["presetSelections"]
    assert list(selections) == [preset.id for preset in PHOTO_PRESETS]
    for preset in PHOTO_PRESETS:
        assert set(selections[preset.id]["steps"]) <= set(preset.step_ids())
        assert set(selections[preset.id]["options"]) == set(selections[preset.id]["steps"])
    proposed = selections[body["proposedPreset"]]
    assert (proposed["steps"], proposed["options"]) == (body["proposedSteps"], body["proposedOptions"])


def test_analyze_estimates_each_step_so_the_total_follows_the_selection(harness_factory) -> None:
    harness = harness_factory(detectors=damage_and_face())

    body = harness.analyze()

    per_step = body["eta"]["perStep"]
    assert set(per_step) == set(step_ids(RESTORE_CHAIN))
    proposed = [per_step[step] for step in body["proposedSteps"]]
    assert sum(entry["cpuSeconds"] for entry in proposed) == pytest.approx(body["eta"]["cpuSeconds"])
    assert sum(entry["gpuSeconds"] for entry in proposed) == pytest.approx(body["eta"]["gpuSeconds"])
    assert per_step["faces"]["cpuSeconds"] > 0


def test_analyze_without_packs_names_the_missing_packs(harness_factory) -> None:
    harness = harness_factory()

    body = harness.analyze()

    missing = {finding["missingPack"] for finding in body["diagnosis"]["findings"] if finding["missingPack"]}
    assert missing == {"restore-core", "restore-faces"}
    assert body["damage"] == {"coverage": None, "probUrl": None, "largeHoles": 0}
    assert body["faces"] == []


def test_analyze_refuses_a_file_that_is_not_an_image(harness_factory) -> None:
    harness = harness_factory()

    response = harness.client.post("/api/v1/restore/analyze", files={"file": ("x.png", b"nope", "image/png")})

    assert response.status_code == 400
    assert not any(harness.settings.video_work_path.glob("restore-*"))
    assert not any(harness.settings.uploads_path.iterdir())


class SpyEngine:
    def __init__(self) -> None:
        self.devices: list[str] = []

    def begin_phase(self, device: str) -> None:
        self.devices.append(device)

    def tile_infer(self, model_id: str, device: str, precision: str, **kwargs):
        self.devices.append(device)
        return lambda tile: np.full(tile.shape, -10.0, dtype=np.float32)

    def session(self, model_id: str, device: str, precision: str):
        self.devices.append(device)
        return FakeRetinaSession()


class FakeRetinaSession:
    class _Input:
        name = "input"

    def get_inputs(self):
        return [self._Input()]

    def run(self, names, feeds):
        batch = feeds["input"]
        count = len(priors(batch.shape[2], batch.shape[3]))
        conf = np.zeros((1, count, 2), dtype=np.float32)
        conf[..., 0] = 1.0
        return [np.zeros((1, count, 4), np.float32), conf, np.zeros((1, count, 10), np.float32)]


def test_analysis_runs_its_detectors_only_on_the_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Settings, "restore_core_installed", property(lambda self: "core.installed.json"))
    monkeypatch.setattr(Settings, "restore_faces_installed", property(lambda self: "faces.installed.json"))
    settings = make_settings(tmp_path)
    StorageService(settings)
    engine = SpyEngine()
    sessions = RestoreSessionStore(settings, default_detectors(settings, engine))
    upload = settings.uploads_path / "upload.png"
    upload.write_bytes(png_bytes())

    analysis = asyncio.run(sessions.open(upload, "photo.png"))

    assert analysis.has_damage_map
    assert engine.devices and set(engine.devices) == {"cpu"}


def test_two_analyses_never_run_at_the_same_time(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    StorageService(settings)
    active: list[int] = []
    peak: list[int] = []

    def slow_damage(rgb: np.ndarray) -> np.ndarray:
        active.append(1)
        peak.append(len(active))
        time.sleep(0.05)
        active.pop()
        return np.zeros(rgb.shape[:2], dtype=np.float32)

    sessions = RestoreSessionStore(settings, lambda: AnalysisDetectors(damage=slow_damage))

    async def scenario() -> None:
        uploads = []
        for index in range(3):
            path = settings.uploads_path / f"u{index}.png"
            path.write_bytes(png_bytes(64, 48))
            uploads.append(path)
        await asyncio.gather(*(sessions.open(path, path.name) for path in uploads))

    asyncio.run(scenario())

    assert max(peak) == 1


# ---------------------------------------------------------------------------
# Geometria, archivos de la sesion y mascara
# ---------------------------------------------------------------------------


def test_geometry_redoes_the_working_copy_and_drops_the_painted_mask(harness_factory) -> None:
    harness = harness_factory(detectors=damage_and_face())
    token = harness.analyze(png_bytes(SIZE, 128))["token"]
    mask = harness.client.post(f"/api/v1/restore/analysis/{token}/mask", files={"file": ("m.png", mask_png(SIZE, 128))})
    assert mask.status_code == 200

    response = harness.client.post(f"/api/v1/restore/analysis/{token}/geometry", json={"rotate90": 1, "angle": 2.5})

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["width"], body["height"]) == (128, SIZE)
    assert body["geometry"] == {"rotate90": 1, "crop": None, "angle": 2.5}
    assert harness.sessions.job_inputs(token).user_mask is None
    assert harness.sessions.original_path(token).is_file(), "geometry never touches the original"


def test_geometry_crop_outside_the_photo_is_refused(harness_factory) -> None:
    harness = harness_factory()
    token = harness.analyze()["token"]

    response = harness.client.post(
        f"/api/v1/restore/analysis/{token}/geometry", json={"crop": [200, 200, 100, 100]}
    )

    assert response.status_code == 400
    assert "outside" in response.json()["detail"]


def test_geometry_rejects_an_angle_beyond_the_straighten_limit(harness_factory) -> None:
    harness = harness_factory()
    token = harness.analyze()["token"]

    response = harness.client.post(f"/api/v1/restore/analysis/{token}/geometry", json={"angle": 60})

    assert response.status_code == 422


@pytest.mark.parametrize("token", ["0" * 32, "not-a-token", "../" + "0" * 29])
def test_unknown_sessions_are_not_found(harness_factory, token: str) -> None:
    harness = harness_factory()

    response = harness.client.post(f"/api/v1/restore/analysis/{token}/geometry", json={"rotate90": 1})

    assert response.status_code == 404


@pytest.mark.parametrize("name", ["session.json", "original.png", "faces.json", "damage_mask.png", "face-x.jpg"])
def test_session_files_outside_the_whitelist_are_not_served(harness_factory, name: str) -> None:
    harness = harness_factory(detectors=damage_and_face())
    token = harness.analyze()["token"]

    assert harness.client.get(f"/api/v1/restore/analysis/{token}/{name}").status_code == 404


def mask_png(width: int, height: int, mode: str = "L", fill: int = 0) -> bytes:
    pixels = np.full((height, width), fill, dtype=np.uint8)
    pixels[10:20, 10:40] = 255
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="L").convert(mode).save(buffer, format="PNG")
    return buffer.getvalue()


def test_mask_upload_stores_the_painted_mask_for_the_job(harness_factory) -> None:
    harness = harness_factory()
    token = harness.analyze()["token"]

    response = harness.client.post(f"/api/v1/restore/analysis/{token}/mask", files={"file": ("m.png", mask_png(SIZE, SIZE))})

    assert response.status_code == 200, response.text
    assert response.json() == {"coverage": pytest.approx(300 / SIZE**2), "width": SIZE, "height": SIZE}
    stored = harness.sessions.job_inputs(token).user_mask
    assert stored.shape == (SIZE, SIZE) and stored.sum() == 300


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (lambda: mask_png(SIZE - 1, SIZE), "photo is"),
        (lambda: mask_png(SIZE, SIZE, fill=128), "only contain black"),
        (lambda: mask_png(SIZE, SIZE, mode="RGB"), "black and white PNG"),
        (lambda: b"not a png", "black and white PNG"),
    ],
)
def test_bad_masks_are_refused(harness_factory, content, message: str) -> None:
    harness = harness_factory()
    token = harness.analyze()["token"]

    response = harness.client.post(f"/api/v1/restore/analysis/{token}/mask", files={"file": ("m.png", content())})

    assert response.status_code == 400
    assert message in response.json()["detail"]


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def test_a_session_job_uses_the_session_mask_faces_and_geometry(harness_factory) -> None:
    spy = RepairSpy()
    harness = harness_factory(step_runners={"repair": spy}, detectors=damage_and_face())
    analysis = harness.analyze(png_bytes(SIZE, 200))
    token = analysis["token"]
    rotated = harness.client.post(f"/api/v1/restore/analysis/{token}/geometry", json={"rotate90": 1}).json()
    harness.client.post(f"/api/v1/restore/analysis/{token}/mask", files={"file": ("m.png", mask_png(200, SIZE))})
    options = {"repair": {"engine": "classic"}, "geometry": {"rotate90": 2}}

    body = harness.restore(token=token, restore_steps="repair", restore_options=json.dumps(options))

    assert body["restoreSteps"] == ["repair"]
    request = spy.calls[0].request
    assert request.image.shape[:2] == (rotated["height"], rotated["width"]) == (SIZE, 200)
    assert request.hints.user_mask.shape == (SIZE, 200) and request.hints.user_mask.sum() == 300
    assert request.hints.damage_probability.shape == (SIZE, 200)
    assert [face.index for face in request.faces] == [0]
    job = harness.manager.jobs[body["jobId"]]
    assert job.restore_options["geometry"] == {"rotate90": 1, "crop": None, "angle": 0.0}
    sidecar = json.loads((harness.settings.outputs_path / f"{job.id}.restore.json").read_text(encoding="utf-8"))
    assert sidecar["geometry"] == {"rotate90": 1, "crop": None, "angle": 0.0}
    assert sidecar["input"]["sha256"] == analysis["sha256"]
    assert harness.sessions.original_path(token).is_file(), "the session original must survive its jobs"


def test_a_session_can_run_a_preview_and_then_the_full_photo(harness_factory) -> None:
    harness = harness_factory()
    token = harness.analyze()["token"]
    preview_options = json.dumps({"preview_crop": [10, 10, 64, 64]})

    preview = harness.restore(token=token, restore_options=preview_options)
    full = harness.restore(token=token)

    preview_job = harness.manager.jobs[preview["jobId"]]
    assert preview_job.output_path.name.endswith(".preview.jpg")
    names = {path.name for path in harness.settings.outputs_path.iterdir() if path.name.startswith(preview["jobId"])}
    assert names == {f"{preview['jobId']}.preview.jpg"}
    artifact = harness.client.get(f"/api/v1/jobs/{preview['jobId']}/artifacts/preview")
    assert artifact.status_code == 200
    assert download_name(artifact) == "Grandma 1952_preview.jpg"
    assert harness.client.get(f"/api/v1/jobs/{preview['jobId']}/artifacts/view").status_code == 404
    assert harness.manager.jobs[full["jobId"]].status == JobStatus.completed


def test_an_uploaded_photo_job_downloads_with_readable_names(harness_factory) -> None:
    harness = harness_factory()

    body = harness.restore(file=png_bytes())

    job_id = body["jobId"]
    download = harness.client.get(f"/api/v1/jobs/{job_id}/download")
    assert download.status_code == 200
    assert download_name(download) == "Grandma 1952_restored.png"
    expected = {
        "beforeafter": ("Grandma 1952_before-after.jpg", "image/jpeg"),
        "sidecar": ("Grandma 1952_restore.json", "application/json"),
        "view": ("Grandma 1952_view.jpg", "image/jpeg"),
    }
    for name, (filename, media_type) in expected.items():
        response = harness.client.get(f"/api/v1/jobs/{job_id}/artifacts/{name}")
        assert response.status_code == 200, name
        assert download_name(response) == filename
        assert response.headers["content-type"].startswith(media_type)
    assert not any(harness.settings.uploads_path.iterdir()), "the uploaded photo is the job's and is removed"


@pytest.mark.parametrize("name", ["..%2F..%2Fsecret", "original", "face:0:after", "uncolored"])
def test_artifacts_outside_the_whitelist_or_missing_are_not_found(harness_factory, name: str) -> None:
    harness = harness_factory()
    job_id = harness.restore(file=png_bytes())["jobId"]

    assert harness.client.get(f"/api/v1/jobs/{job_id}/artifacts/{name}").status_code == 404


def test_job_json_carries_restore_steps(harness_factory) -> None:
    harness = harness_factory()
    job_id = harness.restore(file=png_bytes())["jobId"]

    body = harness.client.get(f"/api/v1/jobs/{job_id}").json()

    assert body["restoreSteps"] == ["tone"]


@pytest.mark.parametrize(
    ("form", "status", "detail"),
    [
        ({"restore_steps": ""}, 400, "at least one restore step"),
        ({"restore_steps": "sharpen"}, 400, "sharpen"),
        ({"restore_options": "{not json"}, 422, None),
        ({"restore_options": json.dumps({"repair": {"sensitivty": 0.5}})}, 422, None),
        ({"restore_options": json.dumps({"tone": {"strength": 2}})}, 422, None),
        ({"scale": "1", "restore_options": json.dumps({"upscale_mode": "ai"})}, 400, "does not match"),
    ],
)
def test_bad_job_requests_are_refused(harness_factory, form: dict, status: int, detail: str | None) -> None:
    harness = harness_factory()

    response = harness.post_job(file=png_bytes(), **form)

    assert response.status_code == status, response.text
    if detail is not None:
        assert detail in response.json()["detail"]
    assert not any(harness.settings.uploads_path.iterdir()), "a refused upload must not stay on disk"


def test_a_job_needs_exactly_one_of_photo_and_token(harness_factory) -> None:
    harness = harness_factory()
    token = harness.analyze()["token"]

    neither = harness.post_job()
    both = harness.post_job(file=png_bytes(), token=token)

    assert neither.status_code == both.status_code == 400
    assert harness.sessions.original_path(token).is_file()


def test_a_job_with_an_unknown_token_is_not_found(harness_factory) -> None:
    harness = harness_factory()

    assert harness.post_job(token="f" * 32).status_code == 404


def test_create_restore_job_coroutine_refuses_a_missing_selection(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    with pytest.raises(HTTPException) as caught:
        asyncio.run(restore_routes.create_restore_job(settings=settings))

    assert caught.value.status_code == 400


# ---------------------------------------------------------------------------
# Lote con caras (P4-BATCH)
# ---------------------------------------------------------------------------


def read_sidecar(harness: Harness, job_id: str) -> dict:
    return json.loads((harness.settings.outputs_path / f"{job_id}.restore.json").read_text(encoding="utf-8"))


def test_a_batch_photo_restores_its_own_faces_and_is_marked_as_batch(harness_factory) -> None:
    harness = harness_factory(step_runners={"faces": fake_faces})
    options = json.dumps({"batch": True, "faces": {"blend": 0.6}})

    job_id = harness.restore(file=png_bytes(), restore_steps="faces", restore_options=options)["jobId"]

    job = harness.manager.jobs[job_id]
    assert job.restore_options["batch"] is True
    assert job.metadata["restore"]["batch"] is True
    sidecar = read_sidecar(harness, job_id)
    assert sidecar["batch"] is True
    assert [step["id"] for step in sidecar["steps"]] == ["faces"]


def test_each_batch_result_can_be_reviewed_face_by_face(harness_factory) -> None:
    harness = harness_factory(step_runners={"faces": fake_faces})
    options = json.dumps({"batch": True, "faces": {"blend": 0.6}})
    job_id = harness.restore(file=png_bytes(), restore_steps="faces", restore_options=options)["jobId"]

    response = harness.client.post(
        f"/api/v1/restore/jobs/{job_id}/recompose", json={"faces": {"0": {"enabled": False, "blend": 0.6}}}
    )

    assert response.status_code == 200, response.text
    sidecar = response.json()["sidecar"]
    assert sidecar["batch"] is True
    assert next(face for face in sidecar["faces"] if face["index"] == 0)["enabled"] is False


def test_a_single_photo_is_not_marked_as_batch(harness_factory) -> None:
    harness = harness_factory()

    job_id = harness.restore(file=png_bytes())["jobId"]

    assert harness.manager.jobs[job_id].metadata["restore"]["batch"] is False
    assert read_sidecar(harness, job_id)["batch"] is False


@pytest.mark.parametrize(
    ("options", "named"),
    [
        ({"faces": {"selected": [0]}}, "faces.selected"),
        ({"faces": {"per_face": {"0": 0.4}}}, "faces.per_face"),
        ({"geometry": {"rotate90": 1}}, "geometry"),
        ({"preview_crop": [0, 0, 64, 64]}, "preview_crop"),
        ({"tone": {"gray_point": [10, 10]}}, "tone.gray_point"),
        ({"repair": {"use_user_mask": True}}, "repair.use_user_mask"),
    ],
)
def test_a_batch_refuses_choices_made_on_another_photo(harness_factory, options: dict, named: str) -> None:
    harness = harness_factory()

    response = harness.post_job(
        file=png_bytes(), restore_steps="tone,faces", restore_options=json.dumps({"batch": True, **options})
    )

    assert response.status_code == 400, response.text
    assert named in response.json()["detail"]
    assert not any(harness.settings.uploads_path.iterdir()), "a refused batch photo must not stay on disk"


def test_a_batch_photo_cannot_reuse_an_analysis_session(harness_factory) -> None:
    harness = harness_factory()
    token = harness.analyze()["token"]

    response = harness.post_job(token=token, restore_options=json.dumps({"batch": True}))

    assert response.status_code == 400
    assert "batch" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Recomposicion
# ---------------------------------------------------------------------------


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def test_recompose_turns_a_face_off_and_rewrites_outputs_and_sidecar(harness_factory) -> None:
    harness = harness_factory(step_runners={"faces": fake_faces})
    job_id = harness.restore(file=png_bytes(), restore_steps="faces")["jobId"]
    outputs = harness.settings.outputs_path
    before = json.loads((outputs / f"{job_id}.restore.json").read_text(encoding="utf-8"))
    assert before["digitalSourceType"] == DIGITAL_SOURCE_COMPOSITE

    response = harness.client.post(
        f"/api/v1/restore/jobs/{job_id}/recompose", json={"faces": {"0": {"enabled": False, "blend": 0.6}}}
    )

    assert response.status_code == 200, response.text
    sidecar = response.json()["sidecar"]
    face = next(entry for entry in sidecar["faces"] if entry["index"] == 0)
    assert face["enabled"] is False and face["recomposedAt"]
    assert sidecar["digitalSourceType"] == DIGITAL_SOURCE_ENHANCED
    assert sidecar["compositeReasons"] == []
    saved = json.loads((outputs / f"{job_id}.restore.json").read_text(encoding="utf-8"))
    assert saved == sidecar
    original = np.round(smooth_rgb() * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(read_rgb(outputs / f"{job_id}.png"), original)
    assert read_rgb(outputs / f"{job_id}.beforeafter.jpg").shape[1] == 2 * SIZE + 4
    roles = {entry["role"]: entry["sha256"] for entry in sidecar["outputs"]}
    assert set(roles) == {"restored", "view", "preview", "beforeafter"}
    assert harness.manager.jobs[job_id].metadata["restore"]["badge"] is False


def test_recompose_back_on_restores_the_face_and_the_badge(harness_factory) -> None:
    harness = harness_factory(step_runners={"faces": fake_faces})
    job_id = harness.restore(file=png_bytes(), restore_steps="faces")["jobId"]
    url = f"/api/v1/restore/jobs/{job_id}/recompose"
    harness.client.post(url, json={"faces": {"0": {"enabled": False, "blend": 0.6}}})

    response = harness.client.post(url, json={"faces": {"0": {"enabled": True, "blend": 1.0}}})

    assert response.status_code == 200
    assert response.json()["sidecar"]["digitalSourceType"] == DIGITAL_SOURCE_COMPOSITE
    assert harness.manager.jobs[job_id].metadata["restore"]["badge"] is True


def test_recompose_of_a_face_that_was_not_restored_is_a_bad_request(harness_factory) -> None:
    harness = harness_factory(step_runners={"faces": fake_faces})
    job_id = harness.restore(file=png_bytes(), restore_steps="faces")["jobId"]

    response = harness.client.post(
        f"/api/v1/restore/jobs/{job_id}/recompose", json={"faces": {"4": {"enabled": True, "blend": 0.5}}}
    )

    assert response.status_code == 400


def test_recompose_without_restored_faces_is_a_conflict(harness_factory) -> None:
    harness = harness_factory()
    job_id = harness.restore(file=png_bytes())["jobId"]

    response = harness.client.post(
        f"/api/v1/restore/jobs/{job_id}/recompose", json={"faces": {"0": {"enabled": False, "blend": 0.5}}}
    )

    assert response.status_code == 409


def test_recompose_of_an_unknown_job_is_not_found(harness_factory) -> None:
    harness = harness_factory()

    response = harness.client.post("/api/v1/restore/jobs/nope/recompose", json={"faces": {}})

    assert response.status_code == 404


def test_face_artifacts_are_served_after_a_face_restore(harness_factory) -> None:
    harness = harness_factory(step_runners={"faces": fake_faces})
    job_id = harness.restore(file=png_bytes(), restore_steps="faces")["jobId"]

    for side in ("before", "after"):
        response = harness.client.get(f"/api/v1/jobs/{job_id}/artifacts/face:0:{side}")
        assert response.status_code == 200
        assert download_name(response) == f"Grandma 1952_face-0-{side}.png"


# ---------------------------------------------------------------------------
# Capacidades y cableado real
# ---------------------------------------------------------------------------


def test_capabilities_list_the_chain_presets_and_pack_state(harness_factory) -> None:
    harness = harness_factory()

    body = harness.client.get("/api/v1/restore/capabilities").json()

    steps = {step["id"]: step for step in body["steps"]}
    assert list(steps) == ["descreen", "repair", "deblock", "denoise", "tone", "faces", "colorize"]
    assert steps["tone"]["installed"] is True and steps["tone"]["pack"] is None
    assert steps["faces"]["installed"] is False and steps["faces"]["pack"] == "restore-faces"
    assert "gentle" in {preset["id"] for preset in body["presets"]}
    assert body["halftoneDenoiseLimit"] > 0


def test_the_app_lifespan_wires_sessions_into_the_restore_runner() -> None:
    from app.main import app

    with TestClient(app) as client:
        sessions = app.state.restore_sessions
        assert isinstance(sessions, RestoreSessionStore)
        assert app.state.job_manager.restore_runner.sessions is sessions
        assert client.get("/api/v1/restore/capabilities").status_code == 200
        assert client.get("/api/v1/licenses").status_code == 200
        assert client.post("/api/v1/restore/analysis/" + "0" * 32 + "/geometry", json={}).status_code == 404


def test_the_recolorize_from_luminance_flag_reaches_the_colorize_step_params() -> None:
    options = restore_routes.parse_restore_options(json.dumps({"colorize": {"from_luminance": True, "strength": 0.6}}))

    assert options["colorize"] == {"from_luminance": True, "strength": 0.6}
