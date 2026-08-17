import pytest

from app.mcp.jobs import FAMILIES, family_or_raise, is_terminal, normalize_job


def test_families_dict():
    assert len(FAMILIES) == 7
    expected_keys = {"image", "video", "audio", "generation", "transcribe", "download", "shape3d"}
    assert set(FAMILIES.keys()) == expected_keys


def test_every_family_lists_at_its_base_path():
    """Las 7 listan con un GET a su base_path: ya no hay familias que obliguen a
    guardar el jobId al crearlas. shape3d es la rara — su listado es
    /print/generate, no /print/generate/jobs."""
    assert all(family.base_path.startswith("/api/v1/") for family in FAMILIES.values())
    assert FAMILIES["transcribe"].base_path == "/api/v1/transcribe/jobs"
    assert FAMILIES["download"].base_path == "/api/v1/download/jobs"
    assert FAMILIES["shape3d"].base_path == "/api/v1/print/generate"


def test_family_or_raise_valid():
    family = family_or_raise("image")
    assert family.name == "image"
    assert family.base_path == "/api/v1/jobs"
    assert family.id_field == "jobId"


def test_family_or_raise_invalid_lists_valid_names():
    with pytest.raises(ValueError) as excinfo:
        family_or_raise("invalid_family")
    message = str(excinfo.value)
    for name in FAMILIES:
        assert name in message


def test_normalize_job_image():
    payload = {"jobId": "1234", "status": "queued"}
    job = normalize_job(FAMILIES["image"], payload)
    assert job == {
        "family": "image",
        "jobId": "1234",
        "status": "queued",
        "progressPct": None,
        "error": None,
        "downloadUrl": None,
        "createdAt": None,
        "finishedAt": None,
    }


def test_normalize_job_audio_uses_id_field():
    job = normalize_job(FAMILIES["audio"], {"id": "5678", "status": "running"})
    assert job["jobId"] == "5678"
    assert job["family"] == "audio"


def test_normalize_job_video_extracts_stage():
    payload = {"jobId": "v1", "status": "running", "metadata": {"stage": "encode"}}
    job = normalize_job(FAMILIES["video"], payload)
    assert job["stage"] == "encode"


def test_normalize_job_transcribe_includes_text_only_when_present():
    with_text = normalize_job(
        FAMILIES["transcribe"], {"id": "t1", "status": "completed", "text": "hola"}
    )
    assert with_text["text"] == "hola"
    without_text = normalize_job(FAMILIES["transcribe"], {"id": "t2", "status": "queued"})
    assert "text" not in without_text


def test_normalize_job_download_extras():
    payload = {"id": "d1", "status": "completed", "outputFiles": ["a.mp4"], "mediaTitle": "Clip"}
    job = normalize_job(FAMILIES["download"], payload)
    assert job["outputFiles"] == ["a.mp4"]
    assert job["mediaTitle"] == "Clip"


def test_normalize_job_shape3d_extras():
    payload = {"id": "s1", "status": "completed", "canPrint": True, "sizeMm": [10, 10, 10]}
    job = normalize_job(FAMILIES["shape3d"], payload)
    assert job["canPrint"] is True
    assert job["sizeMm"] == [10, 10, 10]


def test_is_terminal_terminal_statuses():
    assert is_terminal({"status": "completed"}) is True
    assert is_terminal({"status": "failed"}) is True
    assert is_terminal({"status": "cancelled"}) is True
    assert is_terminal({"status": "Completed"}) is True


def test_is_terminal_non_terminal_statuses():
    assert is_terminal({"status": "queued"}) is False
    assert is_terminal({"status": "running"}) is False
    assert is_terminal({}) is False
