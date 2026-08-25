import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.services import model3d_service
from app.services.turnaround import Box


PANEL_BOXES = (
    Box(20, 20, 80, 120),
    Box(120, 28, 190, 126),
    Box(235, 15, 300, 110),
    Box(350, 25, 430, 117),
)


def _write_sheet(path: Path, boxes: tuple[Box, ...]) -> Path:
    width = max(box.x1 for box in boxes) + 20
    height = max(box.y1 for box in boxes) + 20
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle((box.x0, box.y0, box.x1 - 1, box.y1 - 1), fill="black")
    image.save(path)
    return path


@pytest.fixture
def fake_blender(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, dict[str, object]]] = []
    canned = {"ok": True, "source": "fake-blender"}

    def fake_run_script(_settings, script: str, payload: dict[str, object]):
        calls.append((script, payload))
        return canned

    monkeypatch.setattr(model3d_service.blender_service, "run_script", fake_run_script)
    return calls, canned


def test_audit_mesh_calls_blender_audit_script_and_returns_result(
    tmp_path: Path, fake_blender
) -> None:
    mesh = tmp_path / "character mesh.obj"
    calls, canned = fake_blender

    result = model3d_service.audit_mesh(object(), mesh)

    assert result is canned
    assert calls == [("audit_mesh.py", {"mesh": str(mesh)})]


def test_detect_views_names_every_panel_and_truncates_for_fewer_panels(
    tmp_path: Path,
) -> None:
    four_panel_sheet = _write_sheet(tmp_path / "four-views.png", PANEL_BOXES)
    four_views = model3d_service.detect_views(four_panel_sheet)

    assert [view.name for view in four_views] == list(model3d_service.VIEW_ORDER)
    assert [view.image for view in four_views] == [four_panel_sheet] * 4
    assert [view.ink for view in four_views] == list(PANEL_BOXES)

    three_boxes = PANEL_BOXES[:3]
    three_panel_sheet = _write_sheet(tmp_path / "three-views.png", three_boxes)
    three_views = model3d_service.detect_views(three_panel_sheet)

    assert len(three_views) == len(three_boxes) < len(model3d_service.VIEW_ORDER)
    assert [view.name for view in three_views] == ["front", "side", "back"]
    assert [view.image for view in three_views] == [three_panel_sheet] * 3
    assert [view.ink for view in three_views] == list(three_boxes)


def test_split_views_writes_named_tight_crops_with_crop_coordinate_ink_boxes(
    tmp_path: Path,
) -> None:
    boxes = PANEL_BOXES[:3]
    sheet = _write_sheet(tmp_path / "turnaround.png", boxes)
    out_dir = tmp_path / "missing" / "split"
    assert not out_dir.exists()

    views = model3d_service.split_views(sheet, out_dir)

    assert out_dir.is_dir()
    assert [view.name for view in views] == ["front", "side", "back"]
    assert [view.image.name for view in views] == ["front.png", "side.png", "back.png"]
    for view, sheet_box in zip(views, boxes):
        expected_path = out_dir / f"{view.name}.png"
        expected_crop_box = Box(0, 0, sheet_box.width, sheet_box.height)

        assert view.image == expected_path
        assert expected_path.is_file()
        assert view.ink == expected_crop_box
        assert view.ink != sheet_box
        with Image.open(expected_path) as crop:
            assert crop.size == (sheet_box.width, sheet_box.height)


def test_detected_view_as_payload_uses_json_serializable_ink_list(tmp_path: Path) -> None:
    view = model3d_service.DetectedView(
        name="front",
        image=tmp_path / "front.png",
        ink=Box(1, 2, 30, 40),
    )

    payload = view.as_payload()

    assert payload == {
        "image": str(tmp_path / "front.png"),
        "inkBox": [1, 2, 30, 40],
    }
    assert isinstance(payload["inkBox"], list)
    assert json.loads(json.dumps(payload)) == payload


def test_build_reference_scene_serializes_views_and_supports_height_override(
    tmp_path: Path, fake_blender
) -> None:
    views = [
        model3d_service.DetectedView("front", tmp_path / "front.png", Box(0, 0, 60, 100)),
        model3d_service.DetectedView("side", tmp_path / "side.png", Box(0, 0, 70, 98)),
    ]
    default_output = tmp_path / "default scene.blend"
    custom_output = tmp_path / "custom scene.blend"
    calls, canned = fake_blender

    default_result = model3d_service.build_reference_scene(object(), views, default_output)
    custom_result = model3d_service.build_reference_scene(
        object(), views, custom_output, height_meters=1.83
    )

    expected_views = {view.name: view.as_payload() for view in views}
    assert default_result is canned
    assert custom_result is canned
    assert calls == [
        (
            "build_reference_scene.py",
            {
                "views": expected_views,
                "heightMeters": 1.70,
                "output": str(default_output),
            },
        ),
        (
            "build_reference_scene.py",
            {
                "views": expected_views,
                "heightMeters": 1.83,
                "output": str(custom_output),
            },
        ),
    ]
