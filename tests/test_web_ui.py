from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from app.main import app

# Sin runtime JS en las deps del repo: se fija el fuente exacto de formatFps
# en la pagina renderizada y un espejo en Python valida los casos de borde.
EXPECTED_FORMAT_FPS_SOURCE = """
function formatFps(rawValue) {
if (typeof rawValue !== 'string' || !rawValue.includes('/')) return rawValue;
const [numerator, denominator] = rawValue.split('/').map(Number);
if (!denominator) return rawValue;
const decimal = numerator / denominator;
return Number.isInteger(decimal) ? String(decimal) : decimal.toFixed(2);
}
""".strip()


def get_index_html() -> str:
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        return response.text


def normalize_js(source: str) -> str:
    return "\n".join(line.strip() for line in source.strip().splitlines())


def extract_format_fps_source(html: str) -> str:
    match = re.search(r"function formatFps\(rawValue\) \{.*?\n\s*\}", html, re.DOTALL)
    assert match, "formatFps() is missing from the rendered page"
    return match.group(0)


def reference_format_fps(raw_value: str) -> str:
    if "/" not in raw_value:
        return raw_value
    numerator, denominator = (float(part) for part in raw_value.split("/")[:2])
    if not denominator:
        return raw_value
    decimal = numerator / denominator
    if decimal.is_integer():
        return str(int(decimal))
    return f"{decimal:.2f}"


def test_apply_video_profile_reads_snake_case_catalog_fields() -> None:
    html = get_index_html()

    assert "profile.model_key" in html
    assert "profile.video_codec" in html
    assert "profile.video_preset" in html


def test_apply_video_profile_does_not_reference_camel_case_fields() -> None:
    html = get_index_html()

    assert "profile.modelKey" not in html
    assert "profile.videoCodec" not in html
    assert "profile.videoPreset" not in html


def test_output_fps_display_is_normalized_through_a_formatter() -> None:
    html = get_index_html()

    assert "formatFps(data.metadata.outputFps)" in html


def test_format_fps_source_in_rendered_page_matches_pinned_algorithm() -> None:
    html = get_index_html()

    rendered_source = extract_format_fps_source(html)

    assert normalize_js(rendered_source) == normalize_js(EXPECTED_FORMAT_FPS_SOURCE)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("60/1", "60"),
        ("24000/1001", "23.98"),
        ("30", "30"),
        ("60/0", "60/0"),
    ],
)
def test_format_fps_reference_mirror_normalizes_fractions(raw_value: str, expected: str) -> None:
    assert reference_format_fps(raw_value) == expected
