from __future__ import annotations

import pytest

from app.services.vram_estimate import estimate_peak_bytes

GB = 1024**3


def test_peak_is_always_above_the_weights_themselves():
    assert estimate_peak_bytes(2 * GB, 512, 512) > 2 * GB


def test_monotonic_in_weights():
    small = estimate_peak_bytes(1 * GB, 512, 512)
    large = estimate_peak_bytes(4 * GB, 512, 512)
    assert large > small


def test_monotonic_in_resolution():
    at_512 = estimate_peak_bytes(2 * GB, 512, 512)
    at_768 = estimate_peak_bytes(2 * GB, 768, 768)
    at_1024 = estimate_peak_bytes(2 * GB, 1024, 1024)
    assert at_512 < at_768 < at_1024


def test_saturates_above_the_top_step():
    # Mas alla de 1024x1024 el factor no sigue creciendo: el escalon superior
    # es el techo tabulado.
    assert estimate_peak_bytes(2 * GB, 2048, 2048) == estimate_peak_bytes(2 * GB, 1024, 1024)


def test_clamps_below_the_bottom_step():
    assert estimate_peak_bytes(2 * GB, 64, 64) == estimate_peak_bytes(2 * GB, 512, 512)


def test_non_square_resolutions_use_pixel_count():
    # 1024x256 y 512x512 tienen los mismos pixeles: mismo factor.
    assert estimate_peak_bytes(2 * GB, 1024, 256) == estimate_peak_bytes(2 * GB, 512, 512)


def test_zero_weights_gives_zero():
    assert estimate_peak_bytes(0, 512, 512) == 0


@pytest.mark.parametrize("width,height", [(512, 512), (768, 768), (1024, 1024)])
def test_sd15_fp16_estimate_stays_in_a_plausible_range(width, height):
    # SD1.5 fp16 son ~2.6 GB de pesos. Cualquier estimacion sana cae entre
    # los pesos y 3x los pesos; el test protege de un factor absurdo.
    peak = estimate_peak_bytes(2611 * 1024 * 1024, width, height)
    assert 2611 * 1024 * 1024 < peak < 3 * 2611 * 1024 * 1024
