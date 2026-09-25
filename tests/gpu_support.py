"""Los tests `@pytest.mark.gpu` viven en tests/gpu (fuera de la coleccion por
defecto) y ademas se saltean salvo con UPFLOW_GPU_TESTS=1: nombrar el archivo a
mano no alcanza para ocupar una GPU que otro proceso puede estar usando.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

GPU_TESTS_ENV = "UPFLOW_GPU_TESTS"
GPU_MARKER = "gpu"


def gpu_tests_enabled(environ: Mapping[str, str]) -> bool:
    return environ.get(GPU_TESTS_ENV) == "1"


def enforce_gpu_opt_in(enabled: bool) -> None:
    if not enabled:
        pytest.skip(f"GPU tests run only with {GPU_TESTS_ENV}=1")


def check_gpu_for(item: pytest.Item, environ: Mapping[str, str]) -> None:
    if item.get_closest_marker(GPU_MARKER) is None:
        return
    enforce_gpu_opt_in(gpu_tests_enabled(environ))
