from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from app.services.generation_soft_inpaint import (
    build_soft_inpaint_callback,
    chain_step_callbacks,
)

LATENT_H = LATENT_W = 8


def make_fake_pipeline() -> SimpleNamespace:
    original = torch.zeros((1, 4, LATENT_H, LATENT_W))

    class FakeScheduler:
        timesteps = torch.tensor([800, 600, 400, 200])

        def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            # Marca reconocible: el "original ruidificado" es una constante por
            # timestep, así el test puede afirmar exactamente qué se mezcló.
            return torch.full_like(latents, float(t[0]))

    return SimpleNamespace(
        image_processor=SimpleNamespace(preprocess=lambda image: torch.zeros((1, 3, 64, 64))),
        _encode_vae_image=lambda image, generator: original,
        scheduler=FakeScheduler(),
    )


def make_mask(value: int) -> Image.Image:
    return Image.new("L", (64, 64), value)


def test_blend_uses_the_next_timestep_noise_level() -> None:
    pipeline = make_fake_pipeline()
    # Máscara negra (0 = conservar): el blend debe reemplazar TODO el latent
    # con el original ruidificado al timestep SIGUIENTE (600, no 800).
    callback = build_soft_inpaint_callback(pipeline, Image.new("RGB", (64, 64)), make_mask(0), seed=1)
    latents = torch.ones((1, 4, LATENT_H, LATENT_W)) * 99.0

    callback(0, torch.tensor(800), latents)

    assert torch.allclose(latents, torch.full_like(latents, 600.0))


def test_white_mask_keeps_the_generated_latents_untouched() -> None:
    pipeline = make_fake_pipeline()
    callback = build_soft_inpaint_callback(pipeline, Image.new("RGB", (64, 64)), make_mask(255), seed=1)
    latents = torch.ones((1, 4, LATENT_H, LATENT_W)) * 99.0

    callback(0, torch.tensor(800), latents)

    assert torch.allclose(latents, torch.full_like(latents, 99.0))


def test_last_step_is_never_blended() -> None:
    # Los latents del último paso YA son la imagen final: ruidificarlos la
    # mancharía. El callback debe dejarlos intactos.
    pipeline = make_fake_pipeline()
    callback = build_soft_inpaint_callback(pipeline, Image.new("RGB", (64, 64)), make_mask(0), seed=1)
    latents = torch.ones((1, 4, LATENT_H, LATENT_W)) * 99.0

    callback(3, torch.tensor(200), latents)

    assert torch.allclose(latents, torch.full_like(latents, 99.0))


def test_unknown_timestep_is_a_noop() -> None:
    pipeline = make_fake_pipeline()
    callback = build_soft_inpaint_callback(pipeline, Image.new("RGB", (64, 64)), make_mask(0), seed=1)
    latents = torch.ones((1, 4, LATENT_H, LATENT_W)) * 99.0

    callback(0, torch.tensor(12345), latents)

    assert torch.allclose(latents, torch.full_like(latents, 99.0))


def test_gray_mask_blends_proportionally() -> None:
    pipeline = make_fake_pipeline()
    callback = build_soft_inpaint_callback(pipeline, Image.new("RGB", (64, 64)), make_mask(128), seed=1)
    latents = torch.ones((1, 4, LATENT_H, LATENT_W)) * 100.0

    callback(0, torch.tensor(800), latents)

    weight = 128.0 / 255.0
    expected = 100.0 * weight + 600.0 * (1.0 - weight)
    assert torch.allclose(latents, torch.full_like(latents, expected), atol=1e-4)


def test_chain_disables_soft_after_a_failure_but_keeps_progress() -> None:
    calls = {"progress": 0, "soft": 0}

    def broken_soft(i: int, t: object, latents: object) -> None:
        calls["soft"] += 1
        raise RuntimeError("boom")

    def progress(i: int, t: object, latents: object) -> None:
        calls["progress"] += 1

    step = chain_step_callbacks(broken_soft, progress)
    step(0, None, None)
    step(1, None, None)

    # El soft falló UNA vez y quedó deshabilitado; el progreso siguió siempre.
    assert calls == {"soft": 1, "progress": 2}


def test_chain_without_soft_is_just_progress() -> None:
    seen: list[int] = []
    step = chain_step_callbacks(None, lambda i, t, latents: seen.append(i))
    step(0, None, None)
    assert seen == [0]
