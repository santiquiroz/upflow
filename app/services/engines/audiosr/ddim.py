from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Numpy port of the DDIM sampler audiosr actually runs (ddim.py + the
# v-parameterization helpers in ddpm.py). The schedule is NOT recomputed from
# betas: alphas_cumprod comes verbatim from the checkpoint (alphas_cumprod.npy)
# so there is no cosine-schedule reimplementation to drift.


@dataclass(frozen=True)
class DdimSchedule:
    timesteps: np.ndarray          # ascending ddpm t indices, len = steps
    alphas: np.ndarray             # alphas_cumprod[timesteps]
    alphas_prev: np.ndarray
    sigmas: np.ndarray
    sqrt_alphas_cumprod: np.ndarray        # over the full 1000 ddpm steps
    sqrt_one_minus_alphas_cumprod: np.ndarray

    @staticmethod
    def build(alphas_cumprod: np.ndarray, num_steps: int, eta: float = 1.0) -> "DdimSchedule":
        num_train = alphas_cumprod.shape[0]
        stride = num_train // num_steps
        timesteps = np.arange(0, num_train, stride) + 1  # make_ddim_timesteps "uniform"
        # audiosr's own make_ddim_timesteps overflows to t=1000 when num_steps
        # does not divide 1000 (and crashes upstream); clamp instead.
        timesteps = timesteps[timesteps < num_train]
        alphas = alphas_cumprod[timesteps]
        alphas_prev = np.concatenate([[alphas_cumprod[0]], alphas_cumprod[timesteps[:-1]]])
        sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
        return DdimSchedule(
            timesteps=timesteps,
            alphas=alphas,
            alphas_prev=alphas_prev,
            sigmas=sigmas,
            sqrt_alphas_cumprod=np.sqrt(alphas_cumprod),
            sqrt_one_minus_alphas_cumprod=np.sqrt(1.0 - alphas_cumprod),
        )


def combine_cfg(v_cond: np.ndarray, v_uncond: np.ndarray, guidance_scale: float) -> np.ndarray:
    return v_uncond + guidance_scale * (v_cond - v_uncond)


def ddim_step(
    x: np.ndarray,
    v: np.ndarray,
    t: int,
    index: int,
    schedule: DdimSchedule,
    noise: np.ndarray,
) -> np.ndarray:
    """One p_sample_ddim update with v-parameterization."""
    sqrt_a_t_full = schedule.sqrt_alphas_cumprod[t]
    sqrt_1ma_t_full = schedule.sqrt_one_minus_alphas_cumprod[t]

    e_t = sqrt_a_t_full * v + sqrt_1ma_t_full * x
    pred_x0 = sqrt_a_t_full * x - sqrt_1ma_t_full * v

    a_prev = schedule.alphas_prev[index]
    sigma_t = schedule.sigmas[index]
    dir_xt = np.sqrt(np.maximum(1.0 - a_prev - sigma_t**2, 0.0)) * e_t
    return np.sqrt(a_prev) * pred_x0 + dir_xt + sigma_t * noise
