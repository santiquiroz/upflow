import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from peft.tuners.tuners_utils import BaseTunerLayer
from PIL import Image

LORA_MODULES = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
                "conv_shortcut", "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"]
LOAD = dict(variant="fp16", torch_dtype=torch.float32, local_files_only=True)
NOTICE = """OSOR code: MIT; https://github.com/Zhouqm-Git/osor
OSOR weights: https://huggingface.co/QinmingZhou/OSOR; CreativeML Open RAIL++-M
Base: diffusers/stable-diffusion-xl-1.0-inpainting-0.1; CreativeML Open RAIL++-M
Este paquete contiene el UNet con LoRA OSOR fusionada, VAE y embeddings locales.
"""


def expand_output(unet):
    old = unet.conv_out
    new = torch.nn.Conv2d(old.in_channels, 5, old.kernel_size, old.stride, old.padding)
    with torch.no_grad():
        new.weight[:4].copy_(old.weight)
        new.bias[:4].copy_(old.bias)
        torch.nn.init.kaiming_normal_(new.weight[4:], nonlinearity="relu")
        new.bias[4:].zero_()
    unet.conv_out = new
    unet.register_to_config(out_channels=5)


def remap_checkpoint(raw, expected):
    clean, remapped = {}, {}
    for original, value in raw.items():
        key = original.replace("module.", "").replace("_orig_mod.", "")
        if key not in expected:
            key = key.replace(".lora_A.weight", ".lora_A.default.weight")
            key = key.replace(".lora_B.weight", ".lora_B.default.weight")
        if key in clean:
            raise ValueError(f"Colisión de claves: {original} -> {key}")
        clean[key] = value
        if original != key:
            remapped[original] = key
            print(f"Remapeo: {original} -> {key}", flush=True)
    return clean, remapped


def load_checkpoint(unet, path):
    wrapper = torch.nn.Module()
    wrapper.add_module("unet", unet)
    expected = wrapper.state_dict()
    state, remapped = remap_checkpoint(torch.load(path, map_location="cpu", weights_only=True, mmap=True), expected)
    trainable = {name for name, param in wrapper.named_parameters() if param.requires_grad}
    result = wrapper.load_state_dict(state, strict=False)
    required = [k for k in result.missing_keys if "lora_" in k or "conv_out" in k]
    mismatches = [k for k, v in state.items() if k in expected and
                  not torch.equal(expected[k], v.to(dtype=expected[k].dtype))]
    errors = dict(unexpected=result.unexpected_keys, required=required,
                  absent_trainable=sorted(trainable - state.keys()), mismatches=mismatches)
    if any(errors.values()):
        raise ValueError(f"Checkpoint inválido: {errors}")
    print(f"Checkpoint: {len(state)} tensores verificados; {len(result.missing_keys)} claves congeladas ausentes", flush=True)
    return dict(remapped_keys=remapped, verified_tensors=len(state), missing_frozen=len(result.missing_keys))


def conditioning(path, height=512, width=512):
    data = torch.load(path, map_location="cpu", weights_only=True)
    return dict(encoder_hidden_states=data["prompt_embeds"].float(), added_cond_kwargs={
        "text_embeds": data["pooled_prompt_embeds"].float(),
        "time_ids": torch.tensor([[height, width, 0, 0, height, width]], dtype=torch.float32)})


def compare_tensors(label, actual, expected):
    difference = (actual - expected).abs()
    if not torch.isfinite(difference).all():
        raise ValueError(f"{label}: valores no finitos")
    result = dict(max_abs=difference.max().item(), mean_abs=difference.mean().item())
    print(f"{label}: {result}", flush=True)
    return result


@torch.inference_mode()
def merge_and_check(unet, embeds):
    torch.manual_seed(42)
    latent, timestep = torch.randn(1, 9, 64, 64), torch.tensor([400])
    cond = conditioning(embeds)
    original = unet(latent, timestep, **cond).sample
    unet.fuse_lora(safe_fusing=True)
    unet.unload_lora()
    if any(isinstance(module, BaseTunerLayer) for module in unet.modules()):
        raise ValueError("Persisten capas PEFT después de fusionar")
    full = compare_tensors("Paridad 1 fp32", unet(latent, timestep, **cond).sample, original)
    if full["mean_abs"] > 1e-3:
        raise ValueError("La fusión excede la tolerancia fp32")
    unet.half().float()
    rounded = compare_tensors("Paridad 1 fp16 -> fp32", unet(latent, timestep, **cond).sample, original)
    return dict(fp32=full, fp16_roundtrip=rounded)


def save_pack(unet, args, partial):
    unet.half().save_pretrained(partial / "unet", variant="fp16", safe_serialization=True)
    for relative in ("vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
                     "scheduler/scheduler_config.json"):
        destination = partial / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.base / relative, destination)
    shutil.copy2(args.embeds, partial / "prompt_embeds.pt")
    (partial / "NOTICE.txt").write_text(NOTICE, encoding="utf-8")


def calc_dims(width, height):
    if min(width, height) < 512:
        return (width + 15) // 16 * 16, (height + 15) // 16 * 16
    if width < height:
        return 512, (int(height * 512 / width) + 15) // 16 * 16
    return (int(width * 512 / height) + 15) // 16 * 16, 512


def image_tensor(path, size, mode, resample):
    with Image.open(path) as image:
        array = np.array(image.convert(mode).resize(size, resample), dtype=np.float32) / 255
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


@torch.inference_mode()
def infer_saved(pack, case):
    unet = UNet2DConditionModel.from_pretrained(pack / "unet", **LOAD).eval()
    vae = AutoencoderKL.from_pretrained(pack / "vae", **LOAD).eval()
    scheduler = DDPMScheduler.from_pretrained(pack / "scheduler", local_files_only=True)
    with Image.open(f"{case}_img.png") as image:
        size = calc_dims(*image.size)
    lq = image_tensor(f"{case}_img.png", size, "RGB", Image.Resampling.BICUBIC) * 2 - 1
    mask = image_tensor(f"{case}_mask.png", size, "L", Image.Resampling.NEAREST)
    torch.manual_seed(42)
    z_lq = vae.encode(lq).latent_dist.sample() * vae.config.scaling_factor
    timestep = torch.tensor([400])
    z = scheduler.add_noise(z_lq, torch.randn_like(z_lq), timestep)
    z_in = torch.cat([z, F.max_pool2d(mask, 8, 8), z_lq], dim=1)
    output = unet(z_in, timestep, **conditioning(pack / "prompt_embeds.pt", size[1], size[0])).sample
    alpha = output[:, 4:].sigmoid()
    restored = scheduler.step(output[:, :4], timestep, z).pred_original_sample
    restored = restored * alpha + z_lq * (1 - alpha)
    decoded = vae.decode(restored / vae.config.scaling_factor).sample
    pixels = ((decoded + 1) / 2).clamp(0, 1)[0].permute(1, 2, 0)
    opacity = F.interpolate(alpha, size=(size[1], size[0]), mode="bilinear")[0, 0]
    return pixels, opacity


def check_case(pack, case):
    pixels, opacity = infer_saved(pack, case)
    results = {}
    for name, tensor, method in (("osor", pixels, Image.Resampling.LANCZOS),
                                 ("alpha", opacity, Image.Resampling.BILINEAR)):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Salida no finita: {name}")
        actual = Image.fromarray((tensor.numpy() * 255).astype(np.uint8)).resize((1024, 1024), method)
        with Image.open(case.parent.parent / "out" / f"{case.name}_{name}512.png") as reference:
            results[name] = compare_tensors(f"Paridad 2 {name} (0-255)",
                torch.from_numpy(np.array(actual).astype(np.float32)),
                torch.from_numpy(np.array(reference).astype(np.float32)))
    if results["osor"]["mean_abs"] > 3:
        print("ALERTA: paridad 2 imagen supera 3 unidades", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    for name in ("base", "weights", "embeds", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--check-case", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(8)
    partial = args.out.with_name(args.out.name + ".partial")
    if args.out.exists() or partial.exists():
        raise FileExistsError("El destino o su directorio .partial ya existe")
    partial.mkdir(parents=True)
    torch.manual_seed(42)
    unet = UNet2DConditionModel.from_pretrained(args.base / "unet", **LOAD)
    unet.add_adapter(LoraConfig(r=256, lora_alpha=256, init_lora_weights="gaussian", target_modules=LORA_MODULES))
    expand_output(unet)
    report = dict(checkpoint=load_checkpoint(unet, args.weights))
    unet.eval().requires_grad_(False)
    report["parity1"] = merge_and_check(unet, args.embeds)
    save_pack(unet, args, partial)
    del unet
    gc.collect()
    if args.check_case:
        report["parity2"] = check_case(partial, args.check_case)
    report["wall_seconds"] = time.perf_counter() - started
    (partial / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    partial.rename(args.out)
    size = sum(path.stat().st_size for path in args.out.rglob("*") if path.is_file())
    print(f"Paquete: {args.out}; tamaño: {size} bytes; tiempo: {report['wall_seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
