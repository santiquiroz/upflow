from __future__ import annotations

# Única fuente del formato de device DirectML ("dml:N"): antes había 3
# parsers divergentes (onnx_upscaler, ep_registry, resource_probes) que
# podían desalinearse en el manejo de errores.

DML_DEVICE_PREFIX = "dml:"


def parse_dml_device_id(device: str) -> int:
    device_id = try_parse_dml_device_id(device)
    if device_id is None:
        raise ValueError(
            f"Invalid DirectML device id {device!r}: expected '{DML_DEVICE_PREFIX}<adapter-index>'"
        )
    return device_id


def try_parse_dml_device_id(device: str) -> int | None:
    if not device.startswith(DML_DEVICE_PREFIX):
        return None
    try:
        return int(device[len(DML_DEVICE_PREFIX):])
    except ValueError:
        return None
