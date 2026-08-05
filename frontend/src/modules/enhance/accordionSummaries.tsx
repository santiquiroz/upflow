import type { DeviceInfoResponse, ModelResponse } from "../../lib/apiTypes";

// Funciones puras: reciben `t` en vez de llamar al hook.
export const SELECT_MODEL_PLACEHOLDER_KEY = "enhance.summary.selectModel";
export const SELECT_DEVICE_PLACEHOLDER_KEY = "enhance.summary.selectDevice";

export function formatModelSummary(model: ModelResponse | null, t: (key: string) => string) {
  if (!model) {
    return t(SELECT_MODEL_PLACEHOLDER_KEY);
  }
  if (!model.scale) {
    return model.name;
  }
  return (
    <>
      {model.name} · <span className="font-mono-tabular">{model.scale}x</span>
    </>
  );
}

export function formatDeviceSummary(device: DeviceInfoResponse | null, t: (key: string) => string) {
  return device ? device.name : t(SELECT_DEVICE_PLACEHOLDER_KEY);
}
