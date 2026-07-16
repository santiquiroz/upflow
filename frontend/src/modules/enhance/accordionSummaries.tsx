import type { DeviceInfoResponse, ModelResponse } from "../../lib/apiTypes";

export const SELECT_MODEL_PLACEHOLDER = "Select a model…";
export const SELECT_DEVICE_PLACEHOLDER = "Select a device…";

export function formatModelSummary(model: ModelResponse | null) {
  if (!model) {
    return SELECT_MODEL_PLACEHOLDER;
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

export function formatDeviceSummary(device: DeviceInfoResponse | null) {
  return device ? device.name : SELECT_DEVICE_PLACEHOLDER;
}
