import { useQuery } from "@tanstack/react-query";
import { getDevices } from "../lib/api";
import type { DeviceInfoResponse } from "../lib/apiTypes";

export const AUTO_DEVICE_ID = "auto";

// Synthetic entry, never fetched from GET /devices -- selecting it sends
// device="auto" to the backend, which routes to a free compatible device at
// dequeue time (see app/services/device_router.py). Never disabled: real
// compatibility is enforced server-side, surfaced as a submit-time error.
export const AUTO_DEVICE: DeviceInfoResponse = {
  id: AUTO_DEVICE_ID,
  kind: "auto",
  name: "Auto",
  backend: "auto",
};

interface DevicePickerProps {
  value: string | null;
  onChange: (device: DeviceInfoResponse) => void;
  requiresGpu: boolean;
}

function isCpuDevice(device: DeviceInfoResponse): boolean {
  return device.kind === "cpu";
}

function deviceOptionClassName(isSelected: boolean, isDisabled: boolean): string {
  const base =
    "flex cursor-pointer flex-col gap-1 rounded border px-3 py-2 transition-[background-color,border-color] duration-fast focus-within:outline focus-within:outline-2 focus-within:outline-accent";
  if (isDisabled) {
    return `${base} cursor-not-allowed border-border bg-surface opacity-50`;
  }
  if (isSelected) {
    return `${base} border-accent bg-surface-2`;
  }
  return `${base} border-border bg-surface hover:border-text-faint`;
}

function DeviceOption({
  device,
  isSelected,
  isDefault,
  isDisabled,
  onChange,
}: {
  device: DeviceInfoResponse;
  isSelected: boolean;
  isDefault: boolean;
  isDisabled: boolean;
  onChange: (device: DeviceInfoResponse) => void;
}) {
  return (
    <label className={deviceOptionClassName(isSelected, isDisabled)}>
      <span className="flex items-center gap-2">
        <input
          type="radio"
          name="device"
          value={device.id}
          checked={isSelected}
          disabled={isDisabled}
          onChange={() => onChange(device)}
          className="h-3.5 w-3.5 accent-accent"
        />
        <span className="text-sm text-text">{device.name}</span>
        {isDefault && (
          <span className="rounded-sm bg-surface-2 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-accent">
            Default
          </span>
        )}
      </span>
      {isDisabled && (
        <span className="pl-[22px] text-xs text-warn">Requires a Vulkan GPU for this model (ncnn)</span>
      )}
      {device.id === AUTO_DEVICE_ID && (
        <span className="pl-[22px] text-xs text-text-faint">Routes to the least busy compatible device</span>
      )}
    </label>
  );
}

export function DevicePicker({ value, onChange, requiresGpu }: DevicePickerProps) {
  const devicesQuery = useQuery({ queryKey: ["devices"], queryFn: getDevices });

  if (devicesQuery.isLoading) {
    return <p className="text-sm text-text-dim">Loading devices…</p>;
  }

  if (devicesQuery.isError) {
    return <p className="text-sm text-danger">Could not load devices.</p>;
  }

  const devices = devicesQuery.data?.devices ?? [];
  const defaultDeviceId = devicesQuery.data?.defaultDeviceId;
  // Auto is never disabled here -- real compatibility (e.g. a builtin ncnn
  // model with no GPU present at all) is enforced server-side and surfaces
  // as a submit-time error instead.
  const options = [AUTO_DEVICE, ...devices];

  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">Device</legend>
      {options.map((device) => (
        <DeviceOption
          key={device.id}
          device={device}
          isSelected={device.id === value}
          isDefault={device.id === defaultDeviceId}
          isDisabled={device.id !== AUTO_DEVICE_ID && requiresGpu && isCpuDevice(device)}
          onChange={onChange}
        />
      ))}
    </fieldset>
  );
}
