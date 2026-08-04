import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "../../i18n/LocaleProvider";
import { Info } from "lucide-react";
import { getDevices } from "../../lib/api";
import type { DeviceInfoResponse } from "../../lib/apiTypes";

function findDefaultDevice(devices: DeviceInfoResponse[], defaultDeviceId: string): DeviceInfoResponse | null {
  return devices.find((device) => device.id === defaultDeviceId) ?? null;
}

function DefaultDeviceName({ device }: { device: DeviceInfoResponse | null }) {
  const { t } = useTranslation();
  if (!device) {
    return <p className="text-sm text-text-faint">{t("models.device.none")}</p>;
  }
  return <p className="text-sm text-text">{device.name}</p>;
}

export function DeviceDefault() {
  const { t } = useTranslation();
  const devicesQuery = useQuery({ queryKey: ["devices"], queryFn: getDevices });

  if (devicesQuery.isLoading) {
    return <p className="text-sm text-text-dim">{t("models.device.loading")}</p>;
  }

  if (devicesQuery.isError) {
    return <p className="text-sm text-danger">{t("models.device.loadError")}</p>;
  }

  const devices = devicesQuery.data?.devices ?? [];
  const defaultDeviceId = devicesQuery.data?.defaultDeviceId ?? "";
  const defaultDevice = findDefaultDevice(devices, defaultDeviceId);

  return (
    <div className="flex flex-col gap-2 rounded border border-border bg-surface p-4">
      <h2 className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">{t("models.device.title")}</h2>
      <DefaultDeviceName device={defaultDevice} />
      <div className="flex items-start gap-2 text-xs text-text-faint">
        <Info aria-hidden="true" className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
        <span>{t("models.device.info")}</span>
      </div>
    </div>
  );
}
