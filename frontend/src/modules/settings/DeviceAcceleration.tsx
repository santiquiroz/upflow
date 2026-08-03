import { useQuery } from "@tanstack/react-query";
import { Zap } from "lucide-react";
import { useTranslation } from "../../i18n/LocaleProvider";
import { getDevices } from "../../lib/api";
import type { DeviceInfoResponse } from "../../lib/apiTypes";

// La compilación del primer uso de un EP nativo dura segundos-minutos: mientras
// haya un device en "preparing" se sondea para que el estado nunca quede mudo.
const PREPARING_POLL_MS = 2000;

function hasPreparingDevice(devices: DeviceInfoResponse[] | undefined): boolean {
  return (devices ?? []).some((device) => device.epState === "preparing");
}

function EpStatusCell({ device }: { device: DeviceInfoResponse }) {
  const { t } = useTranslation();
  const label = device.epLabel || device.activeEp || "—";
  if (device.epState === "preparing") {
    return <span className="text-xs text-accent">{t("settings.acceleration.preparing")}</span>;
  }
  if (device.epState === "native") {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-ok">
        <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-ok" />
        {`${label} · ${t("settings.acceleration.native")}`}
      </span>
    );
  }
  if (device.epState === "ready") {
    // Registrado y elegible, pero todavia no lo uso ningun trabajo. Decir
    // "native" antes de eso podia anunciar un acelerador que despues nunca entra.
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-text">
        <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-text-faint" />
        {`${label} · ${t("settings.acceleration.ready")}`}
      </span>
    );
  }
  if (device.epState === "error") {
    // El motivo va VISIBLE, no solo en el tooltip: si alguien bajo el acelerador
    // y fallo, tiene que enterarse sin pasar el mouse por encima.
    return (
      <span className="flex flex-col gap-0.5 text-xs text-text" title={device.epDetail}>
        <span>{`${label} · ${t("settings.acceleration.fallback")}`}</span>
        {device.epDetail && <span className="text-[11px] text-warn">{device.epDetail}</span>}
      </span>
    );
  }
  return <span className="text-xs text-text">{label}</span>;
}

export function DeviceAcceleration() {
  const { t } = useTranslation();
  const devicesQuery = useQuery({
    queryKey: ["devices"],
    queryFn: getDevices,
    refetchInterval: (query) =>
      hasPreparingDevice(query.state.data?.devices) ? PREPARING_POLL_MS : false,
  });

  if (devicesQuery.isLoading) {
    return <p className="text-sm text-text-dim">Loading device info…</p>;
  }
  if (devicesQuery.isError) {
    return <p className="text-sm text-danger">Could not load device info.</p>;
  }

  const devices = devicesQuery.data?.devices ?? [];

  return (
    <div className="flex flex-col gap-3 rounded border border-border bg-surface p-4">
      <h2 className="flex items-center gap-1.5 font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
        <Zap aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
        {t("settings.acceleration.title")}
      </h2>
      <dl className="flex flex-col gap-2 text-sm">
        {devices.map((device) => (
          <div key={device.id} className="flex items-center justify-between gap-4">
            <dt className="text-text-dim">{device.name}</dt>
            <dd>
              <EpStatusCell device={device} />
            </dd>
          </div>
        ))}
      </dl>
      <p className="text-xs text-text-faint">{t("settings.acceleration.description")}</p>
    </div>
  );
}
