import { useNavigate } from "react-router-dom";
import { useCapabilityActivation } from "../hooks/useCapabilityActivation";
import { useCapabilityProvision } from "../hooks/useCapabilityProvision";
import { useTranslation } from "../i18n/LocaleProvider";
import type { CapabilityResponse } from "../lib/apiTypes";
import { CapabilityTree } from "../modules/capabilities/CapabilityTree";
import { surfaceFor } from "../modules/capabilities/capabilityRoutes";

function ProvisionBanner({
  capabilityId,
  isBusy,
  errorMessage,
  isDone,
}: {
  capabilityId: string | null;
  isBusy: boolean;
  errorMessage: string | null;
  isDone: boolean;
}) {
  const { t } = useTranslation();

  if (capabilityId === null) {
    return null;
  }
  if (errorMessage) {
    return (
      <p role="alert" className="text-sm text-danger">
        {t("capability.provision.failed", { error: errorMessage })}
      </p>
    );
  }
  if (isBusy) {
    return (
      <p role="status" className="text-sm text-text-dim">
        {t("capability.provision.running")}
      </p>
    );
  }
  if (isDone) {
    return (
      <p role="status" className="text-sm text-ok">
        {t("capability.provision.done")}
      </p>
    );
  }
  return null;
}

function ActivationBanner({
  capabilityId,
  isBusy,
  errorMessage,
  isDone,
}: {
  capabilityId: string | null;
  isBusy: boolean;
  errorMessage: string | null;
  isDone: boolean;
}) {
  const { t } = useTranslation();

  if (capabilityId === null) {
    return null;
  }
  if (errorMessage) {
    return (
      <p role="alert" className="text-sm text-danger">
        {t("capability.activation.failed", { error: errorMessage })}
      </p>
    );
  }
  if (isBusy) {
    return (
      <p role="status" className="text-sm text-text-dim">
        {t("capability.activation.running")}
      </p>
    );
  }
  if (isDone) {
    return (
      <p role="status" className="text-sm text-ok">
        {t("capability.activation.done")}
      </p>
    );
  }
  return null;
}

export function TasksPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const provision = useCapabilityProvision();
  const activation = useCapabilityActivation();

  function handleSelect(capability: CapabilityResponse): void {
    const surface = surfaceFor(capability);
    if (surface !== null) {
      navigate(surface);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="font-heading text-2xl font-semibold text-text">
          {t("tasks.title")}
        </h1>
        <p className="mt-1 text-sm text-text-dim">{t("tasks.subtitle")}</p>
      </div>
      <ProvisionBanner
        capabilityId={provision.capabilityId}
        isBusy={provision.isBusy}
        errorMessage={provision.errorMessage}
        isDone={provision.job?.status === "done"}
      />
      <ActivationBanner
        capabilityId={activation.capabilityId}
        isBusy={activation.isBusy}
        errorMessage={activation.errorMessage}
        isDone={activation.isDone}
      />
      <CapabilityTree
        onSelect={handleSelect}
        onProvision={(capability) => provision.provision(capability.id)}
        onActivate={activation.activate}
      />
    </div>
  );
}
