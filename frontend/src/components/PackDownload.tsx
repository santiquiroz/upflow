import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download } from "lucide-react";
import { useEffect, useState } from "react";
import { useTranslation } from "../i18n/LocaleProvider";
import type { ProvisionJob } from "../lib/apiTypes";
import { getProvisionStatus, provisionPack } from "../services/capabilities";

const POLL_MS = 2000;

function isFinished(job: ProvisionJob | undefined): boolean {
  return job?.status === "done" || job?.status === "error";
}

/**
 * Falta un paquete, y acá está el botón para bajarlo.
 *
 * Existe porque el botón vivía en UNA sola pantalla mientras 36 mensajes le
 * decían al usuario que abriera una terminal. Cualquier pantalla que sepa qué
 * paquete le falta puede poner esto y ya.
 */
export function PackDownload({
  pack,
  variant,
  reason,
  onDone,
}: {
  pack: string;
  /** Cuál del paquete, cuando el paquete es una familia (par de idiomas). */
  variant?: string;
  /** Qué falta, en palabras. Lo manda el backend. */
  reason?: string | null;
  onDone?: () => void;
}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [jobId, setJobId] = useState<string | null>(null);

  const start = useMutation({
    mutationFn: () => provisionPack(pack, variant),
    onSuccess: (job) => setJobId(job.jobId),
  });

  const status = useQuery({
    queryKey: ["pack-provision", jobId],
    queryFn: () => getProvisionStatus(jobId as string),
    enabled: jobId !== null,
    refetchInterval: (query) => (isFinished(query.state.data) ? false : POLL_MS),
  });

  // Al terminar bien hay que volver a preguntar: el estado de cada pantalla sale
  // de mirar el disco, y el disco recién cambió.
  const terminoBien = status.data?.status === "done";
  useEffect(() => {
    if (terminoBien) {
      queryClient.invalidateQueries();
      onDone?.();
    }
  }, [terminoBien, queryClient, onDone]);

  const trabajando = start.isPending || (jobId !== null && !isFinished(status.data));
  const error =
    (start.error instanceof Error ? start.error.message : null) ??
    (status.data?.status === "error" ? status.data.error : null);

  return (
    <div className="flex flex-col gap-2 rounded border border-border bg-surface-2 p-3">
      {reason && <p className="text-sm text-text-dim">{reason}</p>}

      {terminoBien ? (
        <p role="status" className="text-sm text-ok">
          {t("pack.done")}
        </p>
      ) : (
        <button
          type="button"
          onClick={() => start.mutate()}
          disabled={trabajando}
          className="inline-flex w-fit items-center gap-2 rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg hover:bg-accent-hover disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          <Download aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
          {trabajando ? t("pack.downloading") : t("pack.download")}
        </button>
      )}

      {trabajando && (
        <p role="status" className="text-xs text-text-faint">
          {t("pack.takesAWhile")}
        </p>
      )}

      {error && (
        <p role="alert" className="text-sm text-danger">
          {error}
        </p>
      )}
    </div>
  );
}
