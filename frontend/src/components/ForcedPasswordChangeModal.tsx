import { useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { ApiError } from "../lib/api";
import { changePassword } from "../services/auth";
import { useTranslation } from "../i18n/LocaleProvider";
import { Modal } from "./Modal";

export function ForcedPasswordChangeModal() {
  const queryClient = useQueryClient();
  const { t } = useTranslation();
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await changePassword(currentPassword, newPassword);
      await queryClient.invalidateQueries({ queryKey: ["me"] });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : t("auth.password.change.failed"));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Modal titleId="forced-password-change-title" onClose={() => undefined}>
      <h2 id="forced-password-change-title" className="font-heading text-base font-semibold text-text">
        {t("auth.password.change.title")}
      </h2>
      <form onSubmit={handleSubmit} className="flex flex-col gap-3">
        <label className="flex flex-col gap-1 text-xs text-text-dim">
          {t("auth.password.current")}
          <input
            type="password"
            required
            value={currentPassword}
            onChange={(event) => setCurrentPassword(event.target.value)}
            className="rounded border border-border bg-surface-2 px-2 py-1 text-sm text-text"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-text-dim">
          {t("auth.password.new")}
          <input
            type="password"
            required
            minLength={8}
            value={newPassword}
            onChange={(event) => setNewPassword(event.target.value)}
            className="rounded border border-border bg-surface-2 px-2 py-1 text-sm text-text"
          />
        </label>
        {error && <p role="alert" className="text-xs text-danger">{error}</p>}
        <button
          type="submit"
          disabled={submitting}
          className="rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg disabled:opacity-50"
        >
          {t("auth.password.change.submit")}
        </button>
      </form>
    </Modal>
  );
}
