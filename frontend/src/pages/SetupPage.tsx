import { useState, type FormEvent } from "react";
import { ApiError } from "../lib/api";
import { setup } from "../services/auth";
import { useTranslation } from "../i18n/LocaleProvider";

export function SetupPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const { t } = useTranslation();

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await setup(username, password);
      setDone(true);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : t("auth.setup.failed"));
    } finally {
      setSubmitting(false);
    }
  }

  if (done) {
    return (
      <div className="flex h-screen items-center justify-center bg-bg">
        <p className="text-sm text-text">{t("auth.setup.done")}</p>
      </div>
    );
  }

  return (
    <div className="flex h-screen items-center justify-center bg-bg">
      <form onSubmit={handleSubmit} className="flex w-full max-w-sm flex-col gap-4 rounded border border-border bg-surface p-6">
        <h1 className="font-heading text-lg font-semibold text-text">{t("auth.setup.title")}</h1>
        <p className="text-xs text-text-dim">{t("auth.setup.subtitle")}</p>
        <label className="flex flex-col gap-1 text-xs text-text-dim">
          {t("auth.username")}
          <input
            required
            minLength={3}
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            className="rounded border border-border bg-surface-2 px-2 py-1 text-sm text-text"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-text-dim">
          {t("auth.password")}
          <input
            type="password"
            required
            minLength={8}
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            className="rounded border border-border bg-surface-2 px-2 py-1 text-sm text-text"
          />
        </label>
        {error && <p role="alert" className="text-xs text-danger">{error}</p>}
        <button
          type="submit"
          disabled={submitting}
          className="rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg disabled:opacity-50"
        >
          {t("auth.setup.submit")}
        </button>
      </form>
    </div>
  );
}
