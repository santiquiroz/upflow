import { useState } from "react";
import { ApiError } from "../lib/api";
import { useCreateUser, useUpdateUser, useUserJobs, useUsers } from "../hooks/useUsers";
import { useTranslation } from "../i18n/LocaleProvider";

function CreateUserForm() {
  const createUserMutation = useCreateUser();
  const { t } = useTranslation();
  const [username, setUsername] = useState("");
  const [role, setRole] = useState("user");
  const [temporaryPassword, setTemporaryPassword] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      const result = await createUserMutation.mutateAsync({ username, role });
      setTemporaryPassword(result.temporaryPassword);
      setUsername("");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : t("users.create.failed"));
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-wrap items-end gap-2 rounded border border-border bg-surface-2 p-3">
      <label className="flex flex-col gap-1 text-xs text-text-dim">
        {t("auth.username")}
        <input
          required
          minLength={3}
          value={username}
          onChange={(event) => setUsername(event.target.value)}
          className="rounded border border-border bg-surface px-2 py-1 text-sm text-text"
        />
      </label>
      <label className="flex flex-col gap-1 text-xs text-text-dim">
        {t("users.role")}
        <select
          value={role}
          onChange={(event) => setRole(event.target.value)}
          className="rounded border border-border bg-surface px-2 py-1 text-sm text-text"
        >
          <option value="user">user</option>
          <option value="admin">admin</option>
        </select>
      </label>
      <button
        type="submit"
        disabled={createUserMutation.isPending}
        className="rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg disabled:opacity-50"
      >
        {t("users.create.submit")}
      </button>
      {error && <p role="alert" className="w-full text-xs text-danger">{error}</p>}
      {temporaryPassword && (
        <p className="w-full text-xs text-text-dim">
          {t("users.temporaryPassword")} <span className="font-mono-tabular text-text">{temporaryPassword}</span>
        </p>
      )}
    </form>
  );
}

function UserJobsPanel({ userId, onClose }: { userId: string; onClose: () => void }) {
  const { data } = useUserJobs(userId);
  const { t } = useTranslation();
  return (
    <div className="rounded border border-border bg-surface-2 p-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-text-dim">{t("users.jobs.title")}</h3>
        <button type="button" onClick={onClose} className="text-xs text-text-dim hover:text-text">
          {t("common.close")}
        </button>
      </div>
      <ul className="mt-2 flex flex-col gap-1">
        {(data?.jobs ?? []).map((job) => (
          <li key={job.id} className="text-xs text-text">
            {job.kind} — {job.originalFilename ?? job.id} — {job.status}
          </li>
        ))}
        {data && data.jobs.length === 0 && <li className="text-xs text-text-faint">{t("users.jobs.empty")}</li>}
      </ul>
    </div>
  );
}

export function UsersPage() {
  const { data, isLoading } = useUsers();
  const { t } = useTranslation();
  const updateUserMutation = useUpdateUser();
  const [viewingJobsFor, setViewingJobsFor] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  async function runUpdate(userId: string, params: Parameters<typeof updateUserMutation.mutateAsync>[0]["params"]) {
    setActionError(null);
    try {
      await updateUserMutation.mutateAsync({ userId, params });
    } catch (err) {
      setActionError(err instanceof ApiError ? err.message : t("users.update.failed"));
    }
  }

  async function handleRoleChange(userId: string, role: string) {
    await runUpdate(userId, { role });
  }

  async function handleToggleDisabled(userId: string, disabled: boolean) {
    await runUpdate(userId, { disabled: !disabled });
  }

  async function handleResetPassword(userId: string) {
    await runUpdate(userId, { resetPassword: true });
  }

  return (
    <div className="flex flex-col gap-4">
      <h1 className="font-heading text-xl font-semibold text-text">Users</h1>
      <CreateUserForm />
      {actionError && <p role="alert" className="text-xs text-danger">{actionError}</p>}
      {isLoading && <p className="text-sm text-text-faint">{t("common.loading")}</p>}
      {data && (
        <table className="w-full text-left text-sm text-text">
          <thead>
            <tr className="text-xs uppercase tracking-wide text-text-dim">
              <th className="p-2">{t("auth.username")}</th>
              <th className="p-2">{t("users.role")}</th>
              <th className="p-2">{t("users.status")}</th>
              <th className="p-2">{t("users.usageToday")}</th>
              <th className="p-2">{t("users.actions")}</th>
            </tr>
          </thead>
          <tbody>
            {data.users.map((user) => (
              <tr key={user.id} className="border-t border-border">
                <td className="p-2">{user.username}</td>
                <td className="p-2">
                  <select
                    value={user.role}
                    onChange={(event) => void handleRoleChange(user.id, event.target.value)}
                    className="rounded border border-border bg-surface-2 px-1 py-0.5 text-xs text-text"
                  >
                    <option value="user">user</option>
                    <option value="admin">admin</option>
                  </select>
                </td>
                <td className="p-2">{user.disabled ? t("users.status.disabled") : t("users.status.active")}</td>
                <td className="p-2">{user.usedJobsToday} jobs / {user.usedGpuSecondsToday.toFixed(0)}s GPU</td>
                <td className="flex gap-2 p-2 text-xs">
                  <button type="button" onClick={() => void handleToggleDisabled(user.id, user.disabled)} className="text-accent hover:underline">
                    {user.disabled ? t("users.action.enable") : t("users.action.disable")}
                  </button>
                  <button type="button" onClick={() => void handleResetPassword(user.id)} className="text-accent hover:underline">
                    {t("users.action.resetPassword")}
                  </button>
                  <button type="button" onClick={() => setViewingJobsFor(user.id)} className="text-accent hover:underline">
                    {t("users.action.viewJobs")}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {viewingJobsFor && <UserJobsPanel userId={viewingJobsFor} onClose={() => setViewingJobsFor(null)} />}
    </div>
  );
}
