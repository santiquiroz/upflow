import { useState } from "react";
import { useCreateUser, useUpdateUser, useUserJobs, useUsers } from "../hooks/useUsers";

function CreateUserForm() {
  const createUserMutation = useCreateUser();
  const [username, setUsername] = useState("");
  const [role, setRole] = useState("user");
  const [temporaryPassword, setTemporaryPassword] = useState<string | null>(null);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    const result = await createUserMutation.mutateAsync({ username, role });
    setTemporaryPassword(result.temporaryPassword);
    setUsername("");
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-wrap items-end gap-2 rounded border border-border bg-surface-2 p-3">
      <label className="flex flex-col gap-1 text-xs text-text-dim">
        Usuario
        <input
          required
          minLength={3}
          value={username}
          onChange={(event) => setUsername(event.target.value)}
          className="rounded border border-border bg-surface px-2 py-1 text-sm text-text"
        />
      </label>
      <label className="flex flex-col gap-1 text-xs text-text-dim">
        Rol
        <select
          value={role}
          onChange={(event) => setRole(event.target.value)}
          className="rounded border border-border bg-surface px-2 py-1 text-sm text-text"
        >
          <option value="user">user</option>
          <option value="admin">admin</option>
        </select>
      </label>
      <button type="submit" className="rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg">
        Crear usuario
      </button>
      {temporaryPassword && (
        <p className="w-full text-xs text-text-dim">
          Contraseña temporal para el nuevo usuario: <span className="font-mono-tabular text-text">{temporaryPassword}</span>
        </p>
      )}
    </form>
  );
}

function UserJobsPanel({ userId, onClose }: { userId: string; onClose: () => void }) {
  const { data } = useUserJobs(userId);
  return (
    <div className="rounded border border-border bg-surface-2 p-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-text-dim">Jobs</h3>
        <button type="button" onClick={onClose} className="text-xs text-text-dim hover:text-text">
          Cerrar
        </button>
      </div>
      <ul className="mt-2 flex flex-col gap-1">
        {(data?.jobs ?? []).map((job) => (
          <li key={job.id} className="text-xs text-text">
            {job.kind} — {job.originalFilename ?? job.id} — {job.status}
          </li>
        ))}
        {data && data.jobs.length === 0 && <li className="text-xs text-text-faint">Sin jobs.</li>}
      </ul>
    </div>
  );
}

export function UsersPage() {
  const { data, isLoading } = useUsers();
  const updateUserMutation = useUpdateUser();
  const [viewingJobsFor, setViewingJobsFor] = useState<string | null>(null);

  async function handleRoleChange(userId: string, role: string) {
    await updateUserMutation.mutateAsync({ userId, params: { role } });
  }

  async function handleToggleDisabled(userId: string, disabled: boolean) {
    await updateUserMutation.mutateAsync({ userId, params: { disabled: !disabled } });
  }

  async function handleResetPassword(userId: string) {
    await updateUserMutation.mutateAsync({ userId, params: { resetPassword: true } });
  }

  return (
    <div className="flex flex-col gap-4">
      <h1 className="font-heading text-xl font-semibold text-text">Users</h1>
      <CreateUserForm />
      {isLoading && <p className="text-sm text-text-faint">Cargando...</p>}
      {data && (
        <table className="w-full text-left text-sm text-text">
          <thead>
            <tr className="text-xs uppercase tracking-wide text-text-dim">
              <th className="p-2">Usuario</th>
              <th className="p-2">Rol</th>
              <th className="p-2">Estado</th>
              <th className="p-2">Uso hoy</th>
              <th className="p-2">Acciones</th>
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
                <td className="p-2">{user.disabled ? "Deshabilitado" : "Activo"}</td>
                <td className="p-2">{user.usedJobsToday} jobs / {user.usedGpuSecondsToday.toFixed(0)}s GPU</td>
                <td className="flex gap-2 p-2 text-xs">
                  <button type="button" onClick={() => void handleToggleDisabled(user.id, user.disabled)} className="text-accent hover:underline">
                    {user.disabled ? "Habilitar" : "Deshabilitar"}
                  </button>
                  <button type="button" onClick={() => void handleResetPassword(user.id)} className="text-accent hover:underline">
                    Reset password
                  </button>
                  <button type="button" onClick={() => setViewingJobsFor(user.id)} className="text-accent hover:underline">
                    Ver jobs
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
