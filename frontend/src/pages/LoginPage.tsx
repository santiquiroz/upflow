import { useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { ApiError } from "../lib/api";
import { login } from "../services/auth";

export function LoginPage() {
  const queryClient = useQueryClient();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await login(username, password);
      await queryClient.invalidateQueries({ queryKey: ["me"] });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "No se pudo iniciar sesión");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex h-screen items-center justify-center bg-bg">
      <form onSubmit={handleSubmit} className="flex w-full max-w-sm flex-col gap-4 rounded border border-border bg-surface p-6">
        <h1 className="font-heading text-lg font-semibold text-text">Upflow</h1>
        <label className="flex flex-col gap-1 text-xs text-text-dim">
          Usuario
          <input
            required
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            className="rounded border border-border bg-surface-2 px-2 py-1 text-sm text-text"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-text-dim">
          Contraseña
          <input
            type="password"
            required
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
          Ingresar
        </button>
      </form>
    </div>
  );
}
