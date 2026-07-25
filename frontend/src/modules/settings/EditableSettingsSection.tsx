import { useState, type FormEvent } from "react";
import { useEditableSettings } from "../../hooks/useEditableSettings";

const HF_TOKEN_KEY = "hf_token";

function ConfigurationBadge({ configured }: { configured: boolean }) {
  const toneClassName = configured ? "text-ok" : "text-danger";
  return (
    <span className={`flex items-center gap-1.5 text-xs ${toneClassName}`}>
      <span
        aria-hidden="true"
        className={`h-1.5 w-1.5 rounded-full ${configured ? "bg-ok" : "bg-danger"}`}
      />
      {configured ? "Configured" : "Not configured"}
    </span>
  );
}

export function EditableSettingsSection() {
  const [token, setToken] = useState("");
  const { settingsQuery, patchMutation } = useEditableSettings();
  const configured = settingsQuery.data?.settings.find((setting) => setting.key === HF_TOKEN_KEY)?.configured ?? false;

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (token.trim().length === 0) {
      return;
    }
    patchMutation.mutate(
      { key: HF_TOKEN_KEY, value: token },
      {
        onSuccess: () => setToken(""),
      },
    );
  }

  return (
    <div className="flex flex-col gap-3 rounded border border-border bg-surface p-4">
      <h2 className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">Credentials</h2>
      <div className="flex items-center justify-between gap-4">
        <label htmlFor="hf-token" className="text-sm text-text">
          Hugging Face token
        </label>
        {settingsQuery.isLoading ? (
          <span className="text-xs text-text-faint">Loading…</span>
        ) : (
          <ConfigurationBadge configured={configured} />
        )}
      </div>
      <form onSubmit={handleSubmit} className="flex items-center gap-2">
        <input
          id="hf-token"
          type="password"
          autoComplete="off"
          value={token}
          onChange={(event) => {
            setToken(event.target.value);
            patchMutation.reset();
          }}
          placeholder="Enter a new token"
          className="w-full rounded border border-border bg-surface px-3 py-2 text-sm text-text placeholder:text-text-faint focus:border-accent focus:outline-none disabled:opacity-60"
        />
        <button
          type="submit"
          disabled={token.trim().length === 0 || patchMutation.isPending}
          className="inline-flex shrink-0 items-center gap-2 rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg transition-[background-color] duration-fast hover:bg-accent-hover active:bg-accent-press disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          {patchMutation.isPending ? "Saving…" : "Save"}
        </button>
      </form>
      {settingsQuery.isError && <p className="text-xs text-danger">Could not load credential status.</p>}
      {patchMutation.isSuccess && <p className="text-xs text-ok">Saved</p>}
      {patchMutation.isError && (
        <p role="alert" className="text-xs text-danger">
          {patchMutation.error instanceof Error ? patchMutation.error.message : "Could not save the token."}
        </p>
      )}
    </div>
  );
}
