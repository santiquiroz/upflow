import { useEffect, useState, type ReactNode } from "react";
import { Route, Routes, useLocation } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { ForcedPasswordChangeModal } from "./components/ForcedPasswordChangeModal";
import { useAuth } from "./hooks/useAuth";
import { AudioPage } from "./modules/audio/AudioPage";
import { DownloadPage } from "./pages/DownloadPage";
import { EditorPage } from "./modules/editor/EditorPage";
import { GeneratePage } from "./modules/generate/GeneratePage";
import { TranscribePage } from "./modules/transcribe/TranscribePage";
import { EnhanceRoute } from "./pages/EnhanceRoute";
import { LoginPage } from "./pages/LoginPage";
import { ModelsPage } from "./pages/ModelsPage";
import { RealtimePage } from "./pages/RealtimePage";
import { SettingsPage } from "./pages/SettingsPage";
import { SetupPage } from "./pages/SetupPage";
import { TasksPage } from "./pages/TasksPage";
import { UsersPage } from "./pages/UsersPage";

// Una pestaña de trabajo queda MONTADA (oculta) al salir, para que el formulario,
// el archivo elegido y el job visible sobrevivan a la navegación — la queja real:
// "estoy escalando algo, paso a descargar un audio y se pierde todo". El montaje
// es perezoso: una pestaña nunca visitada no dispara sus consultas.
function KeepMounted({ active, children }: { active: boolean; children: ReactNode }) {
  const [everActive, setEverActive] = useState(active);
  useEffect(() => {
    if (active) {
      setEverActive(true);
    }
  }, [active]);
  if (!everActive) {
    return null;
  }
  return <div hidden={!active}>{children}</div>;
}

function AuthGate({ children }: { children: ReactNode }) {
  const { me, isLoading, isError, needsSetup } = useAuth();
  if (isLoading) {
    return null;
  }
  if (needsSetup) {
    return <SetupPage />;
  }
  if (isError && !me) {
    return <LoginPage />;
  }
  return <>{children}</>;
}

export function App() {
  const { me } = useAuth();
  const { pathname } = useLocation();

  return (
    <AuthGate>
      <AppShell>
        {/* Las pestañas de TRABAJO viven fuera de <Routes>: montadas y ocultas al
            salir (ver KeepMounted). /enhance sin medio abre en imagen, igual que
            antes. Las páginas de consulta (Tasks, Models, Settings...) siguen en
            <Routes> con el ciclo de vida normal. */}
        <KeepMounted active={pathname.startsWith("/enhance")}>
          <EnhanceRoute />
        </KeepMounted>
        <KeepMounted active={pathname === "/audio"}>
          <AudioPage />
        </KeepMounted>
        <KeepMounted active={pathname === "/transcribe"}>
          <TranscribePage />
        </KeepMounted>
        <KeepMounted active={pathname === "/download"}>
          <DownloadPage />
        </KeepMounted>
        <KeepMounted active={pathname === "/generate"}>
          <GeneratePage />
        </KeepMounted>
        <KeepMounted active={pathname === "/editor"}>
          <EditorPage />
        </KeepMounted>
        <Routes>
          <Route path="/" element={<TasksPage />} />
          <Route path="/enhance" element={null} />
          <Route path="/enhance/:medium" element={null} />
          <Route path="/audio" element={null} />
          <Route path="/transcribe" element={null} />
          <Route path="/download" element={null} />
          <Route path="/generate" element={null} />
          <Route path="/editor" element={null} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/realtime" element={<RealtimePage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/users" element={<UsersPage />} />
        </Routes>
      </AppShell>
      {me?.mustChangePassword && <ForcedPasswordChangeModal />}
    </AuthGate>
  );
}
