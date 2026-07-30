import type { ReactNode } from "react";
import { Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { ForcedPasswordChangeModal } from "./components/ForcedPasswordChangeModal";
import { useAuth } from "./hooks/useAuth";
import { AudioPage } from "./modules/audio/AudioPage";
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

  return (
    <AuthGate>
      <AppShell>
        <Routes>
          <Route path="/" element={<TasksPage />} />
          {/* /enhance sin medio abre en imagen, igual que antes. */}
          <Route path="/enhance" element={<EnhanceRoute />} />
          <Route path="/enhance/:medium" element={<EnhanceRoute />} />
          <Route path="/audio" element={<AudioPage />} />
          <Route path="/transcribe" element={<TranscribePage />} />
          <Route path="/generate" element={<GeneratePage />} />
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
