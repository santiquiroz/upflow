import { Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { EnhancePage } from "./pages/EnhancePage";
import { ModelsPage } from "./pages/ModelsPage";
import { RealtimePage } from "./pages/RealtimePage";
import { SettingsPage } from "./pages/SettingsPage";

export function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<EnhancePage />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="/realtime" element={<RealtimePage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Routes>
    </AppShell>
  );
}
