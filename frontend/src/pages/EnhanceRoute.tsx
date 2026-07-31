import { useLocation } from "react-router-dom";
import { EnhancePage } from "./EnhancePage";

type EnhanceMedium = "image" | "video";

function isMedium(value: string | undefined): value is EnhanceMedium {
  return value === "image" || value === "video";
}

// Traduce el segmento de la URL al medio inicial. Un segmento invalido cae a
// imagen en vez de 404: es una pestaña, no un recurso. Lee el pathname y no
// useParams a proposito: la pagina vive montada FUERA de <Routes> (ver
// KeepMounted en App.tsx) y ahi no hay params.
export function EnhanceRoute() {
  const { pathname } = useLocation();
  const medium = pathname.split("/")[2];
  return <EnhancePage initialMedium={isMedium(medium) ? medium : "image"} />;
}
