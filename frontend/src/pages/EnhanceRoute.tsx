import { useLocation } from "react-router-dom";
import { EnhancePage, isEnhanceMedium } from "./EnhancePage";

// Traduce el segmento de la URL al medio inicial. Un segmento invalido cae a
// imagen en vez de 404: es una pestaña, no un recurso. Lee el pathname y no
// useParams a proposito: la pagina vive montada FUERA de <Routes> (ver
// KeepMounted en App.tsx) y ahi no hay params.
export function EnhanceRoute() {
  const { pathname } = useLocation();
  const medium = pathname.split("/")[2];
  return <EnhancePage initialMedium={isEnhanceMedium(medium) ? medium : "image"} />;
}
