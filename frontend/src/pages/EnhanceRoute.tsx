import { useParams } from "react-router-dom";
import { EnhancePage } from "./EnhancePage";

type EnhanceMedium = "image" | "video";

function isMedium(value: string | undefined): value is EnhanceMedium {
  return value === "image" || value === "video";
}

// Traduce el segmento de la URL al medio inicial. Un segmento invalido cae a
// imagen en vez de 404: es una pestaña, no un recurso.
export function EnhanceRoute() {
  const { medium } = useParams<{ medium: string }>();
  return <EnhancePage initialMedium={isMedium(medium) ? medium : "image"} />;
}
