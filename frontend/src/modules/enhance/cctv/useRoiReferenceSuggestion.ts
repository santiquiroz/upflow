import { useMutation } from "@tanstack/react-query";
import { useTranslation } from "../../../i18n/LocaleProvider";
import { suggestCctvRoiReference, type CctvRoiReferenceRequest } from "../../../services/cctv";
import type { RoiChoice } from "./cctvRoi";
import { errorInfoOf, errorText } from "./cctvText";

interface SuggestionInput {
  token: string;
  roi: RoiChoice;
  request: CctvRoiReferenceRequest | null;
  onChange: (roi: RoiChoice) => void;
  onShowFrame: (frame: number) => void;
}

export interface RoiReferenceSuggestion {
  canSuggest: boolean;
  suggesting: boolean;
  error: string | null;
  suggest: () => void;
}

// El sugerido se muestra enseguida: la caja se dibujo en otro cuadro y hay que ver que siga tapando la placa.
export function useRoiReferenceSuggestion({ token, roi, request, onChange, onShowFrame }: SuggestionInput): RoiReferenceSuggestion {
  const { t } = useTranslation();
  const mutation = useMutation({ mutationFn: (body: CctvRoiReferenceRequest) => suggestCctvRoiReference(token, body) });
  const info = errorInfoOf(mutation.error);

  function suggest(): void {
    if (request === null) {
      return;
    }
    mutation.mutate(request, {
      onSuccess: ({ referenceFrame }) => {
        onChange({ ...roi, reference: referenceFrame });
        onShowFrame(referenceFrame);
      },
    });
  }

  return {
    canSuggest: request !== null && !mutation.isPending,
    suggesting: mutation.isPending,
    error: info ? errorText(t, info) : null,
    suggest,
  };
}
