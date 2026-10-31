import { useQuery } from "@tanstack/react-query";
import { fetchEditorCapabilities } from "../../services/editor";

export const EDITOR_CAPABILITIES_KEY = ["editorCapabilities"];

export function useEditorCapabilities() {
  return useQuery({ queryKey: EDITOR_CAPABILITIES_KEY, queryFn: fetchEditorCapabilities });
}
