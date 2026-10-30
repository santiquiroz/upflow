import { useQuery } from "@tanstack/react-query";

import { ApiError } from "../../lib/api";
import type { InitImageResponse } from "../../lib/apiTypes";
import { detectPeople } from "../../services/editor";

export function usePeople(baseInfo: InitImageResponse | null, enabled: boolean) {
  return useQuery({
    queryKey: ["editorPeople", baseInfo?.initImageToken],
    queryFn: () => {
      if (!baseInfo) {
        throw new Error("Cannot detect people without an image");
      }
      return detectPeople(baseInfo.initImageToken);
    },
    enabled: Boolean(baseInfo) && enabled,
    retry: (failureCount, error) =>
      error instanceof ApiError && error.status === 503 ? false : failureCount < 3,
  });
}
