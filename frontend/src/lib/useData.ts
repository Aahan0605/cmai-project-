import { useQuery } from "@tanstack/react-query";
import { DATA_MODE, fetchJson } from "./api";

export function useData<T>(path: string, enabled = true) {
  return useQuery({
    queryKey: [path],
    queryFn: () => fetchJson<T>(path),
    enabled,
    refetchInterval: DATA_MODE === "api" ? 15000 : false,
  });
}
