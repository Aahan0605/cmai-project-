// Both modes use the same Python query layer and JSON contract.
export const DATA_MODE: "api" | "snapshot" =
  import.meta.env.VITE_DATA_MODE === "api" ? "api" : "snapshot";
const apiBase = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
export function url(path: string): string {
  return DATA_MODE === "api"
    ? `${apiBase}/api/${path}`
    : `${import.meta.env.BASE_URL}data/${path}.json`;
}
export async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(url(path));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${path}`);
  if (!res.headers.get("content-type")?.includes("json")) {
    throw new Error(
      `Expected JSON for ${path}. Export the snapshot or check the API URL`,
    );
  }
  return (await res.json()) as T;
}
