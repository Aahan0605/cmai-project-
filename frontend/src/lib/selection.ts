export function positiveId(value: string | null): number | null {
  if (value == null || !/^\d+$/.test(value)) return null;
  const id = Number(value);
  return Number.isSafeInteger(id) && id > 0 ? id : null;
}
export function hasInterval(
  values: (number | null)[],
): values is [number, number] {
  return (
    values.length === 2 && values.every((n) => n != null && Number.isFinite(n))
  );
}
export function initialDarkTheme(): boolean {
  const theme = document.documentElement.dataset.theme;
  return (
    theme === "dark" ||
    (theme !== "light" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches)
  );
}
