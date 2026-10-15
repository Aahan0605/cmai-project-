/** CSV quoting also neutralizes spreadsheet formulas in text cells. */
export function csvCell(value: string | number | null | undefined): string {
  if (value == null) return "";
  let text = String(value);
  if (typeof value === "string" && /^[\s]*[=+@-]/.test(text)) text = `'${text}`;
  return `"${text.replaceAll('"', '""')}"`;
}
export function downloadCsv(
  name: string,
  rows: (string | number | null | undefined)[][],
) {
  const body = rows.map((row) => row.map(csvCell).join(",")).join("\r\n");
  const url = URL.createObjectURL(
    new Blob([body], { type: "text/csv;charset=utf-8" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function comparisonLabel(significant: boolean, count?: number): string {
  if (count != null && count < 2) return "Insufficient folds";
  return significant ? "Difference detected" : "No detected difference";
}
