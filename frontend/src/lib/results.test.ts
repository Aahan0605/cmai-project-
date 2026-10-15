import { describe, expect, it } from "vitest";
import { csvCell, comparisonLabel } from "./results";
describe("downloaded scientific results", () => {
  it("escapes commas, quotes, and line breaks without losing precision", () => {
    expect(csvCell('a,"b"\nc')).toBe('"a,""b""\nc"');
    expect(csvCell(0.123456)).toBe('"0.123456"');
    expect(csvCell(null)).toBe("");
  });
  it("prevents optimizer names from becoming spreadsheet formulas", () => {
    expect(csvCell("=SUM(A1)")).toBe('"\'=SUM(A1)"');
    expect(csvCell(-0.12)).toBe('"-0.12"');
  });
  it("does not call a single fold a statistical result", () => {
    expect(comparisonLabel(false, 1)).toBe("Insufficient folds");
    expect(comparisonLabel(false, 8)).toBe("No detected difference");
  });
});
