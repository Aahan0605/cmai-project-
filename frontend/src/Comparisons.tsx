import { useState } from "react";
import { comparisonLabel } from "./lib/results";
export type Comparison = {
  a: string;
  b: string;
  mean_diff: number;
  ci95: (number | null)[];
  p_holm: number | null;
  distinguishable: boolean;
  n?: number;
};
const fmt = (n: number | null) => (n == null ? "—" : n.toFixed(4));
export function Comparisons({
  rows,
  defaultReference = "",
}: {
  rows: Comparison[];
  defaultReference?: string;
}) {
  const [reference, setReference] = useState(defaultReference);
  const names = [...new Set(rows.flatMap((r) => [r.a, r.b]))].sort();
  const selected = names.includes(reference) ? reference : names[0];
  const matches = rows.filter((r) => r.a === selected || r.b === selected);
  if (!rows.length)
    return (
      <p className="muted">
        Statistical comparisons need at least two optimizers with repeated
        evaluations.
      </p>
    );
  return (
    <details className="comparisons">
      <summary>
        Are the differences meaningful?{" "}
        <span>{rows.length} comparisons · Holm correction</span>
      </summary>
      <p className="muted">
        A detected difference requires both a confidence interval excluding zero
        and an adjusted p-value below 0.05. No detected difference does not
        establish equivalence.
      </p>
      <label className="filter-label">
        Reference optimizer
        <select value={selected} onChange={(e) => setReference(e.target.value)}>
          {names.map((n) => (
            <option key={n}>{n}</option>
          ))}
        </select>
      </label>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Compared with</th>
              <th>Reference − other AUC</th>
              <th>95% interval</th>
              <th>Adjusted p</th>
              <th>Evidence</th>
            </tr>
          </thead>
          <tbody>
            {matches.map((r) => {
              const sign = r.a === selected ? 1 : -1;
              const ci =
                sign === 1
                  ? r.ci95
                  : [
                      r.ci95[1] == null ? null : -r.ci95[1],
                      r.ci95[0] == null ? null : -r.ci95[0],
                    ];
              return (
                <tr key={`${r.a}/${r.b}`}>
                  <td>{sign === 1 ? r.b : r.a}</td>
                  <td>{fmt(sign * r.mean_diff)}</td>
                  <td>{ci.map(fmt).join(" to ")}</td>
                  <td>{fmt(r.p_holm)}</td>
                  <td>
                    <span
                      className={
                        r.distinguishable ? "evidence detected" : "evidence"
                      }
                    >
                      {comparisonLabel(r.distinguishable, r.n)}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="muted">
        {rows[0].n != null
          ? "Paired comparisons of seed-averaged AUC by fold. Shared training data across folds limits independence."
          : "Welch comparisons across seeds. These intervals describe seed variability, not uncertainty across people."}
      </p>
    </details>
  );
}
