import { useState } from "react";
import { useData } from "./lib/useData";
import { downloadCsv } from "./lib/results";
type Gap = {
  optimizer: string;
  in_dataset: string;
  dataset: string;
  n: number;
  in_mean: number;
  cross_mean: number;
  gap_mean: number;
};
export function Generalization({ id }: { id: number }) {
  const q = useData<{ rows: Gap[] }>(`experiments/${id}/generalisation`);
  const [dataset, setDataset] = useState("");
  const external = [...new Set(q.data?.rows.map((r) => r.dataset) ?? [])];
  const selected = external.includes(dataset) ? dataset : external[0];
  const rows = (q.data?.rows ?? [])
    .filter((r) => r.dataset === selected)
    .map((r) => ({
      name: r.optimizer,
      ind: r.in_mean,
      cross: r.cross_mean,
      gap: r.gap_mean,
      n: r.n,
      inDataset: r.in_dataset,
    }));
  return (
    <>
      <div className="section-head">
        <div>
          <span className="eyebrow">DISTRIBUTION SHIFT</span>
          <h2>Does performance travel?</h2>
        </div>
        {selected && (
          <label className="filter-label">
            Comparison dataset
            <select
              value={selected}
              onChange={(e) => setDataset(e.target.value)}
            >
              {external.map((t) => (
                <option key={t}>{t}</option>
              ))}
            </select>
          </label>
        )}
      </div>
      <p className="muted">
        AUC gap = in-distribution − comparison AUC. A negative gap means the
        comparison set was easier; it does not prove stronger generalization.
      </p>
      {q.isLoading && <p role="status">Loading evaluation results…</p>}
      {q.error && (
        <p role="alert">
          Could not load results.{" "}
          <button onClick={() => q.refetch()}>Retry</button>
        </p>
      )}
      {!q.isLoading && !q.error && !rows.length && (
        <p className="empty">
          No matched in-distribution and cross-dataset evaluations recorded.
        </p>
      )}
      {selected === "optlab_timit_lq" && (
        <div className="notice">
          TIMIT LQ is a quality-shift test of the same held-out people. External
          FF++ → Celeb-DF evaluation is still pending dataset access.
        </div>
      )}
      {rows.length > 0 && (
        <>
          <div className="section-head">
            <h3>
              {rows[0]?.inDataset} → {selected}
            </h3>
            <button
              className="action"
              onClick={() =>
                downloadCsv(`generalization-${id}.csv`, [
                  [
                    "optimizer",
                    "in_dataset",
                    "comparison_dataset",
                    "in_auc",
                    "comparison_auc",
                    "gap",
                    "paired_seeds",
                  ],
                  ...rows.map((r) => [
                    r.name,
                    r.inDataset,
                    selected,
                    r.ind,
                    r.cross,
                    r.gap,
                    r.n,
                  ]),
                ])
              }
            >
              Download CSV ↓
            </button>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Optimizer</th>
                  <th>In-distribution AUC</th>
                  <th>Comparison AUC</th>
                  <th>AUC gap</th>
                  <th>Paired seeds</th>
                  <th>Change on 0–1 AUC scale</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.name}>
                    <td>{r.name}</td>
                    <td>{r.ind.toFixed(3)}</td>
                    <td>{r.cross.toFixed(3)}</td>
                    <td>
                      {r.gap > 0 ? "+" : ""}
                      {r.gap.toFixed(3)}
                    </td>
                    <td>{r.n}</td>
                    <td>
                      <div
                        className="dumbbell"
                        aria-label={`In-distribution ${r.ind.toFixed(3)}, comparison ${r.cross.toFixed(3)}`}
                      >
                        <span
                          style={{
                            left: `${Math.min(r.ind, r.cross) * 100}%`,
                            width: `${Math.abs(r.gap) * 100}%`,
                          }}
                        />
                        <i style={{ left: `${r.ind * 100}%` }} />
                        <b style={{ left: `${r.cross * 100}%` }} />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted">
            ● Green: in-distribution · ● Blue: comparison dataset. Means use
            only seeds evaluated on both datasets and are descriptive; inspect
            seed and fold comparisons before drawing conclusions.
          </p>
        </>
      )}
    </>
  );
}
