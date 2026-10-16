import { useState } from "react";
import { useData } from "./lib/useData";
import { Feedback } from "./Feedback";
import { Comparisons, type Comparison } from "./Comparisons";
import { downloadCsv } from "./lib/results";
type Score = {
  optimizer: string;
  auc_mean: number;
  auc_ci95: (number | null)[];
  n_folds?: number;
  n_seeds?: number;
  model?: string;
  dataset?: string;
  seconds?: number;
};
type HybridData = {
  design_commit: string;
  catalogue: {
    name: string;
    parent: string | null;
    ablation_note: string | null;
  }[];
  coverage: { n: number; status: string }[];
  cross_validation: {
    tables: { dataset: string; rows: Score[]; pairwise: Comparison[] }[];
  } | null;
  studies: { experiment: string; rows: Score[] }[];
  interpretation: string;
};
const fmt = (v: number | null | undefined) => (v == null ? "—" : v.toFixed(4));
function Scores({ rows }: { rows: Score[] }) {
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Optimizer</th>
            <th>Model</th>
            <th>Dataset</th>
            <th>Video AUC</th>
            <th>95% interval</th>
            <th>Replicates</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr
              key={`${r.optimizer}/${r.model ?? "cv"}/${r.dataset ?? "selected"}`}
            >
              <td>{r.optimizer}</td>
              <td>{r.model ?? "Calibrated multimodal heads"}</td>
              <td>{r.dataset ?? "Selected above"}</td>
              <td>{fmt(r.auc_mean)}</td>
              <td>{r.auc_ci95.map(fmt).join(" to ")}</td>
              <td>
                {r.n_folds != null
                  ? `${r.n_folds} folds`
                  : `${r.n_seeds} seeds`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
export function HybridView() {
  const q = useData<HybridData>("hybrid");
  const [dataset, setDataset] = useState("");
  const tables = q.data?.cross_validation?.tables ?? [];
  const table = tables.find((t) => t.dataset === dataset) ?? tables[0];
  return (
    <>
      <span className="eyebrow">PHASES 3 & 4</span>
      <h2>The hybrid, tested component by component</h2>
      <p className="muted">
        Adam-style preconditioning + RMS-matched momentum direction + Lookahead
        averaging. This is an exploratory combination of established techniques,
        with one gradient pass per step.
      </p>
      <div className="protocol-grid">
        <article>
          <h3>Adaptive direction</h3>
          <p>
            Use bias-corrected first and second moments to normalize
            coordinates.
          </p>
        </article>
        <article>
          <h3>Momentum blend</h3>
          <p>Match the directions’ group norms before a fixed 50/50 mixture.</p>
        </article>
        <article>
          <h3>Slow weights</h3>
          <p>Every five steps, interpolate halfway toward the fast weights.</p>
        </article>
      </div>
      <Feedback loading={q.isLoading} error={q.error} />
      {q.data && (
        <>
          <div className="notice">
            Design frozen in commit {q.data.design_commit}.{" "}
            {q.data.interpretation}
          </div>
          <h3>Ablation definitions</h3>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Variant</th>
                  <th>Removed component</th>
                </tr>
              </thead>
              <tbody>
                {q.data.catalogue.map((c) => (
                  <tr key={c.name}>
                    <td>{c.name}</td>
                    <td>{c.ablation_note ?? "Full hybrid"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <h3>Equal-budget grouped study</h3>
          <p className="muted">
            Eight folds × seven optimizer variants × five seeds = 280 target
            runs. Twenty tuning trials per variant, then the same final epoch
            budget. Counts can be partial while a sweep is running.
          </p>
          {table ? (
            <>
              <div className="results-tools">
                <label className="filter-label">
                  Test dataset
                  <select
                    value={table.dataset}
                    onChange={(e) => setDataset(e.target.value)}
                  >
                    {tables.map((t) => (
                      <option key={t.dataset}>{t.dataset}</option>
                    ))}
                  </select>
                </label>
                <button
                  className="action"
                  onClick={() =>
                    downloadCsv("hybrid-cv.csv", [
                      [
                        "dataset",
                        "optimizer",
                        "auc",
                        "folds",
                        "ci_low",
                        "ci_high",
                      ],
                      ...table.rows.map((r) => [
                        table.dataset,
                        r.optimizer,
                        r.auc_mean,
                        r.n_folds,
                        ...r.auc_ci95,
                      ]),
                    ])
                  }
                >
                  Download CSV ↓
                </button>
              </div>
              <Scores rows={table.rows} />
              <Comparisons
                rows={table.pairwise}
                defaultReference="hybrid_blend"
              />
            </>
          ) : (
            <p className="empty">No hybrid CV evaluations recorded yet.</p>
          )}
          {q.data.studies.map((study) => (
            <section className="study-block" key={study.experiment}>
              <h3>{study.experiment}</h3>
              <Scores rows={study.rows} />
            </section>
          ))}
          <p className="muted">
            Raw CLIP fine-tuning uses a matched frozen control, one sampled
            frame per video and a fixed common learning rate. It is a limited
            ablation, separate from the main 20-trial study. Image-only accuracy
            still requires a labelled image dataset.
          </p>
        </>
      )}
    </>
  );
}
