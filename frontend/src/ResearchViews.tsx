import { useState } from "react";
import { useData } from "./lib/useData";
import { Feedback } from "./Feedback";
import { LineChart } from "./LineChart";
import { downloadCsv } from "./lib/results";

type Trial = {
  index: number;
  hparams: Record<string, unknown>;
  epochs_run: number;
  val_auc: number | null;
  val_loss: number | null;
  diverged: boolean;
  selected: boolean;
};
type TuningData = {
  optimizers: Record<string, { method: string; trials: Trial[] }>;
};
const fmt = (n: number | null | undefined) => (n == null ? "—" : n.toFixed(4));
export function Tuning({ id }: { id: number }) {
  const q = useData<TuningData>(`experiments/${id}/tuning`);
  const [name, setName] = useState("");
  const names = Object.keys(q.data?.optimizers ?? {});
  const selected = names.includes(name) ? name : names[0];
  const optimizer = q.data?.optimizers[selected];
  const trials = optimizer?.trials ?? [];
  return (
    <>
      <div className="section-head">
        <div>
          <span className="eyebrow">EQUAL-BUDGET SEARCH</span>
          <h2>How hyperparameters were selected</h2>
        </div>
        {!!names.length && (
          <label className="filter-label">
            Optimizer
            <select value={selected} onChange={(e) => setName(e.target.value)}>
              {names.map((n) => (
                <option key={n}>{n}</option>
              ))}
            </select>
          </label>
        )}
      </div>
      <p className="muted">
        Trials select checkpoints and hyperparameters using val_tune video AUC.
        Test results play no part in selection. Failed trials count toward the
        budget.
      </p>
      <Feedback loading={q.isLoading} error={q.error} />
      {!!trials.length && (
        <>
          <div className="results-tools">
            <span>
              {optimizer?.method} search · {trials.length} recorded trials ·{" "}
              {trials.filter((t) => t.diverged).length} diverged
            </span>
            <button
              className="action"
              onClick={() =>
                downloadCsv(`tuning-${id}-${selected}.csv`, [
                  [
                    "trial",
                    "selected",
                    "diverged",
                    "epochs",
                    "val_auc",
                    "val_loss",
                    "hyperparameters",
                  ],
                  ...trials.map((t) => [
                    t.index,
                    String(t.selected),
                    String(t.diverged),
                    t.epochs_run,
                    t.val_auc,
                    t.val_loss,
                    JSON.stringify(t.hparams),
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
                  <th>Trial</th>
                  <th>Status</th>
                  <th>Validation AUC</th>
                  <th>Validation loss</th>
                  <th>Epochs</th>
                  <th>Hyperparameters</th>
                </tr>
              </thead>
              <tbody>
                {trials.map((t) => (
                  <tr
                    key={t.index}
                    className={t.selected ? "selected-run" : ""}
                  >
                    <td>{t.index}</td>
                    <td>
                      {t.diverged
                        ? "Diverged"
                        : t.selected
                          ? "Selected"
                          : "Evaluated"}
                    </td>
                    <td>{fmt(t.val_auc)}</td>
                    <td>{fmt(t.val_loss)}</td>
                    <td>{t.epochs_run}</td>
                    <td>
                      <code className="parameter-code">
                        {JSON.stringify(t.hparams)}
                      </code>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      {!q.isLoading && !q.error && !trials.length && (
        <p className="empty">No tuning trials recorded yet.</p>
      )}
    </>
  );
}

type ToyRun = {
  id: number;
  benchmark: string;
  optimizer: string;
  seed: number;
  steps_to_tol: number | null;
  final_f: number | null;
  final_grad_norm: number | null;
  tolerance: number;
};
type ToyData = {
  tuning_seed: number;
  benchmarks: {
    name: string;
    dimension: number;
    description: string | null;
    minimizer: number[] | null;
  }[];
  runs: ToyRun[];
  curves: Record<string, Record<string, [number, number | null][]>>;
  trajectories: Record<string, Record<string, number[][]>>;
};
function Trajectory({
  points,
  minimum,
}: {
  points: number[][];
  minimum: number[] | null;
}) {
  const finite = points.filter(
    (p) => p.length === 2 && p.every(Number.isFinite),
  );
  if (!finite.length)
    return (
      <p className="muted">
        No saved 2D trajectory available. Curves and numerical results remain
        available.
      </p>
    );
  const bounds = minimum?.length === 2 ? [...finite, minimum] : finite;
  const xs = bounds.map((p) => p[0]),
    ys = bounds.map((p) => p[1]);
  const x0 = Math.min(...xs),
    y0 = Math.min(...ys);
  const dx = Math.max(...xs) - x0 || 1,
    dy = Math.max(...ys) - y0 || 1;
  const pos = (p: number[]) => [
    35 + (530 * (p[0] - x0)) / dx,
    275 - (240 * (p[1] - y0)) / dy,
  ];
  const start = pos(finite[0]),
    end = pos(finite[finite.length - 1]);
  const target = minimum?.length === 2 ? pos(minimum) : null;
  return (
    <>
      <svg
        className="chart trajectory"
        viewBox="0 0 600 310"
        role="img"
        aria-label="Parameter trajectory, circle start, square end, cross known minimum"
      >
        <line x1="35" y1="275" x2="565" y2="275" stroke="var(--border)" />
        <line x1="35" y1="35" x2="35" y2="275" stroke="var(--border)" />
        <polyline
          points={finite.map((p) => pos(p).join(",")).join(" ")}
          stroke="var(--accent)"
          fill="none"
          strokeWidth="2"
        />
        <circle cx={start[0]} cy={start[1]} r="5" fill="var(--ok)" />
        <rect
          x={end[0] - 4}
          y={end[1] - 4}
          width="8"
          height="8"
          fill="var(--accent)"
        />
        {target && (
          <text
            x={target[0]}
            y={target[1] + 5}
            textAnchor="middle"
            fill="var(--danger)"
          >
            ×
          </text>
        )}
        <text x="35" y="302">
          x: {x0.toPrecision(3)} → {(x0 + dx).toPrecision(3)}
        </text>
        <text x="285" y="302">
          y: {y0.toPrecision(3)} → {(y0 + dy).toPrecision(3)}
        </text>
      </svg>
      <p className="muted">
        ● Start · ■ End · × Known minimum. Saved paths are sampled; axes have
        independent scales.
      </p>
    </>
  );
}
export function ToyBenchmarks() {
  const q = useData<ToyData>("toy");
  const [benchmark, setBenchmark] = useState("");
  const [optimizer, setOptimizer] = useState("");
  const benches = q.data?.benchmarks ?? [];
  const chosen = benches.find((b) => b.name === benchmark) ?? benches[0];
  const curves = q.data?.curves[chosen?.name ?? ""] ?? {};
  const names = Object.keys(curves);
  const selected = names.includes(optimizer) ? optimizer : names[0];
  const runs = (q.data?.runs ?? []).filter((r) => r.benchmark === chosen?.name);
  const selectedRuns = runs.filter((r) => r.optimizer === selected);
  return (
    <>
      <div className="section-head">
        <div>
          <span className="eyebrow">KNOWN MINIMA</span>
          <h2>Optimizer mathematics, made visible</h2>
        </div>
        {!!benches.length && (
          <label className="filter-label">
            Benchmark
            <select
              value={chosen?.name}
              onChange={(e) => setBenchmark(e.target.value)}
            >
              {benches.map((b) => (
                <option key={b.name}>{b.name}</option>
              ))}
            </select>
          </label>
        )}
      </div>
      <Feedback loading={q.isLoading} error={q.error} />
      {chosen && (
        <p className="muted">
          {chosen.description ?? chosen.name} · {chosen.dimension} dimensions.
          Curves and trajectories show seed {q.data?.tuning_seed}; the table
          shows every recorded seed.
        </p>
      )}
      {!!names.length && (
        <>
          <label className="filter-label">
            Optimizer
            <select
              value={selected}
              onChange={(e) => setOptimizer(e.target.value)}
            >
              {names.map((n) => (
                <option key={n}>{n}</option>
              ))}
            </select>
          </label>
          <h3>Objective by step</h3>
          <LineChart
            series={[
              {
                name: selected,
                x: curves[selected].map((p) => p[0]),
                y: curves[selected].map((p) => p[1]),
              },
            ]}
            label={`${selected} objective by step`}
            log
          />
          <p className="muted">
            Log scale; objective values below 1e-16 are displayed at 1e-16.
            Missing values break the curve.
          </p>
          {chosen?.dimension === 2 && (
            <>
              <h3>Path through parameter space</h3>
              <Trajectory
                points={q.data?.trajectories[chosen.name]?.[selected] ?? []}
                minimum={chosen.minimizer}
              />
            </>
          )}
        </>
      )}
      {!!runs.length && (
        <>
          <div className="results-tools">
            <span>
              {runs.length} runs recorded
              {selectedRuns.length
                ? ` · ${selectedRuns.filter((r) => r.steps_to_tol != null).length}/${selectedRuns.length} ${selected} seeds reached tolerance`
                : ""}
            </span>
            <button
              className="action"
              onClick={() =>
                downloadCsv(`toy-${chosen?.name}.csv`, [
                  [
                    "optimizer",
                    "seed",
                    "steps_to_tolerance",
                    "final_objective",
                    "final_grad_norm",
                    "tolerance",
                  ],
                  ...runs.map((r) => [
                    r.optimizer,
                    r.seed,
                    r.steps_to_tol,
                    r.final_f,
                    r.final_grad_norm,
                    r.tolerance,
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
                  <th>Seed</th>
                  <th>Steps to tolerance</th>
                  <th>Final objective</th>
                  <th>Gradient norm</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.id}>
                    <td>{r.optimizer}</td>
                    <td>{r.seed}</td>
                    <td>{r.steps_to_tol ?? "Not reached"}</td>
                    <td>{r.final_f?.toExponential(3) ?? "Non-finite"}</td>
                    <td>
                      {r.final_grad_norm?.toExponential(3) ?? "Non-finite"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      {!q.isLoading && !q.error && !runs.length && (
        <p className="empty">
          No toy runs recorded. Run the toy suite and refresh the results.
        </p>
      )}
    </>
  );
}
