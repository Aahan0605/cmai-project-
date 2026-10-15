import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { DATA_MODE, fetchJson } from "./lib/api";
import { positiveId } from "./lib/selection";
import { downloadCsv } from "./lib/results";
type Run = {
  id: number;
  optimizer: string;
  seed: number;
  status: string;
  best_val_auc: number | null;
  wall_clock_sec: number | null;
};
type Detail = {
  id: number;
  optimizer: string;
  seed: number;
  status: string;
  best_epoch: number | null;
  experiment: string;
  model: string;
  hyperparams: Record<string, unknown>;
  evaluations: {
    dataset: string;
    split: string;
    granularity: string;
    auc: number | null;
    eer: number | null;
    accuracy: number | null;
    n_samples: number;
  }[];
  library_versions: Record<string, string> | null;
};
const fmt = (n: number | null) => (n == null ? "—" : n.toFixed(3));
export function RunExplorer({ id }: { id: number }) {
  const [params, setParams] = useSearchParams();
  const selected = positiveId(params.get("run"));
  const choose = (run: number | null) => {
    const next = new URLSearchParams(params);
    if (run) next.set("run", String(run));
    else next.delete("run");
    setParams(next);
  };
  const q = useQuery({
    queryKey: [`experiments/${id}/runs`],
    queryFn: () => fetchJson<Run[]>(`experiments/${id}/runs`),
    refetchInterval: DATA_MODE === "api" ? 15000 : false,
  });
  const belongs = q.data?.some((r) => r.id === selected) ?? false;
  const detail = useQuery({
    queryKey: [`runs/${selected}`],
    queryFn: () => fetchJson<Detail>(`runs/${selected}`),
    enabled: selected !== null && belongs,
    refetchInterval: DATA_MODE === "api" ? 15000 : false,
  });
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("all");
  const rows = (q.data ?? []).filter(
    (r) =>
      (status === "all" || r.status === status) &&
      `${r.id} ${r.optimizer} ${r.seed}`
        .toLowerCase()
        .includes(search.toLowerCase().trim()),
  );
  return (
    <>
      <div className="section-head">
        <div>
          <span className="eyebrow">EXPERIMENT LEDGER</span>
          <h2>Every run, traceable</h2>
        </div>
        <button
          className="action"
          disabled={!rows.length}
          onClick={() =>
            downloadCsv(`runs-${id}.csv`, [
              ["run_id", "optimizer", "seed", "status", "val_auc", "seconds"],
              ...rows.map((r) => [
                r.id,
                r.optimizer,
                r.seed,
                r.status,
                r.best_val_auc,
                r.wall_clock_sec,
              ]),
            ])
          }
        >
          Download filtered CSV ↓
        </button>
      </div>
      <div className="run-filters">
        <label className="filter-label">
          Find a run
          <input
            type="search"
            placeholder="Optimizer, run ID, or seed"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </label>
        <label className="filter-label">
          Status
          <select value={status} onChange={(e) => setStatus(e.target.value)}>
            {[
              "all",
              "completed",
              "running",
              "queued",
              "failed",
              "diverged",
            ].map((s) => (
              <option key={s} value={s}>
                {s === "all" ? "All statuses" : s}
              </option>
            ))}
          </select>
        </label>
        <span className="muted">
          {rows.length} of {q.data?.length ?? 0} runs
        </span>
      </div>
      {q.isLoading && <p role="status">Loading runs…</p>}
      {q.error && (
        <p role="alert">
          Unable to load runs.{" "}
          <button onClick={() => q.refetch()}>Retry</button>
        </p>
      )}
      {selected && !q.isLoading && !belongs && (
        <p role="alert">This run is not part of the selected experiment.</p>
      )}
      {selected && belongs && (
        <section
          className="run-inspector"
          aria-label={`Run ${selected} details`}
        >
          <div className="section-head">
            <h3>Run #{selected}</h3>
            <button className="action" onClick={() => choose(null)}>
              Close ×
            </button>
          </div>
          {detail.isLoading && <p role="status">Loading run details…</p>}
          {detail.error && (
            <p role="alert">
              Unable to load this run.{" "}
              <button onClick={() => detail.refetch()}>Retry</button>
            </p>
          )}
          {detail.data && (
            <>
              <div className="run-facts">
                <div>
                  <small>OPTIMIZER</small>
                  <strong>{detail.data.optimizer}</strong>
                </div>
                <div>
                  <small>STATUS</small>
                  <strong>{detail.data.status}</strong>
                </div>
                <div>
                  <small>SEED</small>
                  <strong>{detail.data.seed}</strong>
                </div>
                <div>
                  <small>BEST EPOCH</small>
                  <strong>{detail.data.best_epoch ?? "—"}</strong>
                </div>
              </div>
              <p className="muted">
                {detail.data.experiment} · {detail.data.model}
              </p>
              <h3>Hyperparameters</h3>
              <dl className="parameters">
                {Object.entries(detail.data.hyperparams).map(([k, v]) => (
                  <div key={k}>
                    <dt>{k}</dt>
                    <dd>
                      {typeof v === "object" ? JSON.stringify(v) : String(v)}
                    </dd>
                  </div>
                ))}
              </dl>
              <h3>Video evaluations</h3>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Dataset</th>
                      <th>Split</th>
                      <th>AUC</th>
                      <th>EER</th>
                      <th>Accuracy</th>
                      <th>Videos</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detail.data.evaluations
                      .filter((e) => e.granularity === "video")
                      .map((e) => (
                        <tr key={`${e.dataset}/${e.split}`}>
                          <td>{e.dataset}</td>
                          <td>{e.split}</td>
                          <td>{fmt(e.auc)}</td>
                          <td>{fmt(e.eer)}</td>
                          <td>{fmt(e.accuracy)}</td>
                          <td>{e.n_samples}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
              {!detail.data.evaluations.length && (
                <p className="muted">
                  No evaluations recorded for this run yet.
                </p>
              )}
              <details>
                <summary>Raw record and library versions</summary>
                <pre>{JSON.stringify(detail.data, null, 2)}</pre>
              </details>
            </>
          )}
        </section>
      )}
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Optimizer</th>
              <th>Seed</th>
              <th>Status</th>
              <th>Validation AUC</th>
              <th>Time (s)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.id}
                className={selected === r.id ? "selected-run" : ""}
              >
                <td>
                  <button
                    aria-label={`Inspect run ${r.id}`}
                    onClick={() => choose(r.id)}
                  >
                    #{r.id} ↗
                  </button>
                </td>
                <td>{r.optimizer}</td>
                <td>{r.seed}</td>
                <td>
                  <span className={`run-status status-${r.status}`}>
                    {r.status}
                  </span>
                </td>
                <td>{fmt(r.best_val_auc)}</td>
                <td>{fmt(r.wall_clock_sec)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!q.isLoading && !q.error && !rows.length && (
        <p className="empty">
          No runs match these filters. Try another optimizer or status.
        </p>
      )}
    </>
  );
}
