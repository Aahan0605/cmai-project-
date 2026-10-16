import { useState } from "react";
import { DATA_MODE } from "./lib/api";
import { useSearchParams } from "react-router-dom";
import { useData } from "./lib/useData";
import { Feedback as State } from "./Feedback";
import { hasInterval, initialDarkTheme, positiveId } from "./lib/selection";
import { Tuning, ToyBenchmarks } from "./ResearchViews";
import { LineChart } from "./LineChart";
import { HybridView } from "./HybridView";
import { MediaDemo } from "./MediaDemo";
import { Comparisons, type Comparison } from "./Comparisons";
import { RunExplorer } from "./RunExplorer";
import { Generalization } from "./Generalization";
import { downloadCsv } from "./lib/results";

type Experiment = {
  id: number;
  name: string;
  description: string;
  n_runs: number;
  n_completed: number;
  n_diverged: number;
  superseded: boolean;
  cv_base: string | null;
};
type Row = {
  optimizer: string;
  auc_mean: number;
  auc_std: number | null;
  auc_ci95: (number | null)[];
  n_seeds?: number;
  n_folds?: number;
  eer_mean?: number;
};
type Board = {
  tables: {
    dataset: string;
    protocol: string;
    granularity?: string;
    rows: Row[];
    pairwise: Comparison[];
  }[];
};
type Fusion = {
  runs: {
    run_id: number;
    optimizer: string;
    seed: number;
    kkt_residual: number;
    weights: Record<string, number>;
    calibration: {
      nll_before: number[];
      nll_after: number[];
      scales: number[];
      modalities?: string[];
    } | null;
  }[];
};
type Curves = {
  optimizers: Record<
    string,
    { metrics: Record<string, { epoch: number[]; mean: (number | null)[] }> }
  >;
};
const fmt = (n: number | null | undefined) => (n == null ? "—" : n.toFixed(3));
const colors = [
  "#306b50",
  "#6495bf",
  "#b87838",
  "#8c6baf",
  "#dc886f",
  "#658c83",
  "#b2a24c",
  "#617dc0",
  "#ba729a",
  "#555f6a",
];
const tabs = [
  "Overview",
  "Convergence",
  "Generalization",
  "Fusion",
  "Cross-validation",
  "Runs",
  "Tuning",
  "Toy benchmarks",
  "Hybrid",
  "Media demo",
  "Protocol",
];
export function Results({ data }: { data: Board }) {
  const [dataset, setDataset] = useState("");
  const tables = data.tables.filter(
    (t) => !t.granularity || t.granularity === "video",
  );
  const table = tables.find((t) => t.dataset === dataset) ?? tables[0];
  if (!table)
    return (
      <p className="empty">
        No test evaluations recorded for this experiment yet.
      </p>
    );
  return (
    <>
      <div className="section-head">
        <div>
          <span className="eyebrow">HELD-OUT PERFORMANCE</span>
          <h2>Optimizer comparison</h2>
        </div>
        <label>
          Test dataset{" "}
          <select
            value={table.dataset}
            onChange={(e) => setDataset(e.target.value)}
          >
            {tables.map((t) => (
              <option key={t.dataset}>{t.dataset}</option>
            ))}
          </select>
        </label>
      </div>
      <p className="muted">
        Video AUC · mean and 95% bootstrap interval. Ordering does not establish
        statistical superiority.
      </p>
      <div className="results-tools">
        <span>
          {table.rows.length} optimizers · {table.protocol.replaceAll("_", " ")}
        </span>
        <button
          className="action"
          onClick={() =>
            downloadCsv(`${table.dataset}-results.csv`, [
              [
                "dataset",
                "optimizer",
                "auc_mean",
                "auc_std",
                "ci95_low",
                "ci95_high",
                "replicates",
                "unit",
              ],
              ...table.rows.map((r) => [
                table.dataset,
                r.optimizer,
                r.auc_mean,
                r.auc_std,
                ...r.auc_ci95,
                r.n_folds ?? r.n_seeds,
                r.n_folds != null ? "folds" : "seeds",
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
              <th>Mean AUC</th>
              <th className="interval">95% confidence interval</th>
              <th>Std. dev.</th>
              <th>Replicates</th>
            </tr>
          </thead>
          <tbody>
            {table.rows.map((r, i) => (
              <tr key={r.optimizer}>
                <td>
                  <i style={{ background: colors[i % colors.length] }} />
                  {r.optimizer}
                </td>
                <td className="number">{fmt(r.auc_mean)}</td>
                <td className="interval">
                  {hasInterval(r.auc_ci95) && (
                    <div className="range">
                      <span
                        style={{
                          left: `${r.auc_ci95[0] * 100}%`,
                          width: `${Math.max(0.5, (r.auc_ci95[1] - r.auc_ci95[0]) * 100)}%`,
                        }}
                      />
                      <b style={{ left: `${r.auc_mean * 100}%` }} />
                    </div>
                  )}
                  <small>
                    {hasInterval(r.auc_ci95)
                      ? r.auc_ci95.map(fmt).join(" – ")
                      : "Insufficient replicates"}
                  </small>
                </td>
                <td>{fmt(r.auc_std)}</td>
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
      <Comparisons rows={table.pairwise ?? []} />
    </>
  );
}
function Convergence({ id }: { id: number }) {
  const q = useData<Curves>(`experiments/${id}/convergence`);
  const [metric, setMetric] = useState("val_auc");
  const lines = Object.entries(q.data?.optimizers ?? {}).filter(
    ([, v]) => v.metrics[metric],
  );
  return (
    <>
      <div className="section-head">
        <h2>Learning, epoch by epoch</h2>
        <select
          aria-label="Curve metric"
          value={metric}
          onChange={(e) => setMetric(e.target.value)}
        >
          <option value="val_auc">Validation AUC</option>
          <option value="val_loss">Validation loss</option>
          <option value="train_loss">Training loss</option>
        </select>
      </div>
      <State loading={q.isLoading} error={q.error} />
      {!!lines.length && (
        <>
          <LineChart
            series={lines.map(([name, v]) => ({
              name,
              x: v.metrics[metric].epoch,
              y: v.metrics[metric].mean,
            }))}
            label={`${metric} by epoch, mean across completed seeds`}
          />
          <p className="muted">
            Means across completed seeds; early stopping can change the number
            of seeds at each epoch.
          </p>
        </>
      )}
      {!q.isLoading && !q.error && !lines.length && (
        <p>No curves recorded yet.</p>
      )}
    </>
  );
}
function FusionView({ id }: { id: number }) {
  const q = useData<Fusion>(`experiments/${id}/fusion`);
  return (
    <>
      <h2>How the streams combine</h2>
      <p className="muted">
        Per-stream calibration reduces scale confounding. Weights still depend
        on redundancy and the validation sample; they are not causal importance.
      </p>
      <State loading={q.isLoading} error={q.error} />
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Optimizer / seed</th>
              <th>Stream weights</th>
              <th>Calibration</th>
              <th>KKT residual</th>
            </tr>
          </thead>
          <tbody>
            {q.data?.runs.map((r) => (
              <tr key={r.run_id}>
                <td>
                  {r.optimizer}
                  <small>Seed {r.seed}</small>
                </td>
                <td>
                  <div className="stack">
                    {Object.entries(r.weights).map(([m, w], i) => (
                      <span
                        key={m}
                        title={`${m}: ${fmt(w)}`}
                        style={{ width: `${w * 100}%`, background: colors[i] }}
                      />
                    ))}
                  </div>
                  <small>
                    {Object.entries(r.weights)
                      .map(([m, w]) => `${m} ${fmt(w)}`)
                      .join(" · ")}
                  </small>
                </td>
                <td>
                  {r.calibration ? (
                    <details className="calibration-details">
                      <summary>Validation NLL before → after</summary>
                      {(
                        r.calibration.modalities ??
                        ["visual", "audio", "frequency", "joint"].filter(
                          (m) => m in r.weights,
                        )
                      ).map((m, i) => (
                        <div key={m}>
                          <strong>{m}</strong>
                          <small>
                            {fmt(r.calibration!.nll_before[i])} →{" "}
                            {fmt(r.calibration!.nll_after[i])} · scale{" "}
                            {r.calibration!.scales[i].toPrecision(3)}
                          </small>
                        </div>
                      ))}
                      <small>
                        Fitted on val_tune; this is training loss for the
                        calibrator, not independent test improvement.
                      </small>
                    </details>
                  ) : (
                    "Uncalibrated streams"
                  )}
                </td>
                <td>{r.kkt_residual.toExponential(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {q.data?.runs.length === 0 && <p>No fusion solves recorded yet.</p>}
    </>
  );
}
function CrossValidation({ base }: { base: string | null }) {
  const q = useData<Board>(`cv/${base}`, !!base);
  return (
    <>
      <h2>Beyond four test people</h2>
      <p className="muted">
        Eight rotating folds hold out two swap pairs each: all 32 people are
        tested once. Tuning and fusion use separate identity groups. Seed means
        are compared by fold.
      </p>
      {!base ? (
        <div className="notice">
          No cross-validation results for this experiment. Run{" "}
          <code>optlab cv --experiment calibrated_timit_cv</code> to produce all
          eight folds.
        </div>
      ) : (
        <>
          <State loading={q.isLoading} error={q.error} />
          {q.data && (
            <>
              <div className="notice">
                Available results cover{" "}
                {Math.max(
                  0,
                  ...q.data.tables.flatMap((t) =>
                    t.rows.map((r) => r.n_folds ?? 0),
                  ),
                )}{" "}
                folds. Counts below can differ while the sweep is running; use
                the final report for conclusions.
              </div>
              <Results key={base} data={q.data} />
            </>
          )}
        </>
      )}
    </>
  );
}
function Protocol() {
  return (
    <>
      <h2>A stronger evaluation protocol</h2>
      <div className="protocol-grid">
        <article>
          <span className="step">01</span>
          <h3>Calibrate, then fuse</h3>
          <p>
            Fit a positive temperature for each stream on val_tune. Freeze these
            transforms before fitting fusion on val_fusion. Reuse the stored
            transforms for every test.
          </p>
          <small>Validation-only fitting · persisted parameters</small>
        </article>
        <article>
          <span className="step">02</span>
          <h3>Rotate the held-out people</h3>
          <p>
            Eight folds preserve face-swap pairs and separate training, tuning,
            fusion, and testing identities. Every person contributes one
            held-out evaluation.
          </p>
          <small>32 people · 16 swap pairs · 8 folds</small>
        </article>
        <article>
          <span className="step">03</span>
          <h3>Test unseen manipulations</h3>
          <p>
            Train on FaceForensics++ and test on Celeb-DF after author-granted
            access and feature extraction. This is a planned external test; its
            difficulty must be measured.
          </p>
          <small className="pending">
            Pending dataset access — no results yet
          </small>
        </article>
      </div>
      <div className="notice">
        The TIMIT lower-quality split uses the same held-out people and proved
        easier. It is not evidence of robustness to harder, unseen datasets.
      </div>
    </>
  );
}
export function App() {
  const experiments = useData<Experiment[]>("experiments");
  const meta = useData<{ generated_at: string }>("meta");
  const [params, setParams] = useSearchParams();
  const chosen = positiveId(params.get("experiment"));
  const tab = tabs.includes(params.get("view") ?? "")
    ? params.get("view")!
    : "Overview";
  const setChosen = (id: number) =>
    setParams({ view: tab, experiment: String(id) });
  const setTab = (view: string) =>
    setParams({ view, ...(exp ? { experiment: String(exp.id) } : {}) });
  const exp =
    experiments.data?.find((e) => e.id === chosen) ??
    [...(experiments.data ?? [])]
      .reverse()
      .find((e) => !e.superseded && e.n_completed > 0) ??
    [...(experiments.data ?? [])].reverse().find((e) => e.n_runs > 0);
  const board = useData<Board>(
    `experiments/${exp?.id}/leaderboard`,
    !!exp && tab === "Overview",
  );
  const [dark, setDark] = useState(initialDarkTheme);
  return (
    <div className="shell">
      <aside>
        <a
          className="brand"
          href="?view=Overview"
          onClick={(e) => {
            e.preventDefault();
            setTab("Overview");
          }}
        >
          <span className="brand-icon">⌁</span> OptLab
          <span className="brand-df">/ DF</span>
        </a>
        <div className="workspace-label">RESEARCH WORKSPACE</div>
        <nav aria-label="Main navigation">
          {tabs.map((t, i) => (
            <button
              key={t}
              aria-current={tab === t ? "page" : undefined}
              onClick={() => setTab(t)}
            >
              <span aria-hidden="true">
                {["◫", "⌁", "↗", "◈", "▦", "≡", "⚙", "∿", "⨁", "▷", "◎"][i]}
              </span>
              {t}
            </button>
          ))}
        </nav>
        <div className="sidebar-note">
          <span className="live-dot" /> Evidence first
          <p>
            Equal budgets.
            <br />
            Repeated seeds.
            <br />
            Honest uncertainty.
          </p>
          <small>CMAI · Nirma University</small>
        </div>
      </aside>
      <main>
        <header>
          <span>
            Research / <strong>{tab}</strong>
          </span>
          <div>
            <button
              className="refresh-button"
              onClick={() => {
                location.reload();
              }}
              aria-label="Refresh results"
            >
              ↻ Refresh
            </button>
            <span className="badge">
              {DATA_MODE === "api" ? "Live API" : "Database snapshot"}
            </span>
            <button
              aria-label="Toggle color theme"
              onClick={() => {
                const next = !dark;
                setDark(next);
                document.documentElement.dataset.theme = next
                  ? "dark"
                  : "light";
                try {
                  localStorage.setItem("optlab-theme", next ? "dark" : "light");
                } catch {
                  /* optional persistence */
                }
              }}
            >
              {dark ? "☀" : "◐"}
            </button>
          </div>
        </header>
        <div className="content">
          <div className="heading">
            <div>
              <span className="eyebrow">OPTIMIZATION × DEEPFAKE DETECTION</span>
              <h1>
                {tab === "Overview"
                  ? "Small differences. Better questions."
                  : tab}
              </h1>
              <p className="subtitle">
                A reproducible look at how optimizers learn—and how well they
                generalize.
              </p>
            </div>
            {!["Toy benchmarks", "Protocol", "Hybrid", "Media demo"].includes(
              tab,
            ) && (
              <label className="experiment-label">
                EXPERIMENT
                <select
                  value={exp?.id ?? ""}
                  onChange={(e) => setChosen(Number(e.target.value))}
                >
                  {experiments.data
                    ?.filter((e) => e.n_runs > 0)
                    .map((e) => (
                      <option value={e.id} key={e.id}>
                        {e.name}
                        {e.superseded ? " (superseded)" : ""}
                      </option>
                    ))}
                </select>
              </label>
            )}
          </div>
          <State loading={experiments.isLoading} error={experiments.error} />
          {exp && (
            <>
              {tab === "Overview" && (
                <>
                  <div className="metrics">
                    <article>
                      <span>RECORDED RUNS</span>
                      <strong>{exp.n_runs}</strong>
                      <small>Selected experiment</small>
                    </article>
                    <article>
                      <span>COMPLETED</span>
                      <strong>
                        {exp.n_completed}
                        <em> / {exp.n_runs}</em>
                      </strong>
                      <small>Finished training runs</small>
                    </article>
                    <article>
                      <span>DIVERGED</span>
                      <strong>{exp.n_diverged}</strong>
                      <small>Failures stay in the record</small>
                    </article>
                    <article>
                      <span>EXTERNAL BENCHMARK</span>
                      <strong className="text-stat">Pending access</strong>
                      <small>FF++ → Celeb-DF</small>
                    </article>
                  </div>
                  <section className="panel">
                    <State loading={board.isLoading} error={board.error} />
                    {board.data && <Results key={exp.id} data={board.data} />}
                  </section>
                  <section className="experiment-note">
                    <span className="eyebrow">EXPERIMENT NOTES</span>
                    <p>{exp.description}</p>
                  </section>
                </>
              )}
              {tab === "Convergence" && (
                <section className="panel">
                  <Convergence key={exp.id} id={exp.id} />
                </section>
              )}
              {tab === "Generalization" && (
                <section className="panel">
                  <Generalization key={exp.id} id={exp.id} />
                </section>
              )}
              {tab === "Fusion" && (
                <section className="panel">
                  <FusionView id={exp.id} />
                </section>
              )}
              {tab === "Tuning" && (
                <section className="panel">
                  <Tuning key={exp.id} id={exp.id} />
                </section>
              )}
              {tab === "Runs" && (
                <section className="panel">
                  <RunExplorer key={exp.id} id={exp.id} />
                </section>
              )}
              {tab === "Cross-validation" && (
                <section className="panel">
                  <CrossValidation base={exp.cv_base} />
                </section>
              )}
            </>
          )}
          {tab === "Toy benchmarks" && (
            <section className="panel">
              <ToyBenchmarks />
            </section>
          )}
          {tab === "Hybrid" && (
            <section className="panel">
              <HybridView />
            </section>
          )}
          {tab === "Media demo" && (
            <section className="panel">
              <MediaDemo />
            </section>
          )}
          {tab === "Protocol" && <Protocol />}
          {!experiments.isLoading &&
            !experiments.error &&
            !exp &&
            !["Toy benchmarks", "Protocol", "Hybrid", "Media demo"].includes(
              tab,
            ) && (
              <p>
                No training experiments recorded. Run a sweep and export the
                dashboard snapshot.
              </p>
            )}
          <footer>
            <span>OptLab-DF · Reproducible by design</span>
            <span>{meta.data?.generated_at ?? "No snapshot loaded"}</span>
          </footer>
        </div>
      </main>
    </div>
  );
}
