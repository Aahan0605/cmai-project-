import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
const base = (
  import.meta.env.VITE_DEMO_API_URL ?? "http://127.0.0.1:8001"
).replace(/\/$/, "");
type Health = {
  input_modalities: string[];
  source_run: { id: number; experiment: string };
  max_bytes: number;
};
type Prediction = {
  prediction: string;
  score: number;
  frames_analyzed: number;
  streams: Record<string, number>;
  notice: string;
  source_run: { id: number; experiment: string };
};
export function MediaDemo() {
  const [connect, setConnect] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<Prediction | null>(null);
  const controller = useRef<AbortController>();
  const health = useQuery({
    queryKey: ["demo-health", base],
    enabled: connect,
    retry: false,
    queryFn: async () => {
      const response = await fetch(`${base}/demo/health`);
      if (!response.ok) throw new Error("Local demo is unavailable");
      return (await response.json()) as Health;
    },
  });
  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => () => controller.current?.abort(), []);
  const image = !!file?.type.startsWith("image/");
  const missingAudio = image && health.data?.input_modalities.includes("audio");
  async function analyze() {
    if (!file) return;
    setBusy(true);
    setError("");
    setResult(null);
    const active = new AbortController();
    controller.current = active;
    try {
      const response = await fetch(`${base}/demo/predict`, {
        method: "POST",
        headers: { "x-filename": `input.${file.name.split(".").at(-1)}` },
        body: file,
        signal: active.signal,
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? "Analysis failed");
      setResult(body);
    } catch (e) {
      if (!active.signal.aborted)
        setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <span className="eyebrow">LOCAL RESEARCH DEMO</span>
      <h2>Analyze an image or short video</h2>
      <p className="muted">
        Start the separate local inference service with an exported model.
        Selected media is sent only when you press Analyze, processed in a
        temporary directory, and deleted afterward.
      </p>
      <div className="notice">
        This model was evaluated on TIMIT videos. Standalone image accuracy and
        performance on unseen generators remain unmeasured. A score is not
        verified proof of authenticity.
      </div>
      <div className="results-tools">
        <button
          className="action"
          onClick={() => {
            setConnect(true);
            if (connect) health.refetch();
          }}
        >
          Connect local service
        </button>
        <span>
          {health.isFetching
            ? "Connecting…"
            : health.data
              ? `Ready · run ${health.data.source_run.id} · ${health.data.input_modalities.join(" + ")}`
              : "Service not connected"}
        </span>
      </div>
      {health.error && (
        <p role="alert">
          Cannot connect to {base}. Export a model with{" "}
          <code>optlab export-model</code> and start{" "}
          <code>optlab demo --bundle PATH</code>.
        </p>
      )}
      <label className="filter-label">
        Image or video (up to 25 MiB)
        <input
          type="file"
          accept=".jpg,.jpeg,.png,.webp,.bmp,.mp4,.mov,.avi,.webm,.mkv"
          disabled={busy}
          onChange={(e) => {
            const next = e.target.files?.[0] ?? null;
            setResult(null);
            setError("");
            if (next && next.size > 25 * 1024 * 1024) {
              setError("Choose a file no larger than 25 MiB");
              setFile(null);
              return;
            }
            setFile(next);
          }}
        />
      </label>
      {preview &&
        (image ? (
          <img
            className="media-preview"
            src={preview}
            alt="Selected media preview"
          />
        ) : (
          <video
            className="media-preview"
            src={preview}
            controls
            preload="metadata"
          />
        ))}
      {missingAudio && (
        <p role="alert">
          This model needs audio. Choose a video with audio or restart the
          service with a visual/frequency model for images.
        </p>
      )}
      <div className="results-tools">
        <button
          className="action"
          disabled={!health.data || !file || busy || !!missingAudio}
          onClick={analyze}
        >
          {busy ? "Analyzing…" : "Analyze media"}
        </button>
        <button
          className="action"
          disabled={busy}
          onClick={() => {
            setFile(null);
            setResult(null);
            setError("");
          }}
        >
          Clear
        </button>
      </div>
      {error && <p role="alert">{error}</p>}
      {result && (
        <section className="run-inspector">
          <h3>Model prediction: {result.prediction}</h3>
          <p>
            Manipulation score: <strong>{result.score.toFixed(3)}</strong> ·{" "}
            {result.frames_analyzed} usable face frames
          </p>
          <dl className="parameters">
            {Object.entries(result.streams).map(([name, score]) => (
              <div key={name}>
                <dt>{name}</dt>
                <dd>{score.toFixed(3)}</dd>
              </div>
            ))}
          </dl>
          <p className="muted">{result.notice}</p>
          <small>
            Source: run {result.source_run.id}, {result.source_run.experiment}
          </small>
        </section>
      )}
    </>
  );
}
