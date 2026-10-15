export type Series = { name: string; x: number[]; y: (number | null)[] };
const palette = [
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
export const seriesColor = (name: string) =>
  palette[[...name].reduce((s, c) => s + c.charCodeAt(0), 0) % palette.length];
export function LineChart({
  series,
  label,
  log = false,
}: {
  series: Series[];
  label: string;
  log?: boolean;
}) {
  const transform = (n: number) => (log ? Math.log10(Math.max(n, 1e-16)) : n);
  const points = series.flatMap((s) =>
    s.y.flatMap((y, i) =>
      y != null && Number.isFinite(y) ? [[s.x[i], transform(y)]] : [],
    ),
  );
  if (!points.length)
    return <p className="empty">No finite curve values recorded yet.</p>;
  const xmin = Math.min(...points.map((p) => p[0]));
  const xmax = Math.max(...points.map((p) => p[0]));
  let ymin = Math.min(...points.map((p) => p[1]));
  let ymax = Math.max(...points.map((p) => p[1]));
  if (ymin === ymax) {
    ymin -= 0.05;
    ymax += 0.05;
  }
  const xy = (x: number, y: number) =>
    `${65 + (810 * (x - xmin)) / (xmax - xmin || 1)},${280 - (240 * (transform(y) - ymin)) / (ymax - ymin)}`;
  return (
    <>
      <svg
        className="chart"
        viewBox="0 0 900 325"
        role="img"
        aria-label={label}
      >
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line
              x1="65"
              x2="875"
              y1={280 - 240 * t}
              y2={280 - 240 * t}
              stroke="var(--border)"
            />
            <text x="0" y={285 - 240 * t}>
              {log
                ? `10^${(ymin + (ymax - ymin) * t).toFixed(1)}`
                : (ymin + (ymax - ymin) * t).toFixed(3)}
            </text>
          </g>
        ))}
        {series.map((s) => {
          // Missing values break the line; they are never drawn as zero.
          let path = "",
            open = false;
          s.y.forEach((y, i) => {
            if (y == null || !Number.isFinite(y)) {
              open = false;
              return;
            }
            path += `${open ? "L" : "M"}${xy(s.x[i], y)} `;
            open = true;
          });
          return (
            <g key={s.name}>
              <path
                d={path}
                fill="none"
                stroke={seriesColor(s.name)}
                strokeWidth="2.5"
              >
                <title>{s.name}</title>
              </path>
              {s.y.length === 1 &&
                s.y[0] != null &&
                Number.isFinite(s.y[0]) && (
                  <circle
                    cx={65}
                    cy={Number(xy(s.x[0], s.y[0]).split(",")[1])}
                    r="4"
                    fill={seriesColor(s.name)}
                  />
                )}
            </g>
          );
        })}
        <text x="65" y="315">
          Step {xmin}
        </text>
        <text x="790" y="315">
          Step {xmax}
        </text>
      </svg>
      <div className="legend">
        {series.map((s) => (
          <span key={s.name}>
            <i style={{ background: seriesColor(s.name) }} />
            {s.name}
          </span>
        ))}
      </div>
    </>
  );
}
