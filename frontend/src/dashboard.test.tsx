// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, useLocation } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { App, Results } from "./App";
import { MediaDemo } from "./MediaDemo";
import { LineChart } from "./LineChart";
import { RunExplorer } from "./RunExplorer";
import { Generalization } from "./Generalization";
import { fetchJson } from "./lib/api";
import { hasInterval, positiveId } from "./lib/selection";

let responses: Record<string, unknown>;
let request: ReturnType<typeof vi.fn>;
const experiment = {
  id: 7,
  name: "running_test",
  description: "A new experiment",
  n_runs: 1,
  n_completed: 0,
  n_diverged: 0,
  superseded: false,
  cv_base: null,
};
function Location() {
  return <output data-testid="location">{useLocation().search}</output>;
}
function mount(node: React.ReactNode, route = "/") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: Infinity } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter
        initialEntries={[route]}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        {node}
        <Location />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}
beforeEach(() => {
  document.documentElement.dataset.theme = "light";
  responses = {
    meta: { generated_at: "2026-10-15" },
    experiments: [],
    toy: {
      tuning_seed: 0,
      benchmarks: [],
      curves: {},
      trajectories: {},
      runs: [],
    },
  };
  request = vi.fn(async (url: string) => {
    const key = url.replace(/^.*data\//, "").replace(/\.json$/, "");
    return new Response(JSON.stringify(responses[key] ?? {}), {
      headers: { "content-type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", request);
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("dashboard workflows", () => {
  it("keeps the toy and protocol pages usable before any detector runs exist", async () => {
    mount(<App />);
    expect(
      await screen.findByText(/No training experiments recorded/),
    ).toBeTruthy();
    await userEvent.click(
      screen.getByRole("button", { name: "Toy benchmarks" }),
    );
    expect(await screen.findByText(/No toy runs recorded/)).toBeTruthy();
    expect(screen.getByTestId("location").textContent).toContain(
      "view=Toy+benchmarks",
    );
    await userEvent.click(screen.getByRole("button", { name: "Protocol" }));
    expect(screen.getByText("Calibrate, then fuse")).toBeTruthy();
  });
  it("selects experiments even when no run has completed", async () => {
    responses.experiments = [experiment];
    responses["experiments/7/leaderboard"] = { tables: [] };
    mount(<App />);
    expect(await screen.findByText("A new experiment")).toBeTruthy();
    expect(screen.queryByText(/No training experiments recorded/)).toBeNull();
  });
  it("does not inspect a run belonging to another experiment", async () => {
    responses["experiments/7/runs"] = [
      {
        id: 8,
        optimizer: "adam",
        seed: 0,
        status: "running",
        best_val_auc: null,
        wall_clock_sec: null,
      },
    ];
    mount(<RunExplorer id={7} />, "/?run=999");
    expect(
      await screen.findByText(
        "This run is not part of the selected experiment.",
      ),
    ).toBeTruthy();
    expect(
      request.mock.calls.some(([url]) => String(url).includes("runs/999")),
    ).toBe(false);
  });
  it("uses paired seed means independently for each comparison dataset", async () => {
    responses["experiments/7/generalisation"] = {
      rows: [
        {
          optimizer: "adam",
          in_dataset: "hq",
          dataset: "celeb",
          n: 2,
          in_mean: 0.8,
          cross_mean: 0.6,
          gap_mean: 0.2,
        },
        {
          optimizer: "adam",
          in_dataset: "hq",
          dataset: "other",
          n: 1,
          in_mean: 0.9,
          cross_mean: 0.4,
          gap_mean: 0.5,
        },
      ],
    };
    mount(<Generalization id={7} />);
    expect(await screen.findByText("0.600")).toBeTruthy();
    await userEvent.selectOptions(
      screen.getByLabelText("Comparison dataset"),
      "other",
    );
    expect(screen.getByText("0.900")).toBeTruthy();
    expect(screen.getByText("0.400")).toBeTruthy();
    expect(screen.queryByText("0.600")).toBeNull();
  });
  it("does not draw a confidence interval for a single replicate", () => {
    const view = render(
      <Results
        data={{
          tables: [
            {
              dataset: "hq",
              protocol: "in_distribution",
              granularity: "video",
              pairwise: [],
              rows: [
                {
                  optimizer: "adam",
                  auc_mean: 0.7,
                  auc_std: null,
                  auc_ci95: [null, null],
                  n_seeds: 1,
                },
              ],
            },
          ],
        }}
      />,
    );
    expect(screen.getByText("Insufficient replicates")).toBeTruthy();
    expect(view.container.querySelector(".range")).toBeNull();
  });
  it("breaks charts at missing observations without generating NaN paths", () => {
    const view = render(
      <LineChart
        label="curve"
        series={[{ name: "adam", x: [0, 1, 2], y: [0.5, null, 0.7] }]}
      />,
    );
    const path = view.container.querySelector("path")?.getAttribute("d");
    expect(path).not.toContain("NaN");
    expect(path?.match(/M/g)).toHaveLength(2);
  });
  it("reports HTML fallbacks as a data-source configuration error", async () => {
    request.mockResolvedValueOnce(
      new Response("<html>app</html>", {
        headers: { "content-type": "text/html" },
      }),
    );
    await expect(fetchJson("missing")).rejects.toThrow("Expected JSON");
  });
});

describe("URL and numerical boundaries", () => {
  it("accepts only positive safe integer IDs", () => {
    for (const invalid of [
      null,
      "-1",
      "1.5",
      "1e3",
      "0",
      "Infinity",
      "9007199254740992",
    ])
      expect(positiveId(invalid)).toBeNull();
    expect(positiveId("42")).toBe(42);
  });
  it("requires two finite interval endpoints", () => {
    expect(hasInterval([null, null])).toBe(false);
    expect(hasInterval([0.5, Infinity])).toBe(false);
    expect(hasInterval([0.5, 0.8])).toBe(true);
  });
});

it("uploads media only after explicit connection and analysis", async () => {
  URL.createObjectURL = vi.fn(() => "blob:preview");
  URL.revokeObjectURL = vi.fn();
  request.mockImplementation(
    async (url: string) =>
      new Response(
        JSON.stringify(
          url.endsWith("health")
            ? {
                input_modalities: ["visual"],
                source_run: { id: 1 },
                max_bytes: 26214400,
              }
            : {
                prediction: "real",
                score: 0.2,
                frames_analyzed: 1,
                streams: { visual: 0.2 },
                notice: "Research",
                source_run: { id: 1 },
              },
        ),
        { headers: { "content-type": "application/json" } },
      ),
  );
  mount(<MediaDemo />);
  expect(request).not.toHaveBeenCalled();
  await userEvent.upload(
    screen.getByLabelText(/Image or video/),
    new File(["image"], "test.png", { type: "image/png" }),
  );
  expect(request).not.toHaveBeenCalled();
  await userEvent.click(
    screen.getByRole("button", { name: "Connect local service" }),
  );
  await screen.findByText(/Ready · run 1/);
  expect(request).toHaveBeenCalledTimes(1);
  await userEvent.click(screen.getByRole("button", { name: "Analyze media" }));
  expect(await screen.findByText("Model prediction: real")).toBeTruthy();
  expect(request.mock.calls[1][1]).toMatchObject({ method: "POST" });
});
