"""Read-only results API (TRD 12). `optlab serve` runs it with uvicorn.

Paths mirror the static snapshot: `/api/<path>` here is `<snapshot>/<path>.json` there, and
both return the objects built by `optlab.export.queries`. The DB is opened read-only
(`mode=ro`) per request, so the API cannot modify results.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from optlab.config import load_toy_experiment
from optlab.db.connection import connect, require_current_schema
from optlab.export import queries as q

EXPERIMENT_VIEWS: dict[str, Callable] = {
    "leaderboard": q.leaderboard,
    "convergence": q.convergence,
    "generalisation": q.generalisation,
    "tuning": q.tuning,
    "fusion": q.fusion,
    "runs": q.runs,
}


def create_app(
    db: str | Path, toy_config: str | Path, cors_origin: str = "http://localhost:5173"
) -> FastAPI:
    # Fail at startup with migration guidance rather than serving broken SQL endpoints.
    c = connect(db, readonly=True)
    try:
        require_current_schema(c)
    finally:
        c.close()
    app = FastAPI(
        title="OptLab-DF results API",
        version="1",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=[cors_origin], allow_methods=["GET"], allow_headers=["*"]
    )
    toy_cfg = load_toy_experiment(toy_config)

    def conn() -> Iterator[Any]:
        c = connect(db, readonly=True)
        try:
            yield c
        finally:
            c.close()

    @app.get("/api/health")
    def health(c=Depends(conn)) -> dict:
        return {"status": "ok", "schema_version": q.meta(c)["schema_version"]}

    @app.get("/api/meta")
    def meta(c=Depends(conn)) -> dict:
        return q.clean(q.meta(c))

    @app.get("/api/experiments")
    def experiments(c=Depends(conn)) -> list:
        return q.clean(q.experiments(c))

    @app.get("/api/optimizers")
    def optimizers(c=Depends(conn)) -> list:
        return q.clean(q.optimizers(c))

    @app.get("/api/toy")
    def toy(c=Depends(conn)) -> dict:
        return q.clean(q.toy(c, toy_cfg))

    @app.get("/api/cv/{base}")
    def cv(base: str, c=Depends(conn)) -> dict:
        out = q.cv(c, base)
        if out is None:
            raise HTTPException(404, f"no cross-validation results for {base!r}")
        return q.clean(out)

    @app.get("/api/experiments/{exp_id}/{view}")
    def experiment_view(exp_id: int, view: str, c=Depends(conn)) -> Any:
        if view not in EXPERIMENT_VIEWS:
            raise HTTPException(404, f"unknown view {view!r}; one of {sorted(EXPERIMENT_VIEWS)}")
        if c.execute("SELECT 1 FROM experiments WHERE id = ?", (exp_id,)).fetchone() is None:
            raise HTTPException(404, f"no experiment {exp_id}")
        return q.clean(EXPERIMENT_VIEWS[view](c, exp_id))

    @app.get("/api/runs/{run_id}")
    def run(run_id: int, c=Depends(conn)) -> dict:
        out = q.run_detail(c, run_id)
        if out is None:
            raise HTTPException(404, f"no run {run_id}")
        return q.clean(out)

    return app
