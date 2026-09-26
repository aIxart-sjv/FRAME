"""FastAPI application factory for the FRAME API (Phase 7).

`create_app()` wires together the router (frame.api.routes), CORS (from
frame.api.config, development-friendly and explicit rather than "*"), and a
single generic exception handler. That handler is the ONE place any
exception -- an `frame.api.errors.APIError`, any `frame.*` package error,
or a genuine internal failure -- is turned into an HTTP response: it never
leaks a Python traceback to the caller, and 5xx failures are logged
server-side for debugging instead.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from frame.analysis.errors import AnalysisError
from frame.consistency.errors import ConsistencyError
from frame.geospatial.errors import GeospatialError
from frame.models.errors import ModelError
from frame.preprocessing.errors import PreprocessingError
from frame.tiling import TilingConfig
from frame.tiling.errors import InvalidSceneError
from frame.uncertainty.errors import UncertaintyError

from frame.api import config
from frame.api.errors import APIError, model_error_code, status_code_for

logger = logging.getLogger("frame.api")


def _describe_validation_errors(exc: RequestValidationError) -> str:
    """One line per distinct problem, e.g. ``1 validation error: body.model: ...``.

    Built from ``exc.errors()`` rather than ``str(exc)``: the latter embeds the
    handler's source-file path and line number, and repeats an error when
    FastAPI validates the same body for more than one dependency (as happens
    on POST /sr/run, where the route and model selection both need the body).
    """
    seen = set()
    problems = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        message = error.get("msg", "invalid value")
        if (location, message) in seen:
            continue
        seen.add((location, message))
        problems.append(f"{location}: {message}" if location else message)
    plural = "s" if len(problems) != 1 else ""
    return f"{len(problems)} validation error{plural}: " + "; ".join(problems)


def create_app() -> FastAPI:
    TilingConfig(overlap=config.TILE_OVERLAP)  # fail at startup, not on the first request, if FRAME_API_TILE_OVERLAP is invalid
    app = FastAPI(title="FRAME API", version=config.API_VERSION)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from frame.api.routes import router

    app.include_router(router)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "request_validation_error",
                "code": "request_validation_error",
                "detail": _describe_validation_errors(exc),
            },
        )

    def _known_error_response(request: Request, exc: Exception) -> JSONResponse:
        # Every case reachable here is a known 4xx (an APIError subclass, or
        # one of the frame.* package's own input-validation error families
        # -- see frame.api.errors.status_code_for). Genuine 5xx failures are
        # NOT registered here; they fall through to Starlette's default
        # server-error handling below, which never echoes a traceback to
        # the caller.
        status_code = status_code_for(exc)
        code = exc.code if isinstance(exc, APIError) else "frame_error"
        return JSONResponse(status_code=status_code, content={"error": code, "code": code, "detail": str(exc)})

    for exc_type in (APIError, PreprocessingError, GeospatialError, ConsistencyError, UncertaintyError, AnalysisError, InvalidSceneError):
        app.add_exception_handler(exc_type, _known_error_response)

    # frame.models errors. `str(exc)` on these is written to be user-safe
    # (it never names the low-level runtime), and is what a caller sees for
    # "unavailable" and "your input violates the model contract". Runtime
    # failures (worker crash, CUDA OOM, weight mismatch) get a generic
    # message instead; their `technical_detail` goes to the server log only.
    def _model_error_response(request: Request, exc: ModelError) -> JSONResponse:
        code = model_error_code(exc)
        status_code = status_code_for(exc)
        if code == "model_runtime_error":
            logger.error("Model runtime error (%s): %s | %s", type(exc).__name__, exc, exc.technical_detail)
            detail = "The selected model failed while processing this request. Try again, or choose another model."
        else:
            if exc.technical_detail:
                logger.warning("Model error (%s): %s | %s", type(exc).__name__, exc, exc.technical_detail)
            detail = str(exc)
        return JSONResponse(status_code=status_code, content={"error": code, "code": code, "detail": detail})

    app.add_exception_handler(ModelError, _model_error_response)

    # Anything else is a genuine internal failure. It still gets the documented JSON body (never Starlette's plain-text
    # "Internal Server Error", which a client cannot tell from a proxy error), with no exception text, and the real
    # exception goes to the server log only.
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled %s on %s %s", type(exc).__name__, request.method, request.url.path, exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "code": "internal_error",
                "detail": "An internal error occurred while processing this request. The details were logged on the server.",
            },
        )

    app.add_exception_handler(Exception, handle_unexpected_error)

    return app


app = create_app()
