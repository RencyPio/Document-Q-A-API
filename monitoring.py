"""Monitoring hooks: one JSON log line per HTTP request (latency), plus an "llm_request" line per /ask (tokens)."""
import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("docqa.metrics")


def log_event(**fields) -> None:
    logger.info(json.dumps(fields, default=str))


class LatencyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Latency-Ms"] = str(latency_ms)
        log_event(
            event="request", request_id=request.state.request_id, method=request.method,
            path=request.url.path, status=response.status_code, latency_ms=latency_ms,
        )
        return response
