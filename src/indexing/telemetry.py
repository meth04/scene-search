"""Fixed, low-cardinality metrics. Query text, URIs and actor IDs are never labels."""
from __future__ import annotations

import os
import time
from contextlib import contextmanager

from opentelemetry import trace
from prometheus_client import Counter, Histogram

STAGES = Histogram("driveretriever_stage_seconds", "Processing duration", ["stage"],
                   buckets=(.01, .05, .1, .5, 1, 3, 5, 10, 30, 60, 300))
WINDOWS = Counter("driveretriever_windows_committed", "Checkpointed source windows")
FAILURES = Counter("driveretriever_failures", "Processing errors", ["stage"])


def configure_tracing():
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    provider = TracerProvider(resource=Resource.create({"service.name": "driveretriever"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)


@contextmanager
def stage(name):
    started = time.monotonic()
    with trace.get_tracer("driveretriever").start_as_current_span(name):
        try:
            yield
        except Exception:
            FAILURES.labels(name).inc()
            raise
        finally:
            STAGES.labels(name).observe(time.monotonic()-started)
