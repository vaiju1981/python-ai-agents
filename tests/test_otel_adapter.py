"""Tests for the OpenTelemetry observer adapter. Skips if opentelemetry not installed."""
import importlib.util
import pytest

otel_available = importlib.util.find_spec("opentelemetry") is not None
pytestmark = pytest.mark.skipif(not otel_available, reason="opentelemetry not installed")


def test_otel_observer_creates_without_error():
    from python_ai_agents.adapters.otel import OtelAgentObserver

    observer = OtelAgentObserver(service_name="test")
    assert observer.service_name == "test"
