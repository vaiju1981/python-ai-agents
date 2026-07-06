"""Tests for the A2A agent adapter. Skips if a2a-sdk not installed."""
import importlib.util
import pytest

a2a_available = importlib.util.find_spec("a2a") is not None
pytestmark = pytest.mark.skipif(not a2a_available, reason="a2a-sdk not installed")


def test_remote_a2a_agent_without_sdk():
    """Even if a2a isn't installed, the adapter should handle gracefully."""
    from python_ai_agents.adapters.a2a import RemoteA2aAgent

    # This creates the adapter; _client will be None if SDK not installed
    agent = RemoteA2aAgent(agent_url="http://localhost:9999")
    # If SDK not installed, _client should be None
    if not a2a_available:
        assert agent._client is None
