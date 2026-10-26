"""Adapter completion-boundary tests."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from uni_agent.framework import entry as entry_module
from uni_agent.framework.base import AgentFramework


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_custom_framework_requires_explicit_admission_support():
    generated = []

    class CustomFramework(AgentFramework):
        @classmethod
        def from_config(cls, **_kwargs):
            return cls()

        async def generate_sequences(self, prompts):
            generated.append(prompts)

    worker_class = entry_module.AgentFrameworkWorker.__ray_metadata__.modified_class
    worker = worker_class.__new__(worker_class)
    worker.framework = CustomFramework()

    with pytest.raises(NotImplementedError, match="does not support admission-only submission"):
        await worker.submit_sessions("admission")
    assert generated == []

    await worker.generate_sequences("generation")
    assert generated == ["generation"]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "method, rpc, waits",
    [
        ("generate_sequences", "generate_sequences", False),
        ("generate_sequences_and_wait", "generate_sequences", True),
        ("submit_sessions", "submit_sessions", True),
    ],
)
def test_adapter_completion_boundaries(monkeypatch, method, rpc, waits):
    worker = SimpleNamespace(generate_sequences=Mock(), submit_sessions=Mock())
    join = Mock()
    monkeypatch.setattr(entry_module.ray, "get", join)
    adapter = entry_module.AgentFrameworkRolloutAdapter()
    adapter.framework_worker = worker
    prompts = object()

    getattr(adapter, method)(prompts)

    selected = getattr(worker, rpc)
    unused = getattr(worker, "submit_sessions" if rpc == "generate_sequences" else "generate_sequences")
    selected.remote.assert_called_once_with(prompts)
    unused.remote.assert_not_called()
    if waits:
        join.assert_called_once_with(selected.remote.return_value)
    else:
        join.assert_not_called()
