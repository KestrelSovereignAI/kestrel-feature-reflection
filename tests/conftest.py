"""One native SQLite/privacy fixture for Reflection component tests."""
import pytest
from kestrel_sovereign.storage.memory_system import MemorySystem

from kestrel_feature_reflection.feature import ReflectionFeature

pytest_plugins = ["kestrel_sovereign.testing.fixtures"]


@pytest.fixture
async def reflection_writer(native_tool_effect_agent):
    agent = await native_tool_effect_agent(did="did:test:reflection-privacy")
    agent.memory_system = MemorySystem(
        agent._raw_storage, agent.did, privacy_storage=agent.storage,
        transition_lock=agent._get_privacy_transition_lock(),
    )
    feature = ReflectionFeature(agent)
    await feature.initialize()
    yield feature
    await feature.on_disable()
