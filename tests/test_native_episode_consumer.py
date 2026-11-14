"""The actual Core consolidator stores/loads writer input without generation."""
from uuid import uuid4

import pytest
from kestrel_sovereign.storage.memory_models import MemoryEpisode


@pytest.mark.asyncio
async def test_original_core_episodes_reach_reflection(reflection_writer):
    feature = reflection_writer
    await feature.agent.memory_system.initialize()
    feature.agent.memory_consolidator = feature.agent.memory_system.consolidator
    feature._init_analyzer()
    episode = MemoryEpisode(id=uuid4().hex, agent_id=feature.agent.did,
                            title="Owned episode input", summary="Owned stored memory excerpt")
    await feature.agent.memory_consolidator._save_episode(episode)
    expected = await feature.agent.memory_consolidator.get_episodes(limit=10)
    assert [e.id for e in expected] == [episode.id]
    loaded = await feature.analyzer._load_episodes("all", 10)
    assert [e["id"] for e in loaded] == [episode.id], "Original Core episodes never reached Reflection"
    prompt = feature.analyzer._build_prompt([], loaded, "normal")
    assert episode.summary in prompt
    assert "<user_input>" in prompt and "</user_input>" in prompt
