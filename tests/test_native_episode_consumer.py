"""The actual Core consolidator stores/loads writer input without generation."""
from uuid import uuid4

import pytest
from kestrel_sovereign.storage.memory_models import MemoryEpisode
from kestrel_sovereign.privacy import PrivacyMode


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(PrivacyMode))
async def test_original_core_episodes_reach_reflection(reflection_writer, mode):
    feature = reflection_writer
    original_analyzer = feature.analyzer
    assert feature._mind_checker.analyzer is original_analyzer
    await feature.agent.memory_system.initialize()
    feature.agent.memory_consolidator = feature.agent.memory_system.consolidator
    assert feature.analyzer is original_analyzer
    assert feature._mind_checker.analyzer is original_analyzer
    episode = MemoryEpisode(id=uuid4().hex, agent_id=feature.agent.did,
                            title="Owned episode input", summary="Owned stored memory excerpt")
    await feature.agent.memory_consolidator._save_episode(episode)
    expected = await feature.agent.memory_consolidator.get_episodes(limit=10)
    assert [e.id for e in expected] == [episode.id]
    async with feature.agent._get_privacy_transition_lock():
        feature.agent.storage.set_privacy_mode(mode)
        feature.agent.privacy_agent.set_mode(mode)
    loaded = await feature.analyzer._load_episodes("all", 10)
    if feature.agent.privacy_agent.privacy_config.is_ephemeral():
        assert loaded == [], "Ephemeral reflection retrieved durable memory episodes"
        return
    assert [e["id"] for e in loaded] == [episode.id], "Original Core episodes never reached Reflection"
    prompt = feature.analyzer._build_prompt([], loaded, "normal")
    assert episode.summary in prompt
    assert "<user_input>" in prompt and "</user_input>" in prompt
