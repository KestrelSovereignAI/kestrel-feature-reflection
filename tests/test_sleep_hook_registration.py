"""Reflection registers its sleep hook via the core sleep_hooks list (#1784)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kestrel_feature_reflection.feature import ReflectionFeature


async def test_post_all_features_loaded_appends_to_sleep_hooks():
    agent = MagicMock()
    agent.sleep_hooks = []
    feature = ReflectionFeature(agent=agent)
    # create_reflection_hook resolves the reflection feature via get_feature
    agent.get_feature = MagicMock(
        side_effect=lambda n: feature if n in ("reflection", "ReflectionFeature") else None
    )

    await feature.post_all_features_loaded(agent)

    assert len(agent.sleep_hooks) == 1
    hook = agent.sleep_hooks[0]
    assert hasattr(hook, "on_pre_sleep") and hasattr(hook, "on_post_consolidation")


async def test_on_disable_unregisters_and_reenable_is_idempotent():
    agent = MagicMock()
    agent.sleep_hooks = []
    feature = ReflectionFeature(agent=agent)
    agent.get_feature = MagicMock(
        side_effect=lambda n: feature if n in ("reflection", "ReflectionFeature") else None
    )

    await feature.post_all_features_loaded(agent)
    await feature.post_all_features_loaded(agent)  # re-enable
    assert len(agent.sleep_hooks) == 1             # no duplicate

    await feature.on_disable()
    assert agent.sleep_hooks == []                 # removed on disable


async def test_registration_initializes_list_when_absent():
    """If core hasn't initialized sleep_hooks yet, reflection creates it."""
    agent = MagicMock()
    agent.sleep_hooks = None
    feature = ReflectionFeature(agent=agent)
    agent.get_feature = MagicMock(
        side_effect=lambda n: feature if n in ("reflection", "ReflectionFeature") else None
    )

    await feature.post_all_features_loaded(agent)

    assert isinstance(agent.sleep_hooks, list)
    assert len(agent.sleep_hooks) == 1
