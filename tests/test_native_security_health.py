"""A genuine permission database failure is failed health, not a truthy envelope."""
import aiosqlite
import pytest
from kestrel_sovereign.features.security.feature import SecurityFeature
from kestrel_sovereign.features.security.permissions import PermissionStore

from kestrel_feature_reflection.models import CheckStatus


@pytest.mark.asyncio
async def test_failed_real_permission_listing_cannot_report_healthy(reflection_writer, tmp_path):
    feature = reflection_writer
    path = tmp_path / "owned-security-health.db"
    store = PermissionStore(str(path))
    await store.initialize()
    security = SecurityFeature(feature.agent)
    security.permission_store = store
    feature.agent.features["SecurityFeature"] = security
    async with aiosqlite.connect(path) as db:
        await db.execute("DROP TABLE security_permissions")
        await db.commit()
    envelope = await security.list_permissions()
    assert envelope.error, "Probe did not trigger a real SQLite permission failure"
    check = await feature._arms_checker.check_security_permissions()
    assert check.status is CheckStatus.FAIL, "Permission database failure became a passing health check"
    assert not check.details or "permission_count" not in check.details


@pytest.mark.asyncio
async def test_actual_disabled_web_search_cannot_report_healthy(reflection_writer, monkeypatch):
    from kestrel_sovereign.features.web_search import tool as tool_module
    from kestrel_sovereign.features.web_search.feature import WebSearchFeature
    feature = reflection_writer
    # Explicitly unconfigured real provider selection, with no ambient account.
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(tool_module, "discover_entry_point_classes", lambda *_args, **_kwargs: {})
    web = WebSearchFeature(feature.agent)
    await web.initialize()
    assert not web.tool.enabled
    feature.agent.features["WebSearchFeature"] = web
    envelope = await web.search("owned disabled-provider probe")
    assert envelope.error, "Probe did not exercise Core's genuine disabled search"
    check = await feature._arms_checker.check_web_search()
    assert check.status is CheckStatus.FAIL, "Disabled web search became a passing health check"
