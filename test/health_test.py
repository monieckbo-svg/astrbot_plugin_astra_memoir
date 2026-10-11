"""Raw ingestion health must distinguish silence from failed ingestion."""
import tempfile
from pathlib import Path

import retriever_test  # installs lightweight AstrBot/sqlite-vec test doubles
from astrbot_plugin_astra_memoir.pipeline.health import (
    HealthState, get_raw_ingestion_health,
)


def main():
    now = 2_000_000

    # Last message was fully stored 10 minutes ago; subsequent silence is idle.
    assert get_raw_ingestion_health(now, now - 600, now - 600, 0) == "idle"

    # A fresh hook gets a short write grace period.
    assert get_raw_ingestion_health(now, now - 5, now - 600, 1) == "pending"

    # A hook two minutes ahead of raw is a real failure signal.
    assert get_raw_ingestion_health(now, now - 120, now - 600, 1) == "error"

    # Hook and raw timestamps that match are healthy.
    assert get_raw_ingestion_health(now, now - 5, now - 4, 0) == "healthy"
    # A same-second raw from an earlier message cannot hide an unmatched hook.
    assert get_raw_ingestion_health(now, now - 5, now - 4, 1) == "pending"

    # No hook is idle, not unknown/error.
    assert get_raw_ingestion_health(now, 0, 0, 0) == "idle"

    with tempfile.TemporaryDirectory() as tmp:
        health = HealthState(Path(tmp) / "health.json")
        health.data.update(
            last_private_hook_at=now - 5,
            last_private_raw_at=now - 4,
            private_pending_hook_count=0,
            # Group has never been active; it must not poison overall health.
            last_group_hook_at=0,
            last_group_raw_at=0,
            group_pending_hook_count=0,
            last_user_hook_at=now - 5,
            last_raw_ingest_at=now - 4,
            raw_error=None,
            last_llm_activity_at=0,
        )
        result = health.watchdog(now)
        assert result["private_ingestion_status"] == "healthy"
        assert result["group_ingestion_status"] == "idle"
        assert result["ingestion_status"] == "healthy"
        assert result["watchdog_warning"] == ""

        health.data.update(
            last_private_hook_at=now - 5,
            last_private_raw_at=now - 600,
            private_pending_hook_count=1,
            last_user_hook_at=now - 5,
            last_raw_ingest_at=now - 600,
        )
        assert health.watchdog(now)["ingestion_status"] == "pending"

        health.data.update(last_private_hook_at=now - 120,
                           last_user_hook_at=now - 120)
        result = health.watchdog(now)
        assert result["ingestion_status"] == "error"
        assert "未及时入库" in result["watchdog_warning"]

    print("Raw ingestion health tests passed")


if __name__ == "__main__": main()
