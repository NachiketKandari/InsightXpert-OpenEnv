"""WebSocket client for the BIRD Text-to-SQL OpenEnv environment."""

from __future__ import annotations

from typing import Any

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from .models import BirdSQLAction, BirdSQLObservation, BirdSQLState


class BirdText2SQLEnv(EnvClient[BirdSQLAction, BirdSQLObservation, BirdSQLState]):
    """Client for connecting to the BIRD Text-to-SQL environment server."""

    def _step_payload(self, action: BirdSQLAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[BirdSQLObservation]:
        obs = BirdSQLObservation(**payload.get("observation", payload))
        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done,
        )

    def _parse_state(self, payload: dict[str, Any]) -> BirdSQLState:
        return BirdSQLState(**payload)
