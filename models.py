"""Pydantic models for the BIRD Text-to-SQL OpenEnv environment."""

from __future__ import annotations

from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class BirdSQLAction(Action):
    """Agent submits a SQL query."""

    sql_query: str = Field("", description="The SQL query to execute")


class BirdSQLObservation(Observation):
    """What the agent sees after reset() or step()."""

    # Task info (always present)
    task_id: str = Field(..., description="Task identifier")
    db_id: str = Field(..., description="Database name")
    difficulty: str = Field(..., description="simple|moderate|challenging")
    question: str = Field(..., description="Natural language question")
    evidence: str = Field("", description="External knowledge hint")
    schema_linking: str = Field("", description="Relevant tables and columns for this question")
    sample_rows: str = Field("", description="Sample data rows per table")

    # Step result (populated after step(), empty after reset())
    execution_result: str = Field("", description="Query output or error message")
    execution_success: bool = Field(True, description="Did the SQL execute?")
    row_count: int = Field(0, description="Number of result rows")
    column_names: list[str] = Field(default_factory=list)
    feedback: str = Field("", description="Grader feedback for the agent")


class BirdSQLState(State):
    """Episode state for the BIRD Text-to-SQL environment."""

    task_id: str = Field("", description="Current task identifier")
    db_id: str = Field("", description="Current database name")
    difficulty: str = Field("", description="Current task difficulty")
    max_steps: int = Field(5, description="Maximum steps per episode")
    accumulated_reward: float = Field(0.0, description="Sum of rewards so far")
    last_action: str = Field("", description="Last SQL query submitted")
    done: bool = Field(False, description="Episode finished?")
