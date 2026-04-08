"""Pydantic models for database schema and column profiles.

Extracted from InsightXpert (src/insightxpert/models/schema.py and
src/insightxpert/models/profile.py).
"""
from __future__ import annotations

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Schema models
# ---------------------------------------------------------------------------

class ColumnSchema(BaseModel):
    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False
    default: str | None = None


class ForeignKey(BaseModel):
    column: str
    ref_table: str
    ref_column: str
    on_delete: str | None = None
    on_update: str | None = None


class TableSchema(BaseModel):
    name: str
    columns: list[ColumnSchema]
    foreign_keys: list[ForeignKey] = []

    @property
    def primary_keys(self) -> list[str]:
        return [c.name for c in self.columns if c.primary_key]


class DatabaseSchema(BaseModel):
    db_id: str
    tables: list[TableSchema]

    @property
    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    def get_table(self, name: str) -> TableSchema | None:
        for t in self.tables:
            if t.name == name:
                return t
        return None


# ---------------------------------------------------------------------------
# Profile models
# ---------------------------------------------------------------------------

class ColumnStats(BaseModel):
    count: int
    null_count: int
    distinct_count: int
    min_value: str | None = None
    max_value: str | None = None
    sample_values: list[str] = []


class ColumnProfile(BaseModel):
    name: str
    type: str
    stats: ColumnStats
    mechanical_description: str = ""
    short_summary: str = ""
    long_summary: str = ""


class TableProfile(BaseModel):
    name: str
    row_count: int
    columns: list[ColumnProfile]


class DatabaseProfile(BaseModel):
    db_id: str
    tables: list[TableProfile]
