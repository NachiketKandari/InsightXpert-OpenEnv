"""Collect per-column statistics from a SQLite database.

Extracted from InsightXpert (src/insightxpert/profiler/stats_collector.py).
"""
from __future__ import annotations

import logging

from schema_linking.db import Database
from schema_linking.models import (
    ColumnProfile,
    ColumnStats,
    ColumnSchema,
    DatabaseProfile,
    DatabaseSchema,
    TableProfile,
    TableSchema,
)

logger = logging.getLogger(__name__)


class StatsCollector:
    """Runs SQL queries against a database to gather per-column statistics."""

    def collect(self, db: Database, schema: DatabaseSchema) -> DatabaseProfile:
        logger.debug("Collecting stats for '%s' (%d tables)", schema.db_id, len(schema.tables))
        tables = [self._collect_table(db, t) for t in schema.tables]
        return DatabaseProfile(db_id=schema.db_id, tables=tables)

    def _collect_table(self, db: Database, table: TableSchema) -> TableProfile:
        row_count = db.execute(f'SELECT COUNT(*) FROM "{table.name}"')[0][0]
        columns = [self._collect_column(db, table.name, col) for col in table.columns]
        logger.debug("  table '%s': %d rows, %d columns", table.name, row_count, len(columns))
        return TableProfile(name=table.name, row_count=row_count, columns=columns)

    def _collect_column(self, db: Database, table: str, col: ColumnSchema) -> ColumnProfile:
        name = col.name
        try:
            count = db.execute(f'SELECT COUNT("{name}") FROM "{table}"')[0][0]
            null_count = db.execute(f'SELECT COUNT(*) FROM "{table}" WHERE "{name}" IS NULL')[0][0]
            distinct_count = db.execute(f'SELECT COUNT(DISTINCT "{name}") FROM "{table}"')[0][0]

            min_val, max_val = db.execute(
                f'SELECT CAST(MIN("{name}") AS TEXT), CAST(MAX("{name}") AS TEXT) FROM "{table}"'
            )[0]

            sample_values = [
                r[0] for r in db.execute(
                    f'SELECT DISTINCT CAST("{name}" AS TEXT) FROM "{table}"'
                    f' WHERE "{name}" IS NOT NULL ORDER BY "{name}" LIMIT 20'
                )
            ]
        except Exception as exc:
            logger.warning("Stats collection failed for %s.%s: %s", table, name, exc)
            return ColumnProfile(
                name=name, type=col.type,
                stats=ColumnStats(count=0, null_count=0, distinct_count=0),
            )

        return ColumnProfile(
            name=name,
            type=col.type,
            stats=ColumnStats(
                count=count,
                null_count=null_count,
                distinct_count=distinct_count,
                min_value=min_val,
                max_value=max_val,
                sample_values=sample_values,
            ),
        )
