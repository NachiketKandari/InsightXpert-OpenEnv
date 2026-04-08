"""Extract table/column/FK metadata from a SQLite database via PRAGMA queries.

Extracted from InsightXpert (src/insightxpert/profiler/schema_extractor.py).
"""
from __future__ import annotations

import logging

from schema_linking.db import Database
from schema_linking.models import (
    ColumnSchema,
    DatabaseSchema,
    ForeignKey,
    TableSchema,
)

logger = logging.getLogger(__name__)


class SchemaExtractor:
    """Extracts table/column/FK metadata from a database via PRAGMA queries."""

    def extract(self, db: Database) -> DatabaseSchema:
        logger.debug("Extracting schema from '%s'", db.db_id)
        tables = self._extract_tables(db)
        total_cols = sum(len(t.columns) for t in tables)
        logger.debug(
            "Schema extracted: %d tables, %d columns, db_id='%s'",
            len(tables), total_cols, db.db_id,
        )
        return DatabaseSchema(db_id=db.db_id, tables=tables)

    def _extract_tables(self, db: Database) -> list[TableSchema]:
        rows = db.execute(
            "SELECT name FROM sqlite_master"
            " WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            " ORDER BY name"
        )
        tables = []
        for (name,) in rows:
            columns = self._extract_columns(db, name)
            foreign_keys = self._extract_foreign_keys(db, name)
            tables.append(TableSchema(name=name, columns=columns, foreign_keys=foreign_keys))
        return tables

    def _extract_columns(self, db: Database, table: str) -> list[ColumnSchema]:
        rows = db.execute(f'PRAGMA table_info("{table}")')
        columns = []
        for _, name, col_type, notnull, default, pk_index in rows:
            columns.append(
                ColumnSchema(
                    name=name,
                    type=col_type or "TEXT",
                    nullable=not bool(notnull),
                    primary_key=pk_index > 0,
                    default=str(default) if default is not None else None,
                )
            )
        return columns

    def _extract_foreign_keys(self, db: Database, table: str) -> list[ForeignKey]:
        rows = db.execute(f'PRAGMA foreign_key_list("{table}")')
        fks = []
        for _, _, ref_table, from_col, to_col, on_update, on_delete, _ in rows:
            if to_col is None:
                to_col = self._resolve_pk(db, ref_table)
                if to_col is None:
                    logger.warning(
                        "FK %s.%s -> %s: cannot resolve ref_column (no PK) -- skipping",
                        table, from_col, ref_table,
                    )
                    continue
            fks.append(
                ForeignKey(
                    column=from_col,
                    ref_table=ref_table,
                    ref_column=to_col,
                    on_delete=on_delete if on_delete != "NO ACTION" else None,
                    on_update=on_update if on_update != "NO ACTION" else None,
                )
            )
        return fks

    def _resolve_pk(self, db: Database, table: str) -> str | None:
        rows = db.execute(f'PRAGMA table_info("{table}")')
        for _, name, _, _, _, pk_index in rows:
            if pk_index > 0:
                return name
        return None
