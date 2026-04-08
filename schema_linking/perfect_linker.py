"""Build perfect schema linking from gold SQL queries.

Parses each gold SQL with sqlglot to extract the exact tables, columns, and
literals used, then adds FK join paths and renders a pruned schema -- giving
the generator the ideal schema context for each question.

Extracted from InsightXpert (src/insightxpert/linker/perfect_linker.py and
src/insightxpert/linker/linking_utils.py), adapted for standalone use.
"""
from __future__ import annotations

import logging

import sqlglot
from sqlglot import exp

from schema_linking.models import (
    ColumnStats,
    DatabaseProfile,
    DatabaseSchema,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gold SQL parsing
# ---------------------------------------------------------------------------

def _parse_gold_sql(sql: str) -> tuple[set[str], set[tuple[str, str]], set[str]]:
    """Parse a gold SQL query and extract tables, columns, and literals.

    Returns (tables, columns_as_(table,col), literals).
    Columns may have table="" if unqualified in the SQL.
    """
    tables: set[str] = set()
    columns: set[tuple[str, str]] = set()
    literals: set[str] = set()

    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.SqlglotError:
        logger.warning("sqlglot failed to parse gold SQL: %.120s", sql)
        return tables, columns, literals

    # Build alias map: alias_name -> real table name
    alias_map: dict[str, str] = {}
    for table_node in parsed.find_all(exp.Table):
        alias = table_node.alias
        if alias:
            alias_map[alias] = table_node.name

    # Extract tables
    tables = {t.name for t in parsed.find_all(exp.Table) if t.name}

    # Extract columns, resolving aliases
    for col in parsed.find_all(exp.Column):
        col_name = col.name
        if not col_name:
            continue
        table_ref = col.table or ""
        real_table = alias_map.get(table_ref, table_ref)
        columns.add((real_table, col_name))

    # Extract string literals
    literals = {
        lit.this for lit in parsed.find_all(exp.Literal) if lit.is_string and lit.this
    }

    return tables, columns, literals


def _resolve_unqualified_columns(
    columns: set[tuple[str, str]],
    schema: DatabaseSchema,
) -> set[tuple[str, str]]:
    """Resolve ("", col_name) entries against the schema (high-recall)."""
    col_to_tables: dict[str, list[str]] = {}
    for t in schema.tables:
        for c in t.columns:
            col_to_tables.setdefault(c.name, []).append(t.name)

    resolved: set[tuple[str, str]] = set()
    for table_ref, col_name in columns:
        if table_ref:
            resolved.add((table_ref, col_name))
        else:
            for tname in col_to_tables.get(col_name, []):
                resolved.add((tname, col_name))
    return resolved


# ---------------------------------------------------------------------------
# Join path expansion
# ---------------------------------------------------------------------------

def _add_join_paths(
    tables: set[str],
    columns: set[tuple[str, str]],
    schema: DatabaseSchema,
) -> tuple[set[str], set[tuple[str, str]]]:
    """For every linked table, include its PK and FK columns connecting to other linked tables."""
    for table in schema.tables:
        if table.name not in tables:
            continue
        # Always include PKs
        for col in table.columns:
            if col.primary_key:
                columns.add((table.name, col.name))
        # Include FK columns that connect to another linked table
        for fk in table.foreign_keys:
            if fk.ref_table in tables:
                columns.add((table.name, fk.column))
    return tables, columns


# ---------------------------------------------------------------------------
# Schema text rendering
# ---------------------------------------------------------------------------

def _joinable_columns_section(
    table_columns: list[tuple[str, set[str]]],
) -> str:
    """Return a text block listing shared column names between table pairs."""
    lines: list[str] = []
    for i, (t1, cols1) in enumerate(table_columns):
        for t2, cols2 in table_columns[i + 1:]:
            shared = sorted(cols1 & cols2)
            if shared:
                lines.append(f"  {t1} <-> {t2}: {', '.join(shared)}")
    if not lines:
        return ""
    return "\nJoinable Columns (shared column names between tables):\n" + "\n".join(lines)


def _render_pruned_schema(
    tables: set[str],
    columns: set[tuple[str, str]],
    schema: DatabaseSchema,
    profile: DatabaseProfile,
) -> str:
    """Render only the linked tables/columns with short_summary and sample values."""
    summaries: dict[str, dict[str, str]] = {}
    profile_stats: dict[str, dict[str, ColumnStats]] = {}
    for tp in profile.tables:
        summaries[tp.name] = {cp.name: cp.short_summary for cp in tp.columns}
        profile_stats[tp.name] = {cp.name: cp.stats for cp in tp.columns}

    lines: list[str] = []
    for table in sorted(schema.tables, key=lambda t: t.name):
        if table.name not in tables:
            continue

        fk_map = {fk.column: (fk.ref_table, fk.ref_column) for fk in table.foreign_keys}
        pk_set = {col.name for col in table.columns if col.primary_key}

        col_lines: list[str] = []
        included_fks: list[tuple[str, str, str]] = []
        for col in table.columns:
            if (table.name, col.name) not in columns:
                continue

            tags: list[str] = []
            if col.name in pk_set:
                tags.append("PK")
            if col.name in fk_map:
                ref_t, ref_c = fk_map[col.name]
                tags.append(f"FK -> {ref_t}.{ref_c}")
                included_fks.append((col.name, ref_t, ref_c))
            tag_str = f", {', '.join(tags)}" if tags else ""

            desc_parts: list[str] = []
            s = summaries.get(table.name, {}).get(col.name, "")
            if s:
                desc_parts.append(s)
            stats = profile_stats.get(table.name, {}).get(col.name)
            if stats and stats.sample_values and stats.distinct_count <= 20:
                vals = ", ".join(repr(v) for v in stats.sample_values)
                desc_parts.append(f"Values: [{vals}]")
            desc_str = f": {' | '.join(desc_parts)}" if desc_parts else ""

            col_lines.append(f'    - "{col.name}" ({col.type}{tag_str}){desc_str}')

        if not col_lines:
            continue

        lines.append(f'Table: "{table.name}"')
        lines.append("  Columns:")
        lines.extend(col_lines)

        if included_fks:
            lines.append("  Foreign Keys:")
            for col_name, ref_t, ref_c in included_fks:
                lines.append(f"    - {table.name}.{col_name} -> {ref_t}.{ref_c}")

        lines.append("")

    # Append joinable columns between linked table pairs
    table_col_sets = [
        (table.name, {col.name for col in table.columns})
        for table in sorted(schema.tables, key=lambda t: t.name)
        if table.name in tables
    ]
    joinable = _joinable_columns_section(table_col_sets)
    if joinable:
        lines.append(joinable)

    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_perfect_schema(
    gold_sql: str,
    schema: DatabaseSchema,
    profile: DatabaseProfile,
) -> str:
    """Build perfect pruned schema text for a single question's gold SQL.

    Returns the rendered schema_text string ready for prompt injection.
    """
    tables, columns, literals = _parse_gold_sql(gold_sql)

    # Resolve unqualified columns against the schema
    columns = _resolve_unqualified_columns(columns, schema)

    # Ensure referenced tables from column qualifiers are included
    for table_ref, _ in columns:
        if table_ref:
            tables.add(table_ref)

    # Add PK/FK join path columns for connectivity
    tables, columns = _add_join_paths(tables, columns, schema)

    # Render the pruned schema
    schema_text = _render_pruned_schema(tables, columns, schema, profile)

    if not schema_text.strip():
        logger.warning(
            "Perfect linking produced empty schema for sql=%.80s; "
            "gold SQL may reference tables/columns not in schema",
            gold_sql,
        )

    return schema_text
