"""Schema linking tools for building perfect schema context from gold SQL queries.

This package is a standalone extraction from the InsightXpert project
(https://github.com/NachiketKandari/InsightXpert). It profiles database columns
and uses gold SQL to identify which tables/columns are relevant to each question,
producing pruned schema text that gives models the ideal context.

This is a TOOL used to generate data/schema_linking.json — it is NOT part of the
runtime environment.
"""
