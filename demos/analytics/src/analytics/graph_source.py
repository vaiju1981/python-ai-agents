"""Neo4j graph data source: projects nodes and relationships into DuckDB tables.

Connects to a Neo4j database, discovers node labels and relationship types,
and projects them into DuckDB tables so the existing profiling/semantic/query
pipeline works unchanged. Node labels become tables (properties become columns);
relationships become join relationships in the semantic model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from demos.analytics.src.analytics.data_source import (
    ColumnSchema,
    DataSource,
    Relationship,
    TableSchema,
    sql_quote,
)


class GraphSource(DataSource):
    """Neo4j graph data source projected into DuckDB tables.

    Each node label becomes a DuckDB table (with a ``_id`` column for the
    internal Neo4j ID and properties as columns). Relationships become
    ``Relationship`` entries for the query planner.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "neo4j",
        db_path: str = ":memory:",
        max_nodes_per_label: int = 100_000,
    ) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._conn = duckdb.connect(db_path)
        self._table_names: list[str] = []
        self._relationships: list[Relationship] = []
        self._max_nodes = max_nodes_per_label
        self._project()

    def _project(self) -> None:
        with self._driver.session() as session:
            # Discover node labels
            labels_result = session.run("CALL db.labels() YIELD label RETURN label")
            labels = [r["label"] for r in labels_result]

            for label in labels:
                table_name = _sanitize(label)
                # Get property keys for this label
                props_result = session.run(
                    f"MATCH (n:`{label}`) UNWIND keys(n) AS key RETURN DISTINCT key ORDER BY key"
                )
                props = [r["key"] for r in props_result]

                if not props:
                    continue

                # Export node data to DuckDB
                prop_select = ", ".join(f"n.`{p}` AS `{p}`" for p in props)
                query = (
                    f"MATCH (n:`{label}`) "
                    f"RETURN id(n) AS _id, {prop_select} "
                    f"LIMIT {self._max_nodes}"
                )
                records = session.run(query)
                rows = [dict(r) for r in records]

                if not rows:
                    continue

                # Create DuckDB table from records
                import pandas as pd

                df = pd.DataFrame(rows)
                self._conn.register("_tmp_graph", df)
                self._conn.execute(
                    f"CREATE OR REPLACE TABLE {sql_quote(table_name)} AS SELECT * FROM _tmp_graph"
                )
                self._conn.unregister("_tmp_graph")
                self._table_names.append(table_name)

            # Discover relationships
            rels_result = session.run(
                "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
            )
            rel_types = [r["relationshipType"] for r in rels_result]

            for rel_type in rel_types:
                # Find which labels are connected by this relationship type
                connected = session.run(
                    f"MATCH (a)-[r:`{rel_type}`]->(b) "
                    f"RETURN DISTINCT labels(a)[0] AS from_label, "
                    f"labels(b)[0] AS to_label, "
                    f"keys(r) AS rel_props "
                    f"LIMIT 100"
                )
                for record in connected:
                    from_label = record["from_label"]
                    to_label = record["to_label"]
                    from_table = _sanitize(from_label)
                    to_table = _sanitize(to_label)
                    if from_table in self._table_names and to_table in self._table_names:
                        # Use _id as the join key
                        self._relationships.append(
                            Relationship(
                                from_table=from_table,
                                from_columns=("_id",),
                                to_table=to_table,
                                to_columns=("_id",),
                                cardinality="many_to_one",
                                coverage=1.0,
                            )
                        )

        self._conn.execute("SET enable_external_access = false")

    def tables(self) -> list[TableSchema]:
        result = []
        for name in self._table_names:
            cols = self._table_columns(name)
            row_count = self._conn.execute(f"SELECT COUNT(*) FROM {sql_quote(name)}").fetchone()[0]
            result.append(TableSchema(name=name, rows=row_count, columns=tuple(cols)))
        return result

    def _table_columns(self, table: str) -> list[ColumnSchema]:
        schema = self._conn.execute(f"DESCRIBE {sql_quote(table)}").fetchall()
        return [ColumnSchema(name=row[0], physical_type=row[1]) for row in schema]

    def relationships(self) -> list[Relationship]:
        return list(self._relationships)

    def sample(self, table: str, limit: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            f"SELECT * FROM {sql_quote(table)} LIMIT {max(1, limit)}"
        ).fetchall()
        cols = [d[0] for d in self._conn.description]
        return [dict(zip(cols, row, strict=False)) for row in rows]

    def native_query(self, sql: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(sql).fetchall()
        cols = [d[0] for d in self._conn.description]
        return [dict(zip(cols, row, strict=False)) for row in rows]

    def native_query_with_limit(self, sql: str, max_rows: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(sql).fetchmany(max_rows)
        cols = [d[0] for d in self._conn.description]
        return [dict(zip(cols, row, strict=False)) for row in rows]

    # --- ingestion seam (PR-12) ---------------------------------------------
    # GraphSource projects a Neo4j graph into DuckDB tables; it is a read-only
    # backend. Use CsvSource for incremental ingestion. ``row_count`` is a pure
    # read.
    def row_count(self, table: str) -> int:
        name = _sanitize(table)
        return self._conn.execute(f"SELECT COUNT(*) FROM {sql_quote(name)}").fetchone()[0]

    def append_rows(self, table: str, rows: list[dict[str, Any]]) -> int:
        raise NotImplementedError("GraphSource is read-only; use CsvSource for ingestion (PR-12)")

    def ingest_csv(self, table: str, csv_path: Path, *, mode: str = "append") -> int:
        raise NotImplementedError("GraphSource is read-only; use CsvSource for ingestion (PR-12)")

    def upsert(self, table: str, rows: list[dict[str, Any]], keys: list[str]) -> int:
        raise NotImplementedError("GraphSource is read-only; use CsvSource for ingestion (PR-12)")

    def close(self) -> None:
        self._conn.close()
        self._driver.close()


def _sanitize(name: str) -> str:
    import re

    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if sanitized and sanitized[0].isdigit():
        sanitized = "g_" + sanitized
    return sanitized.lower()
