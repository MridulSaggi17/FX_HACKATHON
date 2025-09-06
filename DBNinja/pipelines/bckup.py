# pipelines/ingest_schema.py
import uuid
import time
from sqlalchemy import create_engine, inspect, text
from infra.embedding_client import EmbeddingClient
from infra.qdrant_client import Qdrant
from config.settings import settings

table_descriptions = {
    'INSTRUMENT': {
        'Instrument ID': 'A unique identifier for the financial instrument.',
        'Ticker Symbol': 'The instrument’s name or symbol, representing a currency pair with instrument type',
        'Local': 'The currency or asset the client wants to trade from (the “give” side).',
        'Base': 'The currency or asset the client wants to receive to (the “get” side).'
    }
}

def _table_doc(schema, table, description, row_count):
    desc = description or ""
    meta = f"Rows: {row_count}" if row_count is not None else ""
    # Keep doc_text short for embeddings
    return f"{table} table. {desc} {meta}".strip()

def _column_doc(table, column, dtype, nullable, desc):
    col_desc = table_descriptions.get(table.upper(), {}).get(column, desc or "")
    n = "nullable" if nullable else "not nullable"
    return f"{column} column ({dtype}, {n}). {col_desc}".strip()

def ingest():
    engine = create_engine(settings.TARGET_DB_URL)
    inspector = inspect(engine)

    embedder = EmbeddingClient()
    qdrant = Qdrant()
    qdrant.ensure_collection()

    ids, payloads, docs = [], [], []

    for schema in inspector.get_schema_names():
        for table in inspector.get_table_names(schema=schema):
            cols = inspector.get_columns(table, schema=schema)

            # Row count (optional)
            row_count = None
            try:
                with engine.connect() as conn:
                    rc = conn.execute(text(f"SELECT COUNT(*) FROM {schema}.{table}")).scalar()
                    row_count = int(rc)
            except Exception:
                pass

            # Table doc (shortened)
            t_id = str(uuid.uuid4())
            table_doc = _table_doc(schema, table, None, row_count)
            ids.append(t_id)
            docs.append(table_doc)
            payloads.append({
                "object_type": "table",
                "schema_name": schema,
                "table_name": table,
                "keywords": [schema, table],
                "source_id": t_id
            })

            # Column docs (shortened)
            for c in cols:
                c_id = str(uuid.uuid4())
                col_doc = _column_doc(table, c["name"], str(c["type"]), c.get("nullable", True), c.get("comment"))
                ids.append(c_id)
                docs.append(col_doc)
                payloads.append({
                    "object_type": "column",
                    "schema_name": schema,
                    "table_name": table.lower(),
                    "column_name": c["name"].lower(),
                    "keywords": [
                        schema,
                        table,
                        c["name"],
                        c["name"].replace(" ", "_"),
                        c["name"].lower().replace(" ", "_")
                    ],
                    "data_type": str(c["type"]),
                    "source_id": c_id
                })

    # Embed + upsert in small batches
    B = 8
    for i in range(0, len(docs), B):
        batch_docs = docs[i:i+B]
        batch_ids = ids[i:i+B]
        batch_payloads = payloads[i:i+B]

        try:
            batch_vectors = embedder.embed(batch_docs)
            qdrant.upsert(batch_ids, batch_vectors, batch_payloads)
        except Exception as e:
            print(f"[ERROR] Batch {i//B+1} failed: {e}")
            continue

        time.sleep(0.3)  # small pause to keep Qdrant responsive

if __name__ == "__main__":
    ingest()
