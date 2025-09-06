# pipelines/ingest_schema.py
import uuid
import time
from sqlalchemy import create_engine, inspect, text
from infra.embedding_client import EmbeddingClient
from infra.qdrant_client import Qdrant
from config.settings import settings

# Table/column descriptions for richer embeddings
table_descriptions = {
    'TRADES': {
        'Trade ID': 'An internally generated identifier for this trade',
        'Price': 'The executed price of the trade for the specified instrument.',
        'Trade Time': 'The date and time at which the trade was executed',
        'Trade Status': '(Used to determine the status of this trade (A=Active, P=Pending, X=Cancelled)',
        'Account Name': 'Name of the client account they executed the trade from',
        'Quote ID': 'Unique ID of the Quote offered',
        'Instrument': 'The currency pair or financial instrument being traded (e.g., EUR/USD, GBP/JPY).',
        'Hedge ID': 'Identifier linking the trade to a corresponding hedge transaction, if any.'
    },
    'INSTRUMENT': {
        'Instrument ID': 'A unique identifier for the financial instrument.',
        'Ticker Symbol': 'The instrument’s name or symbol, representing a currency pair with instrument type',
        'Local': 'The currency or asset the client wants to trade from (the “give” side).',
        'Base': 'The currency or asset the client wants to receive to (the “get” side).'
    },
    'QUOTES': {
        'Quote ID': 'A unique identifier for this specific quote.',
        'RFQ_ID': 'The ID of the related Request For Quote (RFQ) that this quote is responding to.',
        'Price': 'The quoted price offered by the liquidity provider for the instrument.',
        'Status': 'The current state of the quote (e.g., Active, Pending, Cancelled).',
        'Instrument': 'The financial instrument (currency pair or asset) for which the quote is provided.'
    },
    'RFQ': {
        'RFQ_ID': 'A unique identifier for the Request For Quote (RFQ).',
        'Instrument': 'The financial instrument (e.g., currency pair) for which the quote is requested.',
        'Currency': 'The currency in which the RFQ is denominated.',
        'Tenor': 'The duration or term of the trade requested (e.g., 1W = 1 week, 1M = 1 month).',
        'Status': 'The current state of the RFQ (e.g., New=N, Active=A, Closed=C).',
        'Account name': 'Name of the client account requesting the quote.'
    },
    'CLIENT': {
        'Client Name': 'The official name of the client or organization.',
        'Client Id': 'A unique identifier assigned to each client.',
        'Account Id': 'A unique identifier for each client account.',
        'Account Name': 'The name of the specific client account associated with the client.'
    },
    'HEDGE': {
        'Hedge ID': 'A unique identifier for the hedge transaction.',
        'LP': 'The liquidity provider (LP) involved in the hedge.',
        'Hedge Date': 'The date and time when the hedge was executed.'
    }
}

def _table_doc(db, schema, table, description, row_count, tags):
    title = f"{schema}.{table}"
    desc = description or ""
    meta = f"Rows: {row_count}" if row_count is not None else ""
    return title, f"TABLE {schema}.{table}\n{desc}\n{meta}\nTAGS: {', '.join(tags)}"

def _column_doc(db, schema, table, column, dtype, nullable, desc, examples):
    # Look up description from table_descriptions if available
    col_desc = table_descriptions.get(table.upper(), {}).get(column, desc or "")
    title = f"{schema}.{table}.{column}"
    ex = f"Examples: {examples[:5]}" if examples else ""
    n = "NULLABLE" if nullable else "NOT NULL"
    return title, f"COLUMN {schema}.{table}.{column}\nType: {dtype} {n}\n{col_desc}\n{ex}"

def ingest():
    engine = create_engine(settings.TARGET_DB_URL)
    inspector = inspect(engine)

    embedder = EmbeddingClient()
    qdrant = Qdrant()
    qdrant.ensure_collection()

    ids, payloads, docs = [], [], []

    print("\n=== Starting schema ingestion ===")

    schema = "nlpSqlTest"

    
    for table in inspector.get_table_names(schema=schema):
        print(f"  [Table] {table}")
        cols = inspector.get_columns(table, schema=schema)

        # optional row_count
        row_count = None
        try:
            with engine.connect() as conn:
                rc = conn.execute(text(f"SELECT COUNT(*) FROM {schema}.{table}")).scalar()
                row_count = int(rc)
        except Exception as e:
            print(f"    [WARN] Could not fetch row count for {table}: {e}")

        # table doc
        t_id = str(uuid.uuid4())
        title, doc_text = _table_doc("db", schema, table, None, row_count, [])
        payloads.append({
            "object_type": "table",
            "db_name": "db",
            "schema_name": schema,
            "table_name": table,
            "title": title,
            "doc_text": doc_text,
            "keywords": [schema, table],
            "popularity": 0.5,
            "source_id": t_id
        })
        ids.append(t_id)
        docs.append(doc_text)

        # column docs
        for c in cols:
            c_id = str(uuid.uuid4())
            c_title, c_doc = _column_doc(
                "db", schema, table, c["name"], str(c["type"]),
                c.get("nullable", True), c.get("comment"), None
            )
            payloads.append({
                "object_type": "column",
                "db_name": "db",
                "schema_name": schema,
                "table_name": table.lower(),
                "column_name": c["name"].lower(),
                "title": c_title,
                "doc_text": c_doc,
                "keywords": [
                    schema,
                    table,
                    c["name"],
                    c["name"].replace(" ", "_"),
                    c["name"].lower().replace(" ", "_")
                ],
                "data_type": str(c["type"]),
                "popularity": 0.5,
                "source_id": c_id
            })
            ids.append(c_id)
            docs.append(c_doc)

    # Embed + upsert in small batches to avoid timeouts
    B = 8  # smaller batch size for both embedding and Qdrant
    print(f"\n=== Processing {len(docs)} docs in batches of {B} ===")
    for i in range(0, len(docs), B):
        batch_docs = docs[i:i+B]
        batch_ids = ids[i:i+B]
        batch_payloads = payloads[i:i+B]

        print(f"\n  [Batch {i//B + 1}] Embedding {len(batch_docs)} docs...")
        try:
            batch_vectors = embedder.embed(batch_docs)
        except Exception as e:
            print(f"    [ERROR] Embedding failed: {e}")
            continue

        print(f"    -> Got {len(batch_vectors)} vectors, upserting to Qdrant...")
        try:
            qdrant.upsert(batch_ids, batch_vectors, batch_payloads)
        except Exception as e:
            print(f"    [ERROR] Qdrant upsert failed: {e}")
            continue

        # Small pause to keep Qdrant responsive
        time.sleep(0.5)

    print("\n=== Ingestion complete ===")

if __name__ == "__main__":
    ingest()
