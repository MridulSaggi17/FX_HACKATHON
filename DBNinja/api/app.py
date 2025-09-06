# api/app.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
from pipelines.ingest_schema import ingest
from retrieval.hybrid_retriever import HybridRetriever
from reasoning.sql_generator import SQLGenerator
from reasoning.validator import SQLValidator
from config.settings import settings

app = FastAPI(title="NLP-to-SQL")

class QueryRequest(BaseModel):
    question: str
    dialect: str = "postgres"
    use_rerank: bool = True
    top_k: int = settings.FINAL_TOPK

@app.post("/ingest")
def ingest_endpoint():
    ingest()
    return {"status": "ok"}

@app.post("/query")
def query_endpoint(req: QueryRequest):
    retriever = HybridRetriever(settings.TOPK_SEMANTIC, settings.TOPK_KEYWORD, settings.ALPHA)
    results = retriever.retrieve(req.question, use_rerank=req.use_rerank)
    topK = results[:req.top_k]

    # Collect small in-context examples from payload where object_type == "example"
    examples = [r["payload"]["doc_text"] for r in topK if r["payload"].get("object_type") == "example"][:3]

    gen = SQLGenerator(req.dialect)
    sql = gen.generate(req.question, topK, examples)

    validator = SQLValidator()
    ok, err = validator.dry_run(sql)
    if not ok:
        sql = gen.generate_with_correction(req.question, topK, examples, err)

    return {
        "candidates": [
            {
                "id": r["id"],
                "title": r["payload"].get("title"),
                "object_type": r["payload"].get("object_type"),
                "table": r["payload"].get("table_name"),
                "column": r["payload"].get("column_name"),
                "score": round(r["score"], 4),
                "sem": round(r.get("sem", 0.0), 4),
                "kw": round(r.get("kw", 0.0), 4)
            } for r in topK
        ],
        "sql": sql
    }
