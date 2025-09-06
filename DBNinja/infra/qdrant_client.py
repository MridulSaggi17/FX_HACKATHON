# infra/qdrant_client.py
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchText
from config.settings import settings

class Qdrant:
    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY,timeout=60.0)

    def ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if settings.QDRANT_COLLECTION not in existing:
            self.client.recreate_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(size=settings.EMBEDDING_DIM, distance=Distance.COSINE)
            )
            # create payload indexes
            self.client.create_payload_index(settings.QDRANT_COLLECTION, field_name="doc_text", field_schema={"type": "text"})
            self.client.create_payload_index(settings.QDRANT_COLLECTION, field_name="table_name", field_schema={"type": "keyword"})
            self.client.create_payload_index(settings.QDRANT_COLLECTION, field_name="column_name", field_schema={"type": "keyword"})
            self.client.create_payload_index(settings.QDRANT_COLLECTION, field_name="keywords", field_schema={"type": "keyword"})

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points, wait=True)

    def semantic_search(self, vector, top_k: int):
        return self.client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=vector,
            limit=top_k
        )

    def keyword_search(self, query: str, top_k: int):
        # full-text match on doc_text
        flt = Filter(must=[FieldCondition(key="doc_text", match=MatchText(text=query))])
        return self.client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=None,  # keyword-only
            query_filter=flt,
            limit=top_k
        )
