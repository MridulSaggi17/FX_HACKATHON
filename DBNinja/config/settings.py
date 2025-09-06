# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Embedding + LLM
    API_BASE: str = "https://5z38w37oo6-vpce-0e881c3ec15437336.execute-api.eu-west-1.amazonaws.com/qwen"
    API_KEY: str = "1AROkExTzj6uweMBgylwoaozPLWpQYxS61yvWqrj"
    EMBEDDINGS_PATH: str = "/v1/embeddings"
    RERANK_PATH: str = "/rerank"
    PREDICT_PATH: str = "/predict"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "nlp_sql_schema"
    EMBEDDING_DIM: int = 1536  # adjust to your model

    # DB metadata connection string
    METADATA_DB_URL: str = "sqlite:///./metadata.db"
    # config/settings.py
    TARGET_DB_URL: str = "mysql+pymysql://root:Panda230@localhost:3306/nlpSqlTest"


    # Retrieval
    TOPK_SEMANTIC: int = 30
    TOPK_KEYWORD: int = 30
    FINAL_TOPK: int = 12
    ALPHA: float = 0.7  # fusion weight

settings = Settings()
