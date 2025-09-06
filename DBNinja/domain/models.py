# domain/models.py
from pydantic import BaseModel
from typing import Optional, List, Literal
from uuid import UUID

class TableMeta(BaseModel):
    id: UUID
    db_name: str
    schema_name: str
    table_name: str
    description: Optional[str] = None
    row_count: Optional[int] = None
    tags: List[str] = []

class ColumnMeta(BaseModel):
    id: UUID
    table_id: UUID
    column_name: str
    data_type: str
    is_nullable: bool
    description: Optional[str] = None
    example_values: Optional[list] = None
    tags: List[str] = []

class PayloadDoc(BaseModel):
    object_type: Literal["table","column","example"]
    db_name: str
    schema_name: str
    table_name: str
    column_name: Optional[str] = None
    title: str
    doc_text: str
    keywords: List[str] = []
    data_type: Optional[str] = None
    popularity: float = 0.5
    last_updated: Optional[str] = None
    source_id: str
