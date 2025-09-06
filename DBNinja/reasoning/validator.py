# reasoning/validator.py
from sqlalchemy import create_engine, text
from config.settings import settings

class SQLValidator:
    def __init__(self):
        self.engine = create_engine(settings.TARGET_DB_URL)

    def dry_run(self, sql: str) -> tuple[bool, str | None]:
        try:
            with self.engine.connect() as conn:
                # Vendor-specific dry-run can be different; generic approach:
                conn.execute(text(f"EXPLAIN {sql}"))
            return True, None
        except Exception as e:
            return False, str(e)
