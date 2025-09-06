# reasoning/sql_generator.py
from infra.llm_client import LLMClient
from reasoning.prompt_templates import sql_generation_prompt

class SQLGenerator:
    def __init__(self, dialect: str):
        self.llm = LLMClient()
        self.dialect = dialect

    def generate(self, user_query: str, retrieved: list, examples: list[str]) -> str:
        messages = sql_generation_prompt(self.dialect, user_query, retrieved, examples)
        sql = self.llm.chat(messages)
        return sql.strip()

    # reasoning/sql_generator.py (add correction flow)
    def generate_with_correction(self, user_query: str, retrieved: list, examples: list[str], error: str) -> str:
        messages = sql_generation_prompt(self.dialect, user_query, retrieved, examples)
        messages.append({"role": "user", "content": f"The previous SQL failed with error:\n{error}\nPlease fix and return only corrected SQL."})
        return self.llm.chat(messages).strip()
