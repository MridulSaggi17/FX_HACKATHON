# reasoning/prompt_templates.py
SYSTEM_PROMPT = """You are a meticulous SQL expert. Follow these rules:
- Output only valid SQL for the target dialect.
- Prefer using retrieved tables/columns only. If unsure, ask for clarification.
- Never hallucinate tables or columns. 
- If a function is dialect-specific, use the target dialect specified.
- Use MySQL syntax for date/time and string functions.
- Enclose identifiers with backticks if they contain spaces (e.g., `Trade ID`).

"""

def sql_generation_prompt(dialect: str, user_query: str, retrieved_context: list, examples: list[str]) -> list[dict]:
    dialect: str = "mysql"
    context_blocks = []
    for r in retrieved_context:
        p = r["payload"]
        t = p.get("title")
        d = p.get("doc_text", "")[:800]
        context_blocks.append(f"- {t}\n{d}")

    ctx = "\n".join(context_blocks)
    ex = "\n".join(f"Example:\n{e}" for e in examples[:3])

    user_block = f"""User request: {user_query}
Target SQL dialect: {dialect}

Relevant schema:
{ctx}

{ex}

Return only SQL, no explanations."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_block}
    ]
