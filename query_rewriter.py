"""
Query rewriter: uses a local LLM to expand vague questions into specific,
keyword-rich queries before retrieval. Improves recall for ambiguous inputs.
"""

import os
from langchain_groq import ChatGroq

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=os.environ["GROQ_API_KEY"])
    return _llm


REWRITE_PROMPT = """You are a search query optimizer for a steel manufacturing knowledge base.
Rewrite the user's question into a precise, keyword-rich search query.
- Keep it under 20 words
- Include relevant technical terms
- Do NOT answer the question — only rewrite it
- Return only the rewritten query, nothing else

User question: {question}
Rewritten query:"""


def rewrite_query(question: str) -> str:
    """Return a rewritten version of the question optimized for retrieval."""
    prompt = REWRITE_PROMPT.format(question=question)
    rewritten = _get_llm().invoke(prompt).content.strip()
    # Fallback to original if the rewrite is empty or suspiciously long
    if not rewritten or len(rewritten) > 200:
        return question
    return rewritten
