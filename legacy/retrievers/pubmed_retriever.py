"""PubMed retriever: produce a compact, LLM-friendly digest of case reports
relevant to a candidate diagnosis."""
from __future__ import annotations

import json
import os

from ..llm import call_model
from .pubmed_client import PubMedClient


class PubMedRetriever:
    """Generate a PubMed query for the diagnosis, search, and format results.

    The retriever is intentionally simple — a single LLM-generated query plus
    one retry with a reworded query when the first search returns nothing.
    """

    MAX_RETRIES = 3
    NUM_RESULTS = 5

    def __init__(self, model: str, llm_provider: str = "openai") -> None:
        self.model = model
        self.llm_provider = llm_provider
        self.client = PubMedClient(
            email=os.environ["PUBMED_EMAIL"],
            api_key=os.environ.get("PUBMED_API_KEY"),
        )

    async def _generate_query(self, diagnosis: str, previous: str | None) -> str:
        if previous:
            prompt = (
                f"Generate a PubMed search query for case reports of: {diagnosis}. "
                f"The previous query '{previous}' returned no results — try a different phrasing. "
                "Output only the query."
            )
        else:
            prompt = f"""Generate a PubMed search query to find case reports for: {diagnosis}.
Make sure the query is compatible with Entrez.esearch. Keep it short.

You may use field tags like [Title], [MeSH Terms], [Journal], date ranges, AND/OR/NOT,
and quoted phrases. Output the query only."""
        result = await call_model(
            [{"role": "user", "content": prompt}],
            self.model,
            temperature=0.0,
            llm_provider=self.llm_provider,
        )
        return (result or "").strip().strip('"').strip("'")

    async def search(self, diagnosis: str) -> str:
        query: str | None = None
        articles: list = []
        for _ in range(self.MAX_RETRIES):
            query = await self._generate_query(diagnosis, previous=query)
            articles = await self.client.search_articles(query, self.NUM_RESULTS)
            if articles:
                break

        if not articles:
            return ""

        formatted = [
            {
                "id": idx + 1,
                "title": a["title"],
                "abstract": a["abstract"],
                "url": a.get("pubmed_url", ""),
            }
            for idx, a in enumerate(articles)
        ]
        return json.dumps(formatted, indent=2)
