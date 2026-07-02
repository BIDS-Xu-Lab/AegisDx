"""Cluster equivalent diagnoses (e.g. 'AMI' / 'acute MI' / 'STEMI') so that
votes from the diagnosis ensemble can be tallied."""
from __future__ import annotations

from ..llm import call_model


class AggregateAgent:
    def __init__(self, model: str = "gpt-4o-mini", llm_provider: str = "openai") -> None:
        self.model = model
        self.llm_provider = llm_provider

    @staticmethod
    def _messages(diagnosis_list: list[str]) -> list[dict[str, str]]:
        query = f"""Cluster these medical diagnoses into groups of similar conditions. Different
group names mean different diagnoses; each diagnosis appears in exactly one group.
Return ONLY a JSON array, each item with keys 'diagnosis' (the original) and 'group'.

Format:
[{{"diagnosis": "diagnosis 1", "group": "group 1"}}, ...]

Diagnoses: {diagnosis_list}"""
        return [{"role": "user", "content": query}]

    async def aggregate(self, diagnosis_list: list[str]) -> list[dict[str, str]]:
        return await call_model(
            self._messages(diagnosis_list),
            self.model,
            temperature=0.0,
            response_format="json",
            llm_provider=self.llm_provider,
        )
