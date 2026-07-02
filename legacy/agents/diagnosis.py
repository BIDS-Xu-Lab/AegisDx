"""Sample N candidate diagnoses for a case description."""
from __future__ import annotations

import asyncio

from ..llm import call_model


class DiagnosisAgent:
    def __init__(self, model: str, num_inference: int = 10, llm_provider: str = "openai") -> None:
        self.model = model
        self.num_inference = num_inference
        self.llm_provider = llm_provider

    @staticmethod
    def _initial_messages(case_description: str) -> list[dict[str, str]]:
        query = f"""Please analyze this patient's case and provide ten possible diagnosis results.
You must return ONLY a JSON array of objects, each with the key 'diagnosis' (str).
Output format:
[
    {{"diagnosis": "diagnosis 1"}},
    {{"diagnosis": "diagnosis 2"}}
]
The case description is: {case_description}"""
        return [{"role": "user", "content": query}]

    @staticmethod
    def _reference_messages(case_description: str, diagnosis_list: list[str]) -> list[dict[str, str]]:
        joined = ", ".join(diagnosis_list)
        query = f"""You are provided with a case description and a list of possible diagnoses.
Analyze the case, consider the listed diagnoses, and pick exactly one diagnosis.
Return ONLY a JSON object with keys 'reasoning' (str) and 'diagnosis' (str):
{{
    "reasoning": <str>,
    "diagnosis": <str>
}}
Case description: {case_description}
Possible diagnoses: {joined}
Output:"""
        return [{"role": "user", "content": query}]

    @staticmethod
    def _additional_messages(case_description: str, diagnosis_list: list[str]) -> list[dict[str, str]]:
        joined = ", ".join(diagnosis_list)
        query = f"""Please analyze this patient's case and provide one possible diagnosis that is
different from the previous diagnoses but still consistent with the case description.
Return ONLY a JSON object with keys 'reasoning' (str) and 'diagnosis' (str):
{{
    "reasoning": <str>,
    "diagnosis": <str>
}}
Case description: {case_description}
Previous diagnoses: {joined}
Output:"""
        return [{"role": "user", "content": query}]

    async def diagnose(self, case_description: str) -> list[dict[str, str]]:
        """Return `num_inference` candidate `{diagnosis, reasoning}` items."""
        initial = await call_model(
            self._initial_messages(case_description),
            self.model,
            temperature=0.5,
            response_format="json",
            llm_provider=self.llm_provider,
        )
        diagnosis_list = [r["diagnosis"] for r in initial]
        ref_msgs = self._reference_messages(case_description, diagnosis_list)
        results = await asyncio.gather(
            *(
                call_model(
                    ref_msgs,
                    self.model,
                    temperature=0.5,
                    response_format="json",
                    llm_provider=self.llm_provider,
                )
                for _ in range(self.num_inference)
            )
        )
        return [r for r in results if isinstance(r, dict) and "diagnosis" in r]

    async def additional_diagnosis(self, case_description: str, diagnosis_list: list[str]) -> dict[str, str]:
        return await call_model(
            self._additional_messages(case_description, diagnosis_list),
            self.model,
            temperature=0.5,
            response_format="json",
            llm_provider=self.llm_provider,
        )
