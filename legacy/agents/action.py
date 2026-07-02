"""Diagnostic actions per candidate diagnosis, plus an overall management plan."""
from __future__ import annotations

from ..llm import call_model


class ActionAgent:
    def __init__(self, model: str, llm_provider: str = "openai") -> None:
        self.model = model
        self.llm_provider = llm_provider

    @staticmethod
    def _action_messages(case_description: str, diagnosis: str) -> list[dict[str, str]]:
        query = f"""List the exact, specific clinical actions needed to confirm the diagnosis.
Be concise and concrete.

Case description: {case_description}
Diagnosis: {diagnosis}

Return ONLY a JSON object:
{{
    "actions": <str>
}}"""
        return [{"role": "user", "content": query}]

    @staticmethod
    def _management_messages(
        case_description: str,
        diagnosis_list: list[str],
        action_list: list[str],
        overall_reasoning: str,
    ) -> list[dict[str, str]]:
        query = f"""Produce an overall management plan that takes all candidate diagnoses into account.
Start with a 1–2 sentence summary. Be concise and concrete.

Case description: {case_description}
Diagnoses: {diagnosis_list}
Overall reasoning: {overall_reasoning}
Per-diagnosis actions: {action_list}

Return ONLY a JSON object:
{{
    "management_plan": <str>
}}"""
        return [{"role": "user", "content": query}]

    async def action_plan(self, case_description: str, diagnosis: str) -> str:
        result = await call_model(
            self._action_messages(case_description, diagnosis),
            self.model,
            temperature=0.0,
            response_format="json",
            llm_provider=self.llm_provider,
        )
        return result.get("actions", "")

    async def management_plan(
        self,
        case_description: str,
        diagnosis_list: list[str],
        action_list: list[str],
        overall_reasoning: str,
    ) -> str:
        result = await call_model(
            self._management_messages(case_description, diagnosis_list, action_list, overall_reasoning),
            self.model,
            temperature=0.0,
            response_format="json",
            llm_provider=self.llm_provider,
        )
        return result.get("management_plan", "")
