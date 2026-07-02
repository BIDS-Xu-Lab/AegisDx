"""Pick must-not-miss differentials from a curated symptom→diagnosis registry."""
from __future__ import annotations

import json
from pathlib import Path

from ..llm import call_model

_DEFAULT_REGISTRY = Path(__file__).resolve().parent.parent / "data" / "symptom_diag.json"


class WarningAgent:
    """Choose 3 high-mortality, time-sensitive diagnoses from a curated list.

    The registry — a mapping `symptom → [must-not-miss diagnoses]` — is loaded
    from `aegisdx/data/symptom_diag.json` by default and flattened into a
    deduplicated allow-list that the LLM must copy from verbatim.
    """

    def __init__(
        self,
        model: str,
        llm_provider: str = "openai",
        registry_path: Path | str | None = None,
    ) -> None:
        self.model = model
        self.llm_provider = llm_provider
        path = Path(registry_path) if registry_path else _DEFAULT_REGISTRY
        with open(path) as f:
            self.registry: dict[str, list[str]] = json.load(f)

    @staticmethod
    def _flat_terms(registry: dict[str, list[str]]) -> list[str]:
        seen, terms = set(), []
        for diags in registry.values():
            for d in diags:
                if d not in seen:
                    seen.add(d)
                    terms.append(d)
        return terms

    def _messages(self, case_description: str) -> list[dict[str, str]]:
        allowed = self._flat_terms(self.registry)
        system = (
            "You are an emergency medicine expert.\n\n"
            "Below are must-not-miss diagnoses guidelines:\n"
            f"{json.dumps(allowed, ensure_ascii=False)}"
        )
        user = (
            f"Patient case:\n{case_description}\n\n"
            "Briefly reason: what are the most dangerous diagnostic categories for this presentation? "
            "Then pick exactly 3 diagnoses verbatim from the allowed list, one per distinct category.\n\n"
            "Output ONLY a JSON list of exactly 3 items:\n"
            '[{"warning_diagnosis": "<warning diagnosis>", "reason": "<≤20 words>"}, ...]'
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    async def diagnose(self, case_description: str) -> list[dict[str, str]]:
        return await call_model(
            self._messages(case_description),
            self.model,
            temperature=0.3,
            response_format="json",
            llm_provider=self.llm_provider,
        )
