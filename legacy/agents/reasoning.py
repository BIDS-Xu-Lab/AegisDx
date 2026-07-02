"""Physician-style rationale for each candidate diagnosis, optionally grounded
with PubMed references."""
from __future__ import annotations

from ..llm import call_model
from ..retrievers import PubMedRetriever


class ReasoningAgent:
    def __init__(self, model: str, add_references: bool = False, llm_provider: str = "openai") -> None:
        self.model = model
        self.llm_provider = llm_provider
        self.add_references = add_references
        self.pubmed = PubMedRetriever(model=model, llm_provider=llm_provider) if add_references else None

    @staticmethod
    def _messages_with_refs(case_description: str, diagnosis: str, references: str) -> list[dict[str, str]]:
        query = f"""You are provided with a case description, a diagnosis, and reference papers.
Craft a concise, physician-style rationale (≤100 words, ~3–5 sentences) for the diagnosis.
Cite the supporting papers by their numeric ID at the end of the sentence they support.
List the cited paper titles in the references field, in `"1. Title"` form.

Case Description: {case_description}
Diagnosis: {diagnosis}
References: {references}

Return ONLY a JSON object:
{{
    "reasoning": <str>,
    "references": <list>
}}"""
        return [{"role": "user", "content": query}]

    @staticmethod
    def _messages_without_refs(case_description: str, diagnosis: str) -> list[dict[str, str]]:
        query = f"""Craft a concise, physician-style rationale (≤100 words, ~3–5 sentences)
for the given diagnosis based on the case description.

Case Description: {case_description}
Diagnosis: {diagnosis}

Return ONLY a JSON object:
{{
    "reasoning": <str>
}}"""
        return [{"role": "user", "content": query}]

    @staticmethod
    def _overall_messages(
        case_description: str,
        diagnosis_list: list[str],
        reasoning_list: list[str],
        warning_diagnosis_list: list[str],
        add_references: bool,
    ) -> list[dict[str, str]]:
        ref_clause = (
            "Include references based on each diagnosis. Cite by paper title. "
            "Put all citations in a `references` list, numbered like `[1] Title`."
            if add_references
            else ""
        )
        ret_shape = (
            '{"reasoning": <str>, "references": <list>}'
            if add_references
            else '{"reasoning": <str>}'
        )
        query = f"""Synthesise an overall rationale considering all candidate and warning diagnoses.
Begin with a 1–2 sentence summary. Include every entry from both `diagnosis_list` and `warning_diagnosis_list`.

Case description: {case_description}
Candidate diagnoses: {diagnosis_list}
Reasonings: {reasoning_list}
Warning diagnoses: {warning_diagnosis_list}

{ref_clause}

Return ONLY a JSON object: {ret_shape}"""
        return [{"role": "user", "content": query}]

    async def reason(self, case_description: str, diagnosis: str) -> dict:
        if self.add_references and self.pubmed is not None:
            refs = await self.pubmed.search(diagnosis)
            messages = self._messages_with_refs(case_description, diagnosis, refs)
        else:
            messages = self._messages_without_refs(case_description, diagnosis)
        result = await call_model(
            messages,
            self.model,
            temperature=0.0,
            response_format="json",
            llm_provider=self.llm_provider,
        )
        # Normalise to {reasoning, references} for the downstream payload.
        return {
            "reasoning": result.get("reasoning", ""),
            "references": result.get("references", []),
        }

    async def reasoning_all(
        self,
        case_description: str,
        diagnosis_list: list[str],
        reasoning_list: list[str],
        warning_diagnosis_list: list[str],
    ) -> str:
        result = await call_model(
            self._overall_messages(
                case_description,
                diagnosis_list,
                reasoning_list,
                warning_diagnosis_list,
                add_references=self.add_references,
            ),
            self.model,
            temperature=0.0,
            response_format="json",
            llm_provider=self.llm_provider,
        )
        reasoning = result.get("reasoning", "")
        if self.add_references:
            refs = result.get("references", []) or []
            if refs:
                return reasoning + "\nReferences: \n" + "\n ".join(refs)
        return reasoning
