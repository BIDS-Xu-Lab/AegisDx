import os
from utils.new_utils import call_model
import asyncio
from typing import List
from utils.utils import get_diagnosis, gpt_parse
import re

def extract_diagnosis(response: str) -> str:
    pattern = r"<diagnosis>(.*?)</diagnosis>"
    match = re.search(pattern, response)
    if match:
        return match.group(1)
    else:
        return ""

_specialty_list = [
    "Allergy & Immunology",
    "Cardiology",
    "Dermatology",
    "Endocrinology",
    "Gastroenterology",
    "Hematology",
    "Infectious Disease",
    "Nephrology",
    "Neurology",
    "Pediatrics",
    "Psychiatry",
    "Pulmonology",
    "Rheumatology",
    "Surgery",
]

class DiagnosisAgent:
    def __init__(self, model: str, num_inference: int = 1, llm_provider: str = "openai", by_specialty: bool = False):
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.num_inference = num_inference
        self.llm_provider = llm_provider
        self.by_specialty = by_specialty

    def _messages(self, case_description: str) -> str:
        query = f"""Please analyze this patient's case and provide one possible diagnosis result. You must return ONLY a JSON object, with the key: 'reasoning' (str) and 'diagnosis' (str).'
        The case description is: {case_description}"""
        messages = [{"role": "user", "content": query}]
        return messages

    def _initial_diagnosis_messages(self, case_description: str) -> str:
        query = f"""Please analyze this patient's case and provide ten possible diagnosis results. 
        You must return ONLY ten JSON objects, with the key: 'diagnosis' (str).'
        The output format must be:
        [
            {{"diagnosis": "diagnosis 1"}},
            {{"diagnosis": "diagnosis 2"}},
            {{"diagnosis": "diagnosis 3"}},
            {{"diagnosis": "diagnosis 4"}},
            {{"diagnosis": "diagnosis 5"}},
        ]
        The case description is: {case_description}"""
        messages = [{"role": "user", "content": query}]
        return messages

    def _select_specialty_messages(self, diagnosis_list: List[str], num_specialty: int) -> str:
        query = f"""Please analyze this patient's case and select {num_specialty} specialties from the following list: {_specialty_list}.
        You must return ONLY {num_specialty} JSON objects, with the key: 'specialty' (str).
        The output format must be:
        [
            {{"specialty": "<specialty 1>"}},
            {{"specialty": "<specialty 2>"}},
            {{"specialty": "<specialty 3>"}},
            ...
        ]
        The diagnoses are: {diagnosis_list}"""
        messages = [{"role": "user", "content": query}]
        return messages

    def _specialty_diagnosis_messages(self, case_description: str, specialty: str, diagnosis_list: List[str]) -> str:
        query = f"""Please analyze this patient's case and provide one possible diagnosis result that is related to the specialty: {specialty}.
        You must return ONLY a JSON object, with the key: 'reasoning' (str) and 'diagnosis' (str).
        The output format must be:
        {{
            "reasoning": <str>,
            "diagnosis": <str>
        }}
        The diagnoses that are related to the specialty are: {diagnosis_list}
        The case description is: {case_description}"""
        messages = [{"role": "user", "content": query}]
        return messages

    def _reference_messages(self, case_description: str, diagnosis_list: List[str]) -> str:
        diagnosis_list_str = ", ".join(diagnosis_list)
        query = f"""You are provided with a case description and a list of possible diagnoses. Your task is to analyze the case description, refer to the possible diagnoses and provide only  one diagnosis result. 
        You must return ONLY a JSON object, with the key: 'reasoning' (str) and 'diagnosis' (str).
        The case description is: {case_description}
        The possible diagnoses are: {diagnosis_list_str}
        Please output the result in the following json format, with the key "reasoning" and "diagnosis":
        {{
            "reasoning": <str>,
            "diagnosis": <str>
        }}
        Output:
        """
        messages = [{"role": "user", "content": query}]
        return messages

    def _additional_diagnosis_messages(
        self, case_description: str, diagnosis_list: List[str]
    ) -> str:
        diagnosis_list_str = ", ".join(diagnosis_list)
        query = f"""Please analyze this patient's case and provide one possible diagnosis result. The diagnosis should be different from the previous diagnoses and should be related to the case description.
        You must return ONLY a JSON object, with the key: 'reasoning' (str), 'diagnosis' (str).'
        The case description is: {case_description}
        The previous diagnoses are: {diagnosis_list_str}
        Please output the result in the following json format, with the key "reasoning" and "diagnosis":
        {{
            "reasoning": <str>,
            "diagnosis": <str>
        }}
        Output:
        """
        messages = [{"role": "user", "content": query}]
        return messages

    async def step_diagnosis(self, case_description: str) -> str:
        messages = self._initial_diagnosis_messages(case_description)
        initial_results = await call_model(messages, self.model, temperature=0.7, response_format="json", llm_provider=self.llm_provider)
        diagnosis_list = [result['diagnosis'] for result in initial_results]
        messages = self._reference_messages(case_description, diagnosis_list)
        result = await call_model(messages, self.model, temperature=0.5, response_format="json", llm_provider=self.llm_provider)
        return result

    async def diagnose2(self, case_description: str) -> str:
        results = [self.step_diagnosis(case_description) for _ in range(self.num_inference)]
        results = await asyncio.gather(*results)
        return results

    async def diagnose(
        self, case_description: str, num_inference: int | None = None
    ) -> str:
        messages = self._initial_diagnosis_messages(case_description)
        initial_results = await call_model(messages, self.model, temperature=0.5, response_format="json", llm_provider=self.llm_provider)
        diagnosis_list = [result['diagnosis'] for result in initial_results]
        num_inference = self.num_inference if num_inference is None else num_inference
        if self.by_specialty:
            specialty_messages = self._select_specialty_messages(diagnosis_list, num_inference)
            specialty_list = await call_model(specialty_messages, self.model, temperature=0.5, response_format="json", llm_provider=self.llm_provider)
            specialty_list = [result['specialty'] for result in specialty_list]
            results = [
                await call_model(
                    self._specialty_diagnosis_messages(case_description, specialty, diagnosis_list),
                    self.model, temperature=0.5, response_format="json", llm_provider=self.llm_provider
                )
                for specialty in specialty_list
            ]
            results = await asyncio.gather(*results)
            return results
        else:
            ref_messages = self._reference_messages(case_description, diagnosis_list)
            results = [
                call_model(ref_messages, self.model, temperature=0.5, response_format="json", llm_provider=self.llm_provider)
                for _ in range(num_inference)
            ]
            results = await asyncio.gather(*results)
            # for idx, r in enumerate(results):
            #     try:
            #         print("r", idx, r['diagnosis'])
            #     except Exception as e:
            #         print("r", idx, ref_messages)
            #         print(r)
            #         print("error", e)
            # print("results", results)
            return results

    async def diagnose_bk(
        self, case_description: str, num_inference: int | None = None
    ) -> str:
        messages = self._messages(case_description)
        num_inference = self.num_inference if num_inference is None else num_inference
        results = [
            call_model(messages, self.model, temperature=0.7, response_format="json", llm_provider=self.llm_provider)
            for _ in range(num_inference)
        ]
        results = await asyncio.gather(*results)
        # results = [{'diagnosis': extract_diagnosis(result)} for result in get_diagnosis(case_description, self.model, self.api_key, num_inference)]
        # print(results)
        return results

    async def additional_diagnosis(
        self, case_description: str, diagnosis_list: List[str]
    ) -> str:
        messages = self._additional_diagnosis_messages(case_description, diagnosis_list)
        result = await call_model(
            messages, self.model, temperature=0.5, response_format="json", llm_provider=self.llm_provider
        )
        return result
