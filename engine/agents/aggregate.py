import os
from typing import List
from utils.new_utils import call_model
from collections import defaultdict


class AggregateAgent:
    def __init__(self, model: str, llm_provider: str = "openai"):
        self.model = os.environ.get("OPENDX_FAST_MODEL", "gpt-4o-mini")
        self.llm_provider = llm_provider

    def _messages(self, diagnosis_list: List[str]) -> str:
        # query = f"""Please analyze these medical diagnoses and cluster them into groups of similar conditions. You will be provided with a list of diagnoses, and you should group the diagnoses into different groups.
        # The output must be ONLY a dictionary in a json format. The key is the diagnosis and the value is the group name.
        # The group name should be a short description of the diagnosis, each diagnosis should be only in one group. Different group names mean different diagnoses. Strictly follow the format in the example.
        # For instances, the diagnoses are: "A1", "B1", "A2", "A3", "A1", "C1", "B2". \n The output should be: '{{\n"A1": "A",\n "B1": "B",\n "A2": "A",\n "A3": "A",\n "A1": "A",\n "C1": "C",\n "B2": "B"\n}}'.
        # The provided diagnoses are: {diagnosis_list}
        query = f"""Please analyze these medical diagnoses and cluster them into groups of similar conditions. You will be provided with a list of diagnoses, and you should group the similar diagnoses into different groups. 
        The group name should be a short description of the diagnosis, each diagnosis should be only in one group. Different group names mean different diagnoses. 
        You must return ONLY json objects, with the key: 'diagnosis' (str) and 'group' (str). The key is the orignal diagnosis and the value is the group name. 
        The output format must be:
        [{{"diagnosis": "diagnosis 1", "group": "group 1"}}, {{"diagnosis": "diagnosis 2", "group": "group 2"}}, {{"diagnosis": "diagnosis 3", "group": "group 3"}}, ...].
        The provided diagnoses are: {diagnosis_list}
        """

        # """
        # query = f"""Please analyze these medical diagnoses and cluster them into groups of similar conditions. You will be provided with a list of diagnoses, and you should group the diagnoses into different groups.
        # The output must be ONLY a dictionary in a json format. The key is the diagnosis and the value is the group name.
        # The group name should be the unique name of the diagnosis, each diagnosis should be only in one group. Different group names mean different diagnoses. Strictly follow the format in the example.
        # For instances, the diagnoses are: "A1", "B1", "A2", "A3", "A1", "C1", "B2". \n The output should be: '{{\n"A1": "A",\n "B1": "B",\n "A2": "A",\n "A3": "A",\n "A1": "A",\n "C1": "C",\n "B2": "B"\n}}'.
        # The provided diagnoses are: {diagnosis_list}

        # """
        # query = f"""Please analyze these medical diagnoses and cluster them into groups of similar conditions. You will be provided with a list of diagnoses, and you should put the diagnoses into different groups.
        # The group name should be the unique name of the diagnosis, all diagnoses in the same group should be similar, they are just different names for the same diagnosis. Different group names mean different diagnoses.
        # You must return ONLY json objects, with the key: 'diagnosis' (str) and 'group' (str). The key is the orignal diagnosis and the value is the group name.
        # The output format must be:
        # [{{"diagnosis": "diagnosis 1", "group": "group 1"}}, {{"diagnosis": "diagnosis 2", "group": "group 2"}}, {{"diagnosis": "diagnosis 3", "group": "group 3"}}, ...].
        # The provided diagnoses are: {diagnosis_list}
        # """
        messages = [{"role": "user", "content": query}]
        return messages

    def _rerank_messages(
        self, case_description: str, diagnosis_list: List[str], rewards: List[float]
    ) -> str:
        query = f"""You are provided with a case description and a list of diagnoses. You task is to rerank the diagnoses based on the rewards.
        The case description is: {case_description}. The diagnoses are: {diagnosis_list} and the rewards are: {rewards}.
        The output must be ONLY a list of json objects, with the key: 'diagnosis' (str). The output format must be: 
        [{{"diagnosis": diagnosis1}}, {{"diagnosis": diagnosis2}}, {{"diagnosis": diagnosis3}}, ...].
        """
        messages = [{"role": "user", "content": query}]
        return messages

    async def aggregate(self, diagnosis_list: List[str]) -> str:
        messages = self._messages(diagnosis_list)
        result = await call_model(
            messages,
            self.model,
            temperature=0,
            response_format="json",
            llm_provider=self.llm_provider,
        )
        return result

    async def aggregate2(
        self, diagnosis_list: List[str], rewards: List[float] | None = None
    ) -> str:
        messages = self._messages(diagnosis_list)
        result = await call_model(
            messages,
            self.model,
            temperature=0,
            response_format="json",
            llm_provider=self.llm_provider,
        )
        diag_list = [item["group"] for item in result]
        # diag_list = list(result.values())
        if rewards is None:
            rewards = [1.0] * len(diag_list)
        diag_rewards = defaultdict(lambda: 0.0)
        for idx, diag in enumerate(diag_list):
            diag_rewards[diag] += rewards[idx]
        sort_diag = sorted(diag_rewards.items(), key=lambda x: x[1], reverse=True)
        new_diag_list = [diag[0] for diag in sort_diag]
        return new_diag_list

    async def rerank(
        self, case_description: str, diagnosis_list: List[str], rewards: List[float]
    ) -> str:
        if rewards is None:
            rewards = [1.0] * len(diagnosis_list)
        messages = self._rerank_messages(case_description, diagnosis_list, rewards)
        result = await call_model(
            messages,
            self.model,
            temperature=0,
            response_format="json",
            llm_provider=self.llm_provider,
        )
        return result

    async def review(
        self,
        case_description: str,
        diagnosis_list: List[str],
        rewards: List[float] | None = None,
    ) -> str:

        if rewards is None:
            rewards = [1.0] * len(diagnosis_list)

        messages = self._review_messages(case_description, diagnosis_list, rewards)
        result = await call_model(
            messages,
            self.model,
            temperature=0,
            response_format="json",
            llm_provider=self.llm_provider,
        )
        return result
