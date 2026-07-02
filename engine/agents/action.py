from retrievers import PubMedRetriever
from utils.new_utils import reasoning_agent_with_pubmed
from typing import Tuple, List
from utils.new_utils import call_model


class ActionAgent:
    def __init__(self, model: str, llm_provider: str = "openai"):
        self.model = model
        self.pubmed_retriever = PubMedRetriever(model=model)
        self.llm_provider = llm_provider

    def _messages(self, case_description: str, diagnosis: str, verification: str) -> str:
        query = f"""You are provided with a case description and a diagnosis. You task is to give what exact actions are needed to confirm that the diagnosis is correct.
        List the actions in a concise and to the point manner, and the actions should be specific and detailed.
        For the case description {case_description}. The current diagnosis is {diagnosis}. The previous verification is {verification}.
        Please output the actions in the following format:
        {{
            "actions": <str>
        }}
        Actions:
        
        """
        messages = [{"role": "user", "content": query}]
        return messages

    def _management_messages(self, case_description: str, diagnosis_list: List[str], action_list: List[str], overall_reasoning: str) -> str:
        query = f"""You are provided with a case description and multiple diagnoses. You task is to give the whole management plan considering all the diagnoses. 
        The management plan should be concise and to the point, one of two summary sentences is present at the beginning.
        For the case description {case_description}. 
        The current diagnoses are {diagnosis_list}. 
        The overall reasoning is {overall_reasoning}.
        The actions for each diagnosis are {action_list}.
        Please output the management plan in the following json format, with the key "management_plan":
        {{
            "management_plan": <str>
        }}
        Management Plan:
        
        """
        messages = [{"role": "user", "content": query}]
        return messages

    async def action_plan(self, case_description: str, diagnosis: str, verification: str) -> Tuple[str, List[str]]:
        messages = self._messages(case_description, diagnosis, verification)
        action_text = await call_model(messages, self.model, temperature=0.0, response_format="json", llm_provider=self.llm_provider)
        return action_text 

    async def management_plan(self, case_description: str, diagnosis_list: List[str], action_list: List[str], overall_reasoning: str) -> Tuple[str, List[str]]:
        messages = self._management_messages(case_description, diagnosis_list, action_list, overall_reasoning)
        management_text = await call_model(messages, self.model, temperature=0.0, response_format="json", llm_provider=self.llm_provider)
        return management_text["management_plan"]
