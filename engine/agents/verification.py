from retrievers import PubMedRetriever
from utils.new_utils import reasoning_agent_with_pubmed
from typing import Tuple, List
from utils.new_utils import call_model


class VerificationAgent:
    """
    Verify sufficien evidence for the diagnosis

    
    """
    def __init__(self, model: str, llm_provider: str = "openai"):
        self.model = model
        self.llm_provider = llm_provider

    def _messages(self, case_description: str, diagnosis: str) -> str:
        query = f"""You are provided with a case description and a diagnosis. You task is to measure the confidence of the diagnosis for the case description.
        You must return ONLY a JSON object, with keys: 'verification_score' (float), 'verification_reason' (str).
        The verification reason should indicate what evidences support the diagnosis and what extra evidences are needed to support the diagnosis.
        The verification score should be a number between 0 and 1, where 1 means the diagnosis is sufficiently supported by the case description, and 0 means the diagnosis is totally wrong.
        For the case description {case_description}. The current diagnosis is {diagnosis}.
        Please output the verification score and reason in the following format:
        {{
            "verification_reason": <str>,
            "verification_score": <float>
        }}
        Verification:
        """
        messages = [{"role": "user", "content": query}]
        return messages
    
    def _rerank_messages(self, case_description: str, diagnosis_list: List[str], reasoning_list: List[str]) -> str:
        diag_reason = [f"diagnosis: {diagnosis}, reasoning: {reasoning}" for diagnosis, reasoning in zip(diagnosis_list, reasoning_list)]
        query = f"""You are provided with a case description and a list of diagnoses. You task is to verify the diagnoses based on the confidence of the diagnosis for the case description.
        The case description is: {case_description}. The diagnoses and reasonings are: {diag_reason}.
        The verification score should be a number between 0 and 1, where 1 means the diagnosis is sufficiently supported by the case description, and 0 means the diagnosis is totally wrong.
        Please output the verified diagnoses in the following format:
        [{{"diagnosis": diagnosis1, "verification_score": score1}}, {{"diagnosis": diagnosis2, "verification_score": score2}}, {{"diagnosis": diagnosis3, "verification_score": score3}}, ...].
        """
        messages = [{"role": "user", "content": query}]
        return messages 

    async def verify(self, case_description: str, diagnosis: str) -> Tuple[str, List[str]]:
        messages = self._messages(case_description, diagnosis)
        verification_text = await call_model(messages, self.model, temperature=0, response_format="json", llm_provider=self.llm_provider)
        return verification_text 
    
    async def rerank(self, case_description: str, diagnosis_list: List[str], reasoning_list: List[str]) -> List[Tuple[str, float]]:
        messages = self._rerank_messages(case_description, diagnosis_list, reasoning_list)
        rerank_text = await call_model(messages, self.model, temperature=0, response_format="json", llm_provider=self.llm_provider)
        # Rank the diagnoses in descending order by verification_score
        rerank_list = sorted(
            (
                {"diagnosis": item["diagnosis"], "verification_score": item["verification_score"]}
                for item in rerank_text
            ),
            key=lambda x: x["verification_score"],
            reverse=True,
        )
        return rerank_list 
