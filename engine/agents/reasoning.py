from retrievers import PubMedRetriever
from utils.new_utils import reasoning_agent_with_pubmed
from typing import Tuple, List
from utils.new_utils import call_model


class ReasoningAgent:
    def __init__(self, model: str, add_references: bool = False, llm_provider: str = "openai"):
        self.model = model
        self.pubmed_retriever = PubMedRetriever(model=model, llm_provider=llm_provider)
        self.add_references = add_references
        self.llm_provider = llm_provider

    def _reasoning_messages(self, case_description: str, diagnosis: str, diagnosis_context: str) -> str:
        query = f"""You are provided with a case description, diagnosis result, and diagnosis references from multiple papers. Your task is to craft a concise, physician-style rationale for the diagnosis. 
        Keep the explanation focused—no more than 100 words, or roughly 3–5 sentences.  
        Use the references to support the reasoning. Put the paper ID in the end of the reasoning sentence if relevant.
        You must put all the used reference papers in the references list.  
        References should be in the following format:
        [1. Paper 1 title, 2. Paper 2 title, ...]
        
        Here is the case description, diagnosis result, and diagnosis context from different papers:
        Case Description: {case_description}
        Diagnosis: {diagnosis}
        Diagnosis references: {diagnosis_context}

        You must return ONLY a JSON object, with the key "reasoning" and "references". The output format must be:
        {{
            "reasoning": <str>
            "references": <list>
        }}
        Reasoning:
        """
        messages = [{"role": "user", "content": query}]
        return messages
    
    def _reasoning_messages_without_references(self, case_description: str, diagnosis: str ) -> str:
        query = f"""You are provided with a case description, diagnosis result, and diagnosis references from multiple papers. Your task is to craft a concise, physician-style rationale for the diagnosis. 
        Keep the explanation focused—no more than 100 words, or roughly 3–5 sentences.  
        
        Here is the case description, diagnosis result, and diagnosis context from different papers:
        Case Description: {case_description}
        Diagnosis: {diagnosis}

        You must return ONLY a JSON object, with the key "reasoning". The output format must be:
        {{
            "reasoning": <str>
        }}
        Reasoning:
        """
        messages = [{"role": "user", "content": query}]
        return messages
    
    def _reasoning_all_messages(self, case_description: str, diagnosis_list: List[str], reasoning_list: List[str], warning_diagnosis_list: List[str], verification_list: List[str]) -> str:
        if self.add_references:
            query = f"""You are provided with a case description and multiple diagnoses. The diagnoses contain the possible diagnoses and the warning diagnoses that should not be missed according to patient's symptoms. Each diagnosis has a corresponding reasoning and verification.
            You task is to give the whole reasoning for all the diagnoses, considering the reasoning and verification for each diagnosis.
            The reasoning should be concise and to the point, one of two summary sentences is present at the beginning.
            For the case description {case_description}. The current diagnoses are {diagnosis_list}. 
            The corresponding reasonings are {reasoning_list}. 
            The corresponding verifications are {verification_list}. 
            The warning diagnoses are {warning_diagnosis_list}. 
            Include all the diagnosis_list and warning_diagnosis_list in the reasoning. Write the summary sentence of the reasoning at the beginning.
            Please include references based on the reasoning of each diagnosis. Put the paper ID in the end of the reasoning sentence if relevant. All the references should be present in the references list with ID in the beginning. Please cite the references in the following format:
            [1] Paper 1 title 
            [2] Paper 2 title 
            ...
            Please output the reasoning in the following json format, with the key "reasoning" and "references":
            {{
                "reasoning": <str>
                "references": <list>
            }}
            Reasoning:
            """
        else:
            query = f"""You are provided with a case description and multiple diagnoses. The diagnoses contain the possible diagnoses and the warning diagnoses that should not be missed according to patient's symptoms. Each diagnosis has a corresponding reasoning and verification.
            You task is to give the whole reasoning for all the diagnoses, considering the reasoning and verification for each diagnosis.
            The reasoning should be concise and to the point, one of two summary sentences is present at the beginning.
            For the case description {case_description}. The current diagnoses are {diagnosis_list}. 
            The corresponding reasonings are {reasoning_list}. 
            The corresponding verifications are {verification_list}. 
            The warning diagnoses are {warning_diagnosis_list}. 
            Include all the diagnosis_list and warning_diagnosis_list in the reasoning. Write the summary sentence of the reasoning at the beginning.
            Please output the reasoning in the following json format, with the key "reasoning":
            {{
                "reasoning": <str>
            }}
            Reasoning:
            """
        messages = [{"role": "user", "content": query}]
        return messages
    

    async def reason(self, case_description: str, diagnosis: str) -> Tuple[str, List[str]]:
        if self.add_references:
            print(f"searching pubmed result for {diagnosis}**************")
            pubmed_results = await self.pubmed_retriever.search(diagnosis)
            print(f"finished searching pubmed result for {diagnosis}**************")
            messages = self._reasoning_messages(case_description, diagnosis, pubmed_results)
        else:
            messages = self._reasoning_messages_without_references(case_description, diagnosis)
        print(f"reasoning for {diagnosis}**************")
        reasoning_text = await call_model(messages, self.model, temperature=0.0, response_format="json", llm_provider=self.llm_provider)
        return reasoning_text 
    
    async def reasoning_all(self, case_description: str, diagnosis_list: List[str], reasoning_list: List[str], warning_diagnosis_list: List[str], verification_list: List[str]) -> Tuple[str, List[str]]:
        messages = self._reasoning_all_messages(case_description, diagnosis_list, reasoning_list, warning_diagnosis_list, verification_list)
        reasoning_text = await call_model(messages, self.model, temperature=0.0, response_format="json", llm_provider=self.llm_provider)
        result = reasoning_text["reasoning"]
        refs = reasoning_text.get("references") or []
        if refs:
            result += "\n" + "References: \n" + "\n ".join(refs)
        return result
