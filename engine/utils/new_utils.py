import asyncio
import json
import os
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple

from .llm import call_model

# Hardcoded fast-model call sites below are redirected through this so
# deployments on non-OpenAI gateways (e.g. Azure AI Foundry) can be wired up
# by setting AEGISDX_FAST_MODEL (OPENDX_FAST_MODEL kept as legacy fallback).
_FAST_MODEL = (
    os.environ.get("AEGISDX_FAST_MODEL")
    or os.environ.get("OPENDX_FAST_MODEL")
    or "gpt-4o-mini"
)


async def get_diagnosis(prompt: str, model: str, api_key: str, num_inference: int = 1) -> str | List[str]:
    """
    Analyze patient case and provide diagnosis using the new LLM interface.
    
    Args:
        prompt: Patient case description
        model: Model name to use
        api_key: API key (not used in new implementation)
        num_inference: Number of inference runs
        
    Returns:
        Single diagnosis string or list of diagnosis strings
    """
    system_prompt = "Please analyze this patient's case and provide only one diagnosis result. The output format should strictly follow the format '<analysis> xxx </analysis> \n\n <diagnosis> xxx </diagnosis>'."
    full_prompt = system_prompt + prompt
    
    messages = [{"role": "user", "content": full_prompt}]
    
    if num_inference == 1:
        result = await call_model(messages, model, temperature=0.4)
        return result
    else:
        results = []
        for _ in range(num_inference):
            result = await call_model(messages, model, temperature=0.4)
            results.append(result)
        return results


async def cluster_preds(ranked_preds: List[str], rewards: List[float], api_key: str) -> Tuple[List[str], List[float]]:
    """
    Cluster similar predictions using LLM.
    
    Args:
        ranked_preds: List of predictions to cluster
        rewards: List of reward scores
        api_key: API key (not used in new implementation)
        
    Returns:
        Tuple of (clustered_predictions, rewards)
    """
    prompt = """Please analyze these medical diagnoses and cluster them into groups of similar conditions. I will provide a list of diagnoses, and you should group the diagnoses into different groups. 
    The output should be a dictionary in a json format. The key is the diagnosis and the value is the group name. Do not contain any other information and '\n' in the output.
    The group name should be a short description of the group. Strictly follow the format in the example. For instances, the diagnoses are: "A1", "B1", "A2", "A3", "A1", "C1", "B2". \n The output should be: {\n"A1": "A",\n "B1": "B",\n "A2": "A",\n "A3": "A",\n "A1": "A",\n "C1": "C",\n "B2": "B"\n}.
    """
    
    print('initial preds: ', ranked_preds)
    
    prompt += (
        'The provided diagnoses are: "'
        + '", "'.join(ranked_preds)
        + '"\n '
        + "The output should be: "
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        clustered_text = await call_model(messages, _FAST_MODEL, temperature=0.0)
        cluster_dict = json.loads(clustered_text)
        cluster_name = list(cluster_dict.values())
    except Exception as e:
        print(e)
        print(clustered_text)
        return ranked_preds, rewards
    
    clus_preds = defaultdict(lambda: 0.0)
    for reward, name in zip(rewards, cluster_name):
        clus_preds[name] += reward
    
    sort_preds = sorted(clus_preds.items(), key=lambda x: x[1], reverse=True)
    print('sorted preds: ', sort_preds)
    
    try:
        diag_preds = [x[0] for x in sort_preds]
        diag_rewards = [x[1] for x in sort_preds]
    except Exception as e:
        print(e)
        print(sort_preds)
        return ranked_preds, rewards
    
    return diag_preds, diag_rewards

async def search_pubmed(query: str, model: str, api_key: str) -> str:
    """
    Search PubMed for a query.
    
    Args:
        query: Query to search for
        api_key: API key (not used in new implementation)

    Returns:
        Search results from PubMed
    """
    prompt = f"""Your task is to search PubMed case reports for the given diagnosis. Return the top 5 case reports for the given diagnosis.
    The diagnosis is: {query}
    You must return ONLY a JSON array of 5 objects, each with the fields: 'id' (int), 'title' (str), 'abstract' (str), and 'url' (str). 
    The output format must be:

    [
    {{"id": 1, "title": "paper title 1", "abstract": "paper abstract 1", "url": "paper url 1"}},
    {{"id": 2, "title": "paper title 2", "abstract": "paper abstract 2", "url": "paper url 2"}},
    {{"id": 3, "title": "paper title 3", "abstract": "paper abstract 3", "url": "paper url 3"}},
    {{"id": 4, "title": "paper title 4", "abstract": "paper abstract 4", "url": "paper url 4"}},
    {{"id": 5, "title": "paper title 5", "abstract": "paper abstract 5", "url": "paper url 5"}}
    ]

    Do not include any explanation or text outside the JSON array.
    """
    
    messages = [{"role": "user", "content": prompt}]
    
    result = await call_model(messages, _FAST_MODEL, temperature=0.0)
    return result

async def gpt_parse(preds: str, api_key: str) -> str:
    """
    Extract diagnosis from description using LLM.
    
    Args:
        preds: Prediction description
        api_key: API key (not used in new implementation)
        
    Returns:
        Extracted diagnosis string
    """
    prompt = "Your task is to extract the diagnostic result from the provided description. Donot contain any other words, only give the diagnosis words. "
    new_prompt = prompt + preds
    
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": new_prompt}
    ]
    
    result = await call_model(messages, _FAST_MODEL, temperature=0.0)
    return result.strip()


async def parse_preds(preds: str, api_key: str) -> Dict[str, str]:
    """
    Extract all diagnostic results from description in JSON format.
    
    Args:
        preds: Prediction description
        api_key: API key (not used in new implementation)
        
    Returns:
        Dictionary of diagnoses and reasons
    """
    prompt = """Your task is to extract the all the diagnostic results from the provided description. return json format as the following example:
    {"diagnosis 1": "reason 1", "diagnosis 2": "reason 2", "diagnosis 3": "reason 3"}, if there is no diagnosis, return {}.
    if there is no reason, return {"diagnosis 1": "", "diagnosis 2": "", "diagnosis 3": ""}.
    The description is:
    """
    new_prompt = prompt + preds
    
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": new_prompt}
    ]
    
    try:
        result = await call_model(messages, _FAST_MODEL, temperature=0.0, response_format="json")
        return result
    except Exception as e:
        print(e)
        return {}


async def second_diagnosis_agent(
    case_description: str, 
    diagnosis_list: List[str], 
    api_key: str, 
    model_name: str = "gpt-4.1"
) -> str:
    """
    Generate additional diagnosis different from previous ones.
    
    Args:
        case_description: Patient case description
        diagnosis_list: List of previous diagnoses
        api_key: API key (not used in new implementation)
        model_name: Model to use
        
    Returns:
        Additional diagnosis string
    """
    diagnosis = ", ".join(diagnosis_list)
    prompt = f""" The case description is: {case_description}, the previous diagnoses are: {diagnosis}. 
    Please provide additional diagnosis for the case description. The diagnosis should be different from the previous diagnoses and should be related to the case description.
    The output format should strictly follow the format '<analysis> xxx </analysis> \n\n <diagnosis> xxx </diagnosis>'."""
    
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt}
    ]
    
    result = await call_model(messages, model_name, temperature=0.0)
    parsed_result = await gpt_parse(result, api_key)
    return parsed_result


async def critical_summary(response: str, api_key: str, model_name: str = "gpt-4.1") -> str:
    """
    Summarize response to include only significant issues and clues.
    
    Args:
        response: Response to summarize
        api_key: API key (not used in new implementation)
        model_name: Model to use
        
    Returns:
        Summarized response
    """
    prompt = f"""Please summarize the response to only include the following parts:
    1. Significant Issues in the Medical Decision
    2. Clues That May Not Be Fully Considered or Raise Additional Issues
    
    answer should be concise and to the point, no more than 150 words.
    The response is: {response}
    """
    
    messages = [{"role": "user", "content": prompt}]
    
    result = await call_model(messages, model_name, temperature=0.0)
    return result.strip()


async def critical_agent(
    case_description: str, 
    diagnosis: str, 
    api_key: str, 
    model_name: str = "gpt-4.1"
) -> str:
    """
    Check diagnosis for issues using critical analysis.
    
    Args:
        case_description: Patient case description
        diagnosis: Diagnosis to check
        api_key: API key (not used in new implementation)
        model_name: Model to use
        
    Returns:
        Critical analysis result
    """
    prompt = f"""Please make a rigorous check to the medical decision given the case description. The case description is: {case_description}, the medical decision is: {diagnosis}.
    Find clues in the case description that are not considered in the medical decision or the significant issues in the medical decision."""
    
    messages = [{"role": "user", "content": prompt}]
    
    result = await call_model(messages, model_name, temperature=0.0)
    summary = await critical_summary(result, api_key, model_name)
    return summary.strip()


def get_reference_sentences(sentences: List[str], reasoning: str) -> List[str]:
    """
    Extract reference sentences from reasoning text.
    
    Args:
        sentences: List of sentences
        reasoning: Reasoning text with references
        
    Returns:
        List of reference sentences
    """
    pattern = r"\|(.*?)\|"
    pattern_result = re.findall(pattern, reasoning)
    sentence_ids = []
    for p in pattern_result:
        sids = p.split(',')
        if type(sids[0]) is not int:
            continue
        sentence_ids.extend(p.split(','))
    sentence_ids = [int(i) for i in sentence_ids]
    sentence_ids = list(sorted(set(sentence_ids)))
    reference_sentences = [f"[{i}] {sentences[i-1]}" for i in sentence_ids]
    return reference_sentences

async def reasoning_agent_with_pubmed(
    case_description: str, 
    diagnosis: str, 
    api_key: str, 
    model_name: str = "gpt-4.1"
) -> Tuple[str, List[str]]:
    """
    Generate reasoning for diagnosis with reference sentences.
    
    Args:
        case_description: Patient case description
        diagnosis: Diagnosis to reason about
        api_key: API key (not used in new implementation)
        model_name: Model to use
        
    Returns:
        Tuple of (reasoning_text, reference_sentences)
    """
    sentences = re.split(r"(?<=[.!?])\s+", case_description)
    joined_sentences = "\n".join([f"[{idx+1}] {s}" for idx, s in enumerate(sentences)])

    prompt = f"""Based on the case description and diagnosis result, please provide a concise reasoning to the diagnosis result. Reasoning process should play like a doctor, and the result should be concise and to the point. Limit your response to 75 words (approximately 5 sentences). 
    Please provide a clear, concise answer that references specific sentence IDs when relevant. Each sentence should be on a new line with the cited case description sentence ID(s) enclosed in pipe symbols ("|") at the end followed by \n if relevant. Each case description sentence can be cited only once. Multiple sentences can be cited in one response sentence, joint by comma in the pipe symbol ("|3,5|"). 
    Case Description: {joined_sentences}
    Diagnosis: {diagnosis}
    
    Reasoning: """
    
    
    messages = [{"role": "user", "content": prompt}]
    
    result = await call_model(messages, model_name, temperature=0.0)
    reference_sentences = get_reference_sentences(sentences, result)
    return result, reference_sentences

async def reasoning_agent(
    case_description: str, 
    diagnosis: str, 
    api_key: str, 
    model_name: str = "gpt-4.1"
) -> Tuple[str, List[str]]:
    """
    Generate reasoning for diagnosis with reference sentences.
    
    Args:
        case_description: Patient case description
        diagnosis: Diagnosis to reason about
        api_key: API key (not used in new implementation)
        model_name: Model to use
        
    Returns:
        Tuple of (reasoning_text, reference_sentences)
    """
    sentences = re.split(r"(?<=[.!?])\s+", case_description)
    joined_sentences = "\n".join([f"[{idx+1}] {s}" for idx, s in enumerate(sentences)])

    prompt = f"""Based on the case description and diagnosis result, please provide a concise reasoning to the diagnosis result. Reasoning process should play like a doctor, and the result should be concise and to the point. Limit your response to 75 words (approximately 5 sentences). 
    Please provide a clear, concise answer that references specific sentence IDs when relevant. Each sentence should be on a new line with the cited case description sentence ID(s) enclosed in pipe symbols ("|") at the end followed by \n if relevant. Each case description sentence can be cited only once. Multiple sentences can be cited in one response sentence, joint by comma in the pipe symbol ("|3,5|"). 
    Case Description: {joined_sentences}
    Diagnosis: {diagnosis}
    
    Reasoning: """
    
    messages = [{"role": "user", "content": prompt}]
    
    result = await call_model(messages, model_name, temperature=0.0)
    reference_sentences = get_reference_sentences(sentences, result)
    return result, reference_sentences


async def action_agent(
    case_description: str, 
    diagnosis: str, 
    api_key: str, 
    model_name: str = "gpt-4.1"
) -> str:
    """
    Generate action/treatment plan for diagnosis.
    
    Args:
        case_description: Patient case description
        diagnosis: Diagnosis to create plan for
        api_key: API key (not used in new implementation)
        model_name: Model to use
        
    Returns:
        Action/treatment plan
    """
    sentences = re.split(r"(?<=[.!?])\s+", case_description)
    joined_sentences = "\n".join([f"[{idx+1}] {s}" for idx, s in enumerate(sentences)])

    prompt = f"""Based on the case description and diagnosis result, please provide a concise action/treatment plan for the diagnosis. The action/treatment plan should be concise and to the point. Limit your response to 75 words (approximately 5 sentences). 
    Case Description: {joined_sentences}
    Diagnosis: {diagnosis}
    Action/Treatment Plan: """
    
    messages = [{"role": "user", "content": prompt}]
    
    result = await call_model(messages, model_name, temperature=0.0)
    return result


async def test_diagnose(preds: List[str], true_diagnosis: str, api_key: str, max_k: int = 10) -> List[str]:
    """
    Test diagnosis predictions against true diagnosis.
    
    Args:
        preds: List of predicted diagnoses
        true_diagnosis: True diagnosis
        api_key: API key (not used in new implementation)
        max_k: Maximum number of results
        
    Returns:
        List of test results
    """
    prompt = """Your task is to identify whether the provided predicted differential diagnosis is correct based on the true diagnosis. Carefully review the information and determine the correctness of the prediction. Please notice same diagnosis might be in different words. Only return "Y" for yes or "N" for no, without any other words.
    """
    
    results = []
    for pred in preds:
        messages = [
            {"role": "system", "content": "pysician"},
            {
                "role": "user",
                "content": f"{prompt} \n"
                f"Predict Diagnosis: {pred}\n"
                f"True Differential Diagnosis: {true_diagnosis}",
            },
        ]
        
        answer = await call_model(messages, _FAST_MODEL, temperature=0.0)
        if answer == "Y":
            results += [answer] * (max_k - len(results))
            break
        else:
            results.append(answer)
    return results


async def analyze_agent(prompt: str, api_key: str) -> str:
    """
    Rewrite patient case in better format for diagnosis.
    
    Args:
        prompt: Original patient case
        api_key: API key (not used in new implementation)
        
    Returns:
        Rewritten case description
    """
    new_prompt = (
        "Please rewrite the patient case in a better format, let the rewrite prompt to be understood by the model for disease diagnosis. Do not provide extra information that is not related to the patient case. The original prompt is: "
        + prompt
    )
    
    messages = [
        {"role": "system", "content": "pysician"},
        {"role": "user", "content": new_prompt}
    ]
    
    result = await call_model(messages, _FAST_MODEL, temperature=0.0)
    return result


async def diagnose_and_verify(official_symptoms: str, diagnosis_reason: str, api_key: str) -> float:
    """
    Match official diagnosis with diagnosis reason and return score.
    
    Args:
        official_symptoms: Official symptoms of diagnosis
        diagnosis_reason: Reason for diagnosis
        api_key: API key (not used in new implementation)
        
    Returns:
        Score between 0 and 1
    """
    prompt = f"Please match the diagnosis reason with the official symptoms and return the score as a number between 0 and 1. best score is 1, worst score is 0. only return the score, no other words. The official symptoms are: {official_symptoms}, the diagnosis reason is: {diagnosis_reason}. "
    
    messages = [{"role": "user", "content": prompt}]
    
    result = await call_model(messages, _FAST_MODEL, temperature=0.0)
    return float(result)


async def symptoms_agent(diagnosis: str, official_results: str, api_key: str) -> str:
    """
    Extract official symptoms from official results.
    
    Args:
        diagnosis: Diagnosis to extract symptoms for
        official_results: Official medical results
        api_key: API key (not used in new implementation)
        
    Returns:
        Extracted symptoms
    """
    prompt = f"This is the conetent related to diagnosis: {diagnosis}, {official_results}. Please extract the official symptoms from the input and return the official symptoms in a good format. These symptom will be used to verify the diagnosis reason."
    
    messages = [{"role": "user", "content": prompt}]
    
    result = await call_model(messages, _FAST_MODEL, temperature=0.0)
    return result


async def verify_diagnosis(
    diagnosis: str, 
    diagnosis_reason: str, 
    api_key: str, 
    max_results: int = 1
) -> float:
    """
    Verify diagnosis by searching online and matching with reason.
    
    Args:
        diagnosis: Diagnosis to verify
        diagnosis_reason: Reason for diagnosis
        api_key: API key (not used in new implementation)
        max_results: Maximum search results
        
    Returns:
        Verification score
    """
    # Note: This function requires the verify_agent module which may not be available
    # For now, we'll implement a simplified version
    try:
        from verify_agent import search_google
        
        official_results = search_google(
            query=f"{diagnosis} medical condition", num_results=max_results
        )
        official_results = official_results["results"][0]["body"]
        
        official_symptoms = await symptoms_agent(diagnosis, official_results, api_key)
        results = await diagnose_and_verify(
            official_symptoms=official_symptoms,
            diagnosis_reason=diagnosis_reason,
            api_key=api_key,
        )
        
        return results
    except ImportError:
        print("Warning: verify_agent module not available. Skipping verification.")
        return 0.5  # Return neutral score if verification not available


# Synchronous wrapper functions for backward compatibility
def get_diagnosis_sync(prompt: str, model: str, api_key: str, num_inference: int = 1) -> str | List[str]:
    """Synchronous wrapper for get_diagnosis."""
    try:
        # Check if we're in an event loop
        loop = asyncio.get_running_loop()
        # If we are, we need to run in a new thread
        import concurrent.futures
        import threading
        
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(get_diagnosis(prompt, model, api_key, num_inference))
            finally:
                new_loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()
    except RuntimeError:
        # No event loop running, safe to use asyncio.run
        return asyncio.run(get_diagnosis(prompt, model, api_key, num_inference))


def cluster_preds_sync(ranked_preds: List[str], rewards: List[float], api_key: str) -> Tuple[List[str], List[float]]:
    """Synchronous wrapper for cluster_preds."""
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(cluster_preds(ranked_preds, rewards, api_key))
            finally:
                new_loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()
    except RuntimeError:
        return asyncio.run(cluster_preds(ranked_preds, rewards, api_key))


def gpt_parse_sync(preds: str, api_key: str) -> str:
    """Synchronous wrapper for gpt_parse."""
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(gpt_parse(preds, api_key))
            finally:
                new_loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()
    except RuntimeError:
        return asyncio.run(gpt_parse(preds, api_key))


def parse_preds_sync(preds: str, api_key: str) -> Dict[str, str]:
    """Synchronous wrapper for parse_preds."""
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(parse_preds(preds, api_key))
            finally:
                new_loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()
    except RuntimeError:
        return asyncio.run(parse_preds(preds, api_key))


def second_diagnosis_agent_sync(
    case_description: str, 
    diagnosis_list: List[str], 
    api_key: str, 
    model_name: str = "gpt-4.1"
) -> str:
    """Synchronous wrapper for second_diagnosis_agent."""
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(second_diagnosis_agent(case_description, diagnosis_list, api_key, model_name))
            finally:
                new_loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()
    except RuntimeError:
        return asyncio.run(second_diagnosis_agent(case_description, diagnosis_list, api_key, model_name))


def critical_agent_sync(
    case_description: str, 
    diagnosis: str, 
    api_key: str, 
    model_name: str = "gpt-4.1"
) -> str:
    """Synchronous wrapper for critical_agent."""
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(critical_agent(case_description, diagnosis, api_key, model_name))
            finally:
                new_loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()
    except RuntimeError:
        return asyncio.run(critical_agent(case_description, diagnosis, api_key, model_name))


def reasoning_agent_sync(
    case_description: str, 
    diagnosis: str, 
    api_key: str, 
    model_name: str = "gpt-4.1"
) -> Tuple[str, List[str]]:
    """Synchronous wrapper for reasoning_agent."""
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(reasoning_agent(case_description, diagnosis, api_key, model_name))
            finally:
                new_loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()
    except RuntimeError:
        return asyncio.run(reasoning_agent(case_description, diagnosis, api_key, model_name))


def action_agent_sync(
    case_description: str, 
    diagnosis: str, 
    api_key: str, 
    model_name: str = "gpt-4.1"
) -> str:
    """Synchronous wrapper for action_agent."""
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(action_agent(case_description, diagnosis, api_key, model_name))
            finally:
                new_loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()
    except RuntimeError:
        return asyncio.run(action_agent(case_description, diagnosis, api_key, model_name))
