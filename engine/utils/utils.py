import re

# import deepspeed
from openai import OpenAI
import json
from collections import defaultdict

# Import the verify_agent module
from .verify_agent import search_google


def get_diagnosis(prompt, model, API, num_inference=1):
    client = OpenAI(api_key = API)
    system_prompt = "Please analyze this patient's case and provide only one diagnosis result. The output format should strictly follow the format '<analysis> xxx </analysis> \n\n <diagnosis> xxx </diagnosis>'."
    prompt = system_prompt + prompt
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }

    ],
    model=model,
    temperature=0.4,
    n=num_inference,
    )
    if num_inference == 1:
        result=chat_completion.choices[0].message.content
    else:
        result = [choice.message.content for choice in chat_completion.choices]
        # for s in result:
        #     print('diag: ', s)
    return result

def cluster_preds(ranked_preds, rewards, API_KEY):
    # use large language model to cluster the preds into different groups and add the rewards to the preds.
    # the rewards are the rewards of the preds
    client = OpenAI(api_key=API_KEY)

    # Create prompt to cluster predictions
    prompt = """Please analyze these medical diagnoses and cluster them into groups of similar conditions. I will provide a list of diagnoses, and you should group the diagnoses into different groups. 
    The output should be a dictionary in a json format. The key is the diagnosis and the value is the group name. Do not contain any other information and '\n' in the output.
    The group name should be a short description of the group. Strictly follow the format in the example. For instances, the diagnoses are: "A1", "B1", "A2", "A3", "A1", "C1", "B2". \n The output should be: {\n"A1": "A",\n "B1": "B",\n "A2": "A",\n "A3": "A",\n "A1": "A",\n "C1": "C",\n "B2": "B"\n}.
    """

    # Add each prediction and its reward score to prompt
    print('initial preds: ', ranked_preds)

    prompt += (
        'The provided diagnoses are: "'
        + '", "'.join(ranked_preds)
        + '"\n '
        + "The output should be: "
    )
    # Get clustering response from GPT-4
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )

    # Parse response to get clustered predictions
    clustered_text = response.choices[0].message.content
    try:
        cluster_dict = json.loads(clustered_text)
        cluster_name = list(cluster_dict.values())
        # assert len(cluster_name) == len(ranked_preds), 'number of cluster name is not equal to the number of ranked preds'
    except Exception as e:
        print(e)
        # cluster_name = [a.split(':')[1].strip() for a in clustered_text.split(',')]
        print(clustered_text)
        return ranked_preds, rewards
    clus_preds = defaultdict(lambda: 0.0)
    for reward, name in zip(rewards, cluster_name):
        # clus_preds[name] += 1+reward
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


def gpt_parse(preds, API_KEY):
    client = OpenAI(api_key=API_KEY)
    # prompt = 'Your task is to extract the diagnostic result from the provided description. Only give the concept words.'
    prompt = "Your task is to extract the diagnostic result from the provided description. Donot contain any other words, only give the diagnosis words. "
    new_prompt = prompt + preds
    # prompt = 'Your task is to extract the diagnostic result from the provided description in the Input: . Only return the concept words in the Output: .'
    # prompt = 'Your task is to extract the diagnostic result from the provided description in the Input: . Only return one result in the Output: .'
    # new_prompt = prompt + f"Input: {preds} Output:"
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": new_prompt},
        ],
    )
    result = response.choices[0].message.content
    return result.strip()

def parse_preds(preds, API_KEY):
    client = OpenAI(api_key=API_KEY)
    # prompt = 'Your task is to extract the diagnostic result from the provided description. Only give the concept words.'
    prompt = """Your task is to extract the all the diagnostic results from the provided description. return json format as the following example:
    {"diagnosis 1": "reason 1", "diagnosis 2": "reason 2", "diagnosis 3": "reason 3"}, if there is no diagnosis, return {}.
    if there is no reason, return {"diagnosis 1": "", "diagnosis 2": "", "diagnosis 3": ""}.
    The description is:
    """
    new_prompt = prompt + preds
    # prompt = 'Your task is to extract the diagnostic result from the provided description in the Input: . Only return the concept words in the Output: .'
    # prompt = 'Your task is to extract the diagnostic result from the provided description in the Input: . Only return one result in the Output: .'
    # new_prompt = prompt + f"Input: {preds} Output:"
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": new_prompt},
        ],
    )
    result = response.choices[0].message.content
    try:
        result = json.loads(result)
    except Exception as e:
        print(e)
        return {}
    return result

def second_diagnosis_agent(
    case_description: str, diagnosis_list: list, API_KEY: str, model_name: str = "gpt-4.1"
):
    """
    Use the second diagnosis agent to get the second diagnosis from the case description and to exclude the diagnosis.
    """
    diagnosis = ", ".join(diagnosis_list)
    prompt = f""" The case description is: {case_description}, the previous diagnoses are: {diagnosis}. 
    Please provide additional diagnosis for the case description. The diagnosis should be different from the previous diagnoses and should be related to the case description.
    The output format should strictly follow the format '<analysis> xxx </analysis> \n\n <diagnosis> xxx </diagnosis>'."""
    client = OpenAI(api_key=API_KEY)
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ],
    )
    result = response.choices[0].message.content
    result = gpt_parse(result, API_KEY)
    return result


def critical_summary(response: str, API_KEY: str, model_name: str = "gpt-4.1"):
    """
    suummarize the reponse to only include the following parts:
    1. Significant Issues in the Medical Decision
    2. Clues That May Not Be Fully Considered or Raise Additional Issues
    
    answer should be concise and to the point, no more than 150 words.
    
    Args:
        case_description (str): _description_
        API_KEY (str): _description_
        model_name (str, optional): _description_. Defaults to "gpt-4.1".
    """
    prompt = f"""Please summarize the response to only include the following parts:
    1. Significant Issues in the Medical Decision
    2. Clues That May Not Be Fully Considered or Raise Additional Issues
    
    answer should be concise and to the point, no more than 150 words.
    The response is: {response}
    """
    client = OpenAI(api_key=API_KEY)
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    result = response.choices[0].message.content
    return result.strip()


def critical_agent(
    case_description: str, diagnosis: str, API_KEY: str, model_name: str = "gpt-4.1"
):
    """
    Use the critical agent to check the diagnosis to see if anything is wrong with the diagnosis.
    """
    prompt = f"""Please make a rigorous check to the medical decision given the case description. The case description is: {case_description}, the medical decision is: {diagnosis}.
    Find clues in the case description that are not considered in the medical decision or the significant issues in the medical decision."""
    client = OpenAI(api_key=API_KEY)
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    result = response.choices[0].message.content
    result = critical_summary(result, API_KEY)
    return result.strip()


def get_reference_sentences(sentences, reasoning: str):
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
    # reference_sentences = "\n".join(reference_sentences)
    return reference_sentences

def reasoning_agent(
    case_description: str, diagnosis: str, API_KEY: str, model_name: str = "gpt-4.1"
):
    """
    Use the reasoning agent to reason about the case description and the diagnosis.
    """
    sentences = re.split(r"(?<=[.!?])\s+", case_description)
    joined_sentences = "\n".join([f"[{idx+1}] {s}" for idx, s in enumerate(sentences)])

    prompt = f"""Based on the case description and diagnosis result, please provide a concise reasoning to the diagnosis result. Reasoning process should play like a doctor, and the result should be concise and to the point. Limit your response to 75 words (approximately 5 sentences). 
    Please provide a clear, concise answer that references specific sentence IDs when relevant. Each sentence should be on a new line with the cited case description sentence ID(s) enclosed in pipe symbols ("|") at the end followed by \n if relevant. Each case description sentence can be cited only once. Multiple sentences can be cited in one response sentence, joint by comma in the pipe symbol ("|3,5|"). 
    Case Description: {joined_sentences}
    Diagnosis: {diagnosis}
    
    Reasoning: """
    client = OpenAI(api_key=API_KEY)

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    result = response.choices[0].message.content
    reference_sentences = get_reference_sentences(sentences, result)
    return result, reference_sentences

def action_agent(
    case_description: str, diagnosis: str, API_KEY: str, model_name: str = "gpt-4.1"
):
    """
    Use the action agent to give the action/treatment plan for the diagnosis based on the case description and the diagnosis.
    """
    sentences = re.split(r"(?<=[.!?])\s+", case_description)
    joined_sentences = "\n".join([f"[{idx+1}] {s}" for idx, s in enumerate(sentences)])

    prompt = f"""Based on the case description and diagnosis result, please provide a concise action/treatment plan for the diagnosis. The action/treatment plan should be concise and to the point. Limit your response to 75 words (approximately 5 sentences). 
    Case Description: {joined_sentences}
    Diagnosis: {diagnosis}
    Action/Treatment Plan: """
    client = OpenAI(api_key=API_KEY)

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    result = response.choices[0].message.content
    return result

def test_diagnose(preds, true_diagnosis, API_KEY: str, max_k=10):
    prompt = """Your task is to identify whether the provided predicted differential diagnosis is correct based on the true diagnosis. Carefully review the information and determine the correctness of the prediction. Please notice same diagnosis might be in different words. Only return "Y" for yes or "N" for no, without any other words.
    """
    client = OpenAI(api_key=API_KEY)
    results = []
    for pred in preds:
        chat_return = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=[
                {"role": "system", "content": "pysician"},
                {
                    "role": "user",
                    "content": f"{prompt} \n"
                    f"Predict Diagnosis: {pred}\n"
                    f"True Differential Diagnosis: {true_diagnosis}",
                },
            ],
        )
        answer = chat_return.choices[0].message.content
        if answer == "Y":
            results += [answer] * (max_k - len(results))
            break
        else:
            results.append(answer)
    return results


def analyze_agent(prompt, API_KEY: str):
    ### rewrite the patient case in a more specific way (good format), let the rewrite prompt to be understood by the model
    prompt = (
        "Please rewrite the patient case in a better format, let the rewrite prompt to be understood by the model for disease diagnosis. Do not provide extra information that is not related to the patient case. The original prompt is: "
        + prompt
    )
    client = OpenAI(api_key=API_KEY)
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "pysician"},
            {"role": "user", "content": prompt},
        ],
    )
    result = response.choices[0].message.content
    return result


def diagnose_and_verify(official_symptoms: str, diagnosis_reason: str, API_KEY: str):
    """
    match the official diagnosis with the diagnosis reason with GPT, if the diagnosis reason is not in the official symptoms, then the diagnosis is not correct, and the score is 0, otherwise the score is 1.
    Args:
        official_symptoms: The official symptoms of the diagnosis
        diagnosis_reason: The reason for the diagnosis

    Returns:
        Verification results with medical information
    """
    client = OpenAI(api_key=API_KEY)
    prompt = f"Please match the diagnosis reason with the official symptoms and return the score as a number between 0 and 1. best score is 1, worst score is 0. only return the score, no other words. The official symptoms are: {official_symptoms}, the diagnosis reason is: {diagnosis_reason}. "
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return float(response.choices[0].message.content)


def symptoms_agent(diagnosis: str, official_results: str, API_KEY: str) -> str:
    """
    Extract the official symptoms from the official results
    """
    client = OpenAI(api_key=API_KEY)
    prompt = f"This is the conetent related to diagnosis: {diagnosis}, {official_results}. Please extract the official symptoms from the input and return the official symptoms in a good format. These symptom will be used to verify the diagnosis reason."
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content


def verify_diagnosis(
    diagnosis: str, diagnosis_reason: str, API_KEY: str, max_results: int = 1
) -> str:
    """
    Verify a diagnosis by searching online medical sources first and then verify the diagnosis with the reason.

    Args:
        diagnosis: The diagnosis to verify
        diagnosis_reason: The reason for the diagnosis
        max_results: Maximum number of results to return per source

    Returns:
        Formatted string with verification results
    """
    official_results = search_google(
        query=f"{diagnosis} medical condition", num_results=max_results
    )
    official_results = official_results["results"][0]["body"]
    ### extract the official symptoms from the official results
    official_symptoms = symptoms_agent(diagnosis, official_results, API_KEY)
    results = diagnose_and_verify(
        official_symptoms=official_symptoms,
        diagnosis_reason=diagnosis_reason,
        API_KEY=API_KEY,
    )

    return results
