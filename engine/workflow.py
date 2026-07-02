import numpy as np
import json
import os
import time
import asyncio
from typing import Callable 
from collections import defaultdict
import multiprocessing as mp
from functools import partial
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from utils.utils import (
    get_diagnosis,
    gpt_parse,
    cluster_preds,
    second_diagnosis_agent,
    reasoning_agent,
    action_agent,
)
from agents import (
    ReasoningAgent,
    AggregateAgent,
    DiagnosisAgent,
    ActionAgent,
    WarningAgent,
    VerificationAgent,
)

_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "CASEREPORTS-ED-rewritten.json")
try:
    with open(_DATA_PATH, "r") as _f:
        data = json.load(_f)
except FileNotFoundError:
    data = []


class DiagnosisState(TypedDict):
    """State for the diagnosis workflow"""

    case_description: Annotated[str, "single"]  # Ensure only one value per step
    base_model: str
    api_key: str
    num_inference: int

    # Intermediate results
    parsed_predictions: List[str]
    clustered_predictions: List[str]
    final_predictions: List[str]
    warning_diagnosis: List[str]
    verification: List[str]
    verification_score: List[float]
    # Final outputs
    reasoning: List[str]
    overall_reasoning: str
    actions: List[str]
    management: str
    reference_sentences: List[str]

    # Control flow
    needs_more_predictions: bool
    iteration_count: int


class DiagnosisWorkflow:
    """LangGraph-based diagnosis workflow"""

    def __init__(
        self,
        base_model: str,
        api_key: str,
        num_inference: int = 10,
        add_reasoning: bool = False,
        add_references: bool = False,
        llm_provider: str = "openai",
        print_func: Callable[[str], None] = None,
    ):
        self.base_model = base_model
        self.api_key = api_key
        self.num_inference = num_inference
        self.add_reasoning = add_reasoning
        self.add_references = add_references
        self.llm_provider = llm_provider
        self.print_func = print_func if print_func is not None else print
        self.graph = self._build_graph()

    def _init_agents(self):
        self.reasoning_agent = ReasoningAgent(self.base_model, add_references=self.add_references, llm_provider=self.llm_provider)
        self.aggregate_agent = AggregateAgent(self.base_model, llm_provider=self.llm_provider)
        self.action_agent = ActionAgent(self.base_model, llm_provider=self.llm_provider)
        self.warning_agent = WarningAgent(self.base_model, llm_provider=self.llm_provider)
        self.verification_agent = VerificationAgent(self.base_model, llm_provider=self.llm_provider)
        self.diagnosis_agent = DiagnosisAgent(self.base_model, self.num_inference, llm_provider=self.llm_provider)

    def _build_graph(self) -> StateGraph:
        self._init_agents()
        """Build the LangGraph workflow"""
        workflow = StateGraph(DiagnosisState)

        # Add nodes
        workflow.add_node("initial_diagnosis", self._initial_diagnosis_node)
        workflow.add_node("warning_diagnosis", self._warning_diagnosis_node)
        workflow.add_node("cluster_predictions", self._cluster_predictions_node)
        workflow.add_node("check_predictions", self._check_predictions_node)
        workflow.add_node("generate_additional", self._generate_additional_node)
        workflow.add_node("verify_diagnosis", self._verify_diagnosis_node, defer=True)
        workflow.add_node("generate_reasoning", self._generate_reasoning_node)
        workflow.add_node("overall_reasoning", self._overall_reasoning_node)
        workflow.add_node("generate_actions", self._generate_actions_node)
        workflow.add_node("generate_management", self._generate_management_node, defer=True)

        # Set entry point
        # workflow.set_entry_point("initial_diagnosis")
        workflow.add_edge(START, "initial_diagnosis")
        workflow.add_edge(START, "warning_diagnosis")

        # Add edges
        workflow.add_edge("initial_diagnosis", "cluster_predictions")
        # workflow.add_edge("initial_diagnosis", "warning_diagnosis")
        workflow.add_edge("cluster_predictions", "check_predictions")

        # Conditional routing
        workflow.add_conditional_edges(
            "check_predictions",
            self._should_generate_more,
            {"generate_more": "generate_additional", "continue": "verify_diagnosis"},
        )
        workflow.add_edge("generate_additional", "check_predictions")

        # Both warning_diagnosis and check_predictions paths converge at wait_for_both
        workflow.add_edge("warning_diagnosis", "verify_diagnosis")
        workflow.add_edge("verify_diagnosis", "generate_reasoning")
        workflow.add_edge("generate_reasoning", "overall_reasoning")
        workflow.add_edge("overall_reasoning", "generate_management")

        workflow.add_edge("verify_diagnosis", "generate_actions")
        workflow.add_edge("generate_actions", "generate_management")
        workflow.add_edge("generate_management", END)

        # workflow.add_edge("generate_reasoning", END)
        # workflow.add_edge("generate_actions", END)

        return workflow.compile()

    async def _initial_diagnosis_node(self, state: DiagnosisState) -> DiagnosisState:
        """Generate initial diagnosis responses"""
        self.print_func("start initial diagnosis **************")

        responses = await self.diagnosis_agent.diagnose(
            state["case_description"], state["num_inference"]
        )
        try:
            update_state = {
                "parsed_predictions": [res["diagnosis"] for res in responses],
                "reasoning": [res["reasoning"] for res in responses],
            }
        except KeyError:
            print("KeyError in initial diagnosis")
            return state
        
        return update_state

    async def _warning_diagnosis_node(self, state: DiagnosisState) -> DiagnosisState:
        """Generate warning diagnosis"""
        self.print_func("generating warning diagnosis **************")

        warning_diagnosis = await self.warning_agent.diagnose(state["case_description"])
        print("warning diagnosis: ", warning_diagnosis)
        update_state = {
            "warning_diagnosis": [d["warning_diagnosis"] for d in warning_diagnosis]
        }
        # print(update_state["warning_diagnosis"])
        return update_state

    async def _cluster_predictions_node(self, state: DiagnosisState) -> DiagnosisState:
        """Cluster similar predictions"""
        self.print_func("Aggregating predictions **************")

        results = await self.aggregate_agent.aggregate(state["parsed_predictions"])
        # Group the same prediction and its corresponding reasoning, sorted by frequency (highest first)
        pred_group_map = {result["diagnosis"]: result["group"] for result in results}
        preds = state["parsed_predictions"]
        reasonings = state["reasoning"]
        group_reasoning_pairs = {}
        for pred, reason in zip(preds, reasonings):
            group = pred_group_map.get(pred, pred)
            if group not in group_reasoning_pairs:
                group_reasoning_pairs[group] = []
            group_reasoning_pairs[group].append(reason)

        freq = {group: len(reasons) for group, reasons in group_reasoning_pairs.items()}
        sorted_groups = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        # Choose the first reasoning per group for deduplicated final output (optional)
        grouped_predictions = []
        grouped_reasonings = []
        for group, _ in sorted_groups:
            grouped_predictions.append(group)
            grouped_reasonings.append(group_reasoning_pairs[group][0])

        update_state = {
            "clustered_predictions": grouped_predictions,
            "final_predictions": grouped_predictions.copy(),
            "reasoning": grouped_reasonings,
        }
        return update_state

    def _check_predictions_node(self, state: DiagnosisState) -> DiagnosisState:
        """Check if we have enough predictions"""
        self.print_func("checking predictions **************")

        needs_more = len(state["final_predictions"]) < 3
        update_state = {
            "needs_more_predictions": needs_more,
            "iteration_count": state.get("iteration_count", 0),
        }

        update_state["iteration_count"] += 1
        return update_state

    def _should_generate_more(self, state: DiagnosisState) -> str:
        """Determine if we need more predictions"""
        if (
            state["needs_more_predictions"] and state["iteration_count"] < 5
        ):  # Prevent infinite loops
            return "generate_more"
        return "continue"

    async def _generate_additional_node(self, state: DiagnosisState) -> DiagnosisState:
        """Generate additional predictions"""
        self.print_func("generating additional predictions **************")

        # Use the first 3 predictions as context for generating more
        context_preds = state["final_predictions"][:3]
        additional_pred = await self.diagnosis_agent.additional_diagnosis(
            state["case_description"],
            context_preds,
        )

        update_state = {
            "final_predictions": state["final_predictions"]
            + [additional_pred["diagnosis"]],
            "reasoning": state["reasoning"] + [additional_pred["reasoning"]],
        }
        return update_state

    def _wait_for_both_node(self, state: DiagnosisState) -> DiagnosisState:
        """Wait for both warning_diagnosis and check_predictions paths to complete"""
        self.print_func("waiting for both paths to complete **************")
        while True:
            if state["warning_diagnosis"] is not None and not state["needs_more_predictions"]:
                return {}
            else:
                time.sleep(1)

    async def _verify_diagnosis_node(self, state: DiagnosisState) -> DiagnosisState:
        """Generate verification"""
        self.print_func("generating verification **************")

        # Call the verification agent's rerank method
        results = await self.verification_agent.rerank(
            state["case_description"], state["final_predictions"], state["reasoning"]
        )

        # Sort results by verification_score (already sorted in agent, but re-check if needed)
        sorted_results = sorted(
            results, key=lambda x: x["verification_score"], reverse=True
        )

        # Update predictions and reasoning to be sorted by verification_score
        sorted_predictions = [item["diagnosis"] for item in sorted_results]

        # Align reasonings according to new sorted_predictions order
        # Build map from diagnosis to reasoning
        diagnosis_to_reasoning = {
            d: r for d, r in zip(state["final_predictions"], state["reasoning"])
        }
        sorted_reasoning = [diagnosis_to_reasoning.get(d, "") for d in sorted_predictions]

        update_state = {
            "final_predictions": sorted_predictions,
            "reasoning": sorted_reasoning,
            "verification_score": [item["verification_score"] for item in sorted_results],
        }

        # update_state = {
        #     "verification": [d["verification_reason"] for d in results],
        #     "verification_score": [d["verification_score"] for d in results],
        # }
        return update_state


    async def _generate_reasoning_node(self, state: DiagnosisState) -> DiagnosisState:
        """Generate reasoning for each prediction"""
        self.print_func("generating reasoning **************")

        # reasoning_list = []
        # for pred in state["final_predictions"]:
        #     reasoning_text = await self.reasoning_agent.reason(
        #         state["case_description"],
        #         pred,
        #     )
        #     reasoning_list.append(reasoning_text)
            # time.sleep(2) ## wait for pubmed to search
        reasoning_text = [
            self.reasoning_agent.reason(
                state["case_description"],
                pred,
            )
            for pred in state["final_predictions"]
        ]
        reasoning_list = await asyncio.gather(*reasoning_text)
        # print(reasoning_text)

        # reasoning_list = [res for res in reasoning_text]
        try:
            update_state = {"reasoning": reasoning_list}
        except KeyError:
            print("KeyError in reasoning")
            return state

        # update_state = {"reasoning": reasoning_list}
        return update_state

    async def _overall_reasoning_node(self, state: DiagnosisState) -> DiagnosisState:
        """Generate overall reasoning"""
        self.print_func("generating overall reasoning **************")

        overall_reasoning = await self.reasoning_agent.reasoning_all(
            state["case_description"],
            state["final_predictions"],
            state["reasoning"],
            state["warning_diagnosis"],
            state["verification"],
        )
        update_state = {"overall_reasoning": overall_reasoning}

        return update_state

    async def _generate_actions_node(self, state: DiagnosisState) -> DiagnosisState:
        """Generate action plans for each prediction"""
        self.print_func("generating actions **************")

        diagnosis_list = state["final_predictions"]
        action_futures = [
            self.action_agent.action_plan(
                state["case_description"],
                diagnosis,
                state["reasoning"]
                # state["verification"][diagnosis_list.index(diagnosis)],
            )
            for diagnosis in diagnosis_list
        ]
        action_results = await asyncio.gather(*action_futures)
        try:
            actions_list = [res["actions"] for res in action_results]
        except KeyError:
            print("KeyError in actions")
            return state
        update_state = {"actions": actions_list}
        return update_state

    async def _generate_management_node(self, state: DiagnosisState) -> DiagnosisState:
        """Generate management plan"""
        self.print_func("generating management plan **************")

        if state["overall_reasoning"] == "":
            return {}

        management_text = await self.action_agent.management_plan(
            state["case_description"],
            state["final_predictions"],
            state["actions"],
            state["overall_reasoning"],
        )
        update_state = {"management": management_text}
        return update_state

    async def diagnose(self, case_description: str) -> Dict[str, Any]:
        """Run the complete diagnosis workflow"""
        initial_state = DiagnosisState(
            case_description=case_description,
            base_model=self.base_model,
            api_key=self.api_key,
            num_inference=self.num_inference,
            parsed_predictions=[],
            clustered_predictions=[],
            final_predictions=[],
            warning_diagnosis=[],
            verification=[],
            verification_score=[],
            reasoning=[],
            overall_reasoning="",
            actions=[],
            management="",
            reference_sentences=[],
            needs_more_predictions=False,
            iteration_count=0,
        )

        final_state = await self.graph.ainvoke(initial_state, recursion_limit=100)

        return {
            "case_description": case_description,
            "predictions": final_state["final_predictions"],
            "warning_diagnosis": final_state.get("warning_diagnosis", []),
            "reasoning": final_state.get("reasoning", []),
            "overall_reasoning": final_state.get("overall_reasoning", ""),
            "management": final_state.get("management", ""),
            'actions': final_state["actions"],
            # 'reference_sentences': final_state["reference_sentences"]
        }


def process_single_case_langgraph(
    case_info, base_model, api_key, num_inference, output_dir, llm_provider=None
):
    """
    Process a single case using LangGraph and save the result to a file.

    Args:
        case_info: tuple of (idx, case)
        base_model: model name
        api_key: API key
        num_inference: number of inference runs
        output_dir: directory to save individual results
        llm_provider: override for model provider (str)

    Returns:
        tuple of (idx, prediction) or None if already processed
    """
    idx, case = case_info
    output_file = os.path.join(output_dir, f"pred_{idx:04d}.json")

    # Check if already processed
    if os.path.exists(output_file):
        print(f"Case {idx} already processed, skipping...")
        try:
            with open(output_file, "r") as f:
                result = json.load(f)
            return idx, result["prediction"]
        except:
            print(f"Error reading existing file for case {idx}, reprocessing...")

    try:
        print("*" * 100)
        print(f"diagnosing {idx} / {len(data)}")
        query = case["case"]
        gold = case["answer"]

        # Detect model provider issue and give a helpful error
        # If model name is not recognized, suggest specifying llm_provider
        try:
            workflow = DiagnosisWorkflow(base_model, api_key, num_inference, llm_provider=llm_provider)
        except ValueError as err:
            print(f"Error diagnosing case {idx}: {err}")
            # Save error info
            error_result = {
                "idx": idx,
                "error": str(err),
                "query": case["case"],
                "gold": case["answer"],
            }
            with open(output_file, "w") as f:
                json.dump(error_result, f, indent=2)
            return None

        # Run diagnosis
        result = asyncio.run(workflow.diagnose(query))

        pred = result["predictions"]
        reasoning = result.get("reasoning", [])
        actions = result.get("actions", [])
        pred_str = ",".join(pred)

        # Save individual result
        result_data = {
            "idx": idx,
            "query": query,
            "gold": gold,
            "prediction": pred_str,
            "predictions_list": pred,
            "reasoning": reasoning,
            "actions": actions,
            "reference_sentences": result.get("reference_sentences", []),
        }

        with open(output_file, "w") as f:
            json.dump(result_data, f, indent=2)

        print(f"Saved prediction for case {idx}")
        return idx, pred_str

    except Exception as e:
        print(f"Error processing case {idx}: {e}")
        # Save error info
        error_result = {
            "idx": idx,
            "error": str(e),
            "query": case["case"],
            "gold": case["answer"],
        }
        with open(output_file, "w") as f:
            json.dump(error_result, f, indent=2)
        return None


def merge_predictions_langgraph(
    output_dir,
    total_cases,
    regenerate_missing=False,
    base_model=None,
    api_key=None,
    num_inference=None,
    data=None,
    llm_provider=None,
):
    """
    Merge all individual prediction files into a single array.

    Args:
        output_dir: directory containing individual prediction files
        total_cases: total number of cases
        regenerate_missing: whether to regenerate missing predictions
        base_model: model name for regeneration
        api_key: API key for regeneration
        num_inference: number of inference runs for regeneration
        data: original data for regeneration
        llm_provider: provider override

    Returns:
        dict with predictions, reasoning, and actions
    """
    preds = [None] * total_cases
    reasoning = [None] * total_cases
    actions = [None] * total_cases
    reference_sentences = [None] * total_cases

    for idx in range(total_cases):
        output_file = os.path.join(output_dir, f"pred_{idx:04d}.json")
        if os.path.exists(output_file):
            try:
                with open(output_file, "r") as f:
                    result = json.load(f)
                if "prediction" in result:
                    preds[idx] = result["prediction"]
                else:
                    print(f"Warning: No prediction found in file for case {idx}")
                if "reasoning" in result:
                    reasoning[idx] = result["reasoning"]
                if "actions" in result:
                    actions[idx] = result["actions"]
                if "reference_sentences" in result:
                    reference_sentences[idx] = result["reference_sentences"]
            except Exception as e:
                print(f"Error reading file for case {idx}: {e}")
        else:
            print(f"Warning: No file found for case {idx}")

    # Check for missing predictions
    missing_indices = [i for i, pred in enumerate(preds) if pred is None]
    if missing_indices:
        print(f"Warning: Missing predictions for cases: {missing_indices}")

        if regenerate_missing and base_model and api_key and num_inference and data:
            print(f"Regenerating {len(missing_indices)} missing predictions...")
            regenerate_missing_predictions_langgraph(
                missing_indices, output_dir, base_model, api_key, num_inference, data, llm_provider=llm_provider
            )

            # Re-read the regenerated predictions
            for idx in missing_indices:
                output_file = os.path.join(output_dir, f"pred_{idx:04d}.json")
                if os.path.exists(output_file):
                    try:
                        with open(output_file, "r") as f:
                            result = json.load(f)
                        if "prediction" in result:
                            preds[idx] = result["prediction"]
                        if "reasoning" in result:
                            reasoning[idx] = result["reasoning"]
                        if "actions" in result:
                            actions[idx] = result["actions"]
                        if "reference_sentences" in result:
                            reference_sentences[idx] = result["reference_sentences"]
                    except Exception as e:
                        print(f"Error reading regenerated file for case {idx}: {e}")

    results = {
        "predictions": preds,
        "reasoning": reasoning,
        "actions": actions,
        "reference_sentences": reference_sentences,
    }
    return results


def regenerate_missing_predictions_langgraph(
    missing_indices, output_dir, base_model, api_key, num_inference, data, llm_provider=None
):
    """
    Regenerate predictions for missing cases using LangGraph.

    Args:
        missing_indices: list of case indices to regenerate
        output_dir: directory to save results
        base_model: model name
        api_key: API key
        num_inference: number of inference runs
        data: original data
        llm_provider: model provider override
    """
    import multiprocessing as mp
    from functools import partial

    # Prepare missing case data for multiprocessing
    missing_case_data = [(idx, data[idx]) for idx in missing_indices]

    # Create partial function with fixed arguments
    process_func = partial(
        process_single_case_langgraph,
        base_model=base_model,
        api_key=api_key,
        num_inference=num_inference,
        output_dir=output_dir,
        llm_provider=llm_provider
    )

    # Use fewer processes for regeneration to avoid overwhelming the API
    num_processes = min(2, len(missing_indices))

    # Process missing cases using multiprocessing
    print(f"Regenerating with {num_processes} processes...")
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_func, missing_case_data)

    # Filter out None results (errors)
    valid_results = [r for r in results if r is not None]
    print(
        f"Successfully regenerated {len(valid_results)} out of {len(missing_indices)} missing cases"
    )


def main_langgraph():
    """Main function using LangGraph implementation"""
    # Configuration
    base_model = "gpt-4.1"
    api_key = os.environ.get("OPENAI_API_KEY", "")
    num_inference = 10
    num_processes = 4  # Adjust based on your system capabilities
    llm_provider = None  # Change to "openai" or "anthropic" as needed for custom/unknown models

    # Create output directory
    output_dir = "predictions-gpt-4.1-rewritten-langgraph"
    save_npy_path = "results/gpt-4.1-rewritten-preds-langgraph.npy"
    save_json_path = "results/gpt-4.1-rewritten-preds-langgraph.json"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare case data for multiprocessing
    case_data = [(idx, case) for idx, case in enumerate(data)]

    # Create partial function with fixed arguments
    process_func = partial(
        process_single_case_langgraph,
        base_model=base_model,
        api_key=api_key,
        num_inference=num_inference,
        output_dir=output_dir,
        llm_provider=llm_provider
    )

    # Process cases using multiprocessing
    print(f"Starting multiprocessing with {num_processes} processes...")
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_func, case_data)

    # Filter out None results (errors)
    valid_results = [r for r in results if r is not None]
    print(f"Successfully processed {len(valid_results)} out of {len(data)} cases")

    # Merge all predictions
    print("Merging predictions...")
    results = merge_predictions_langgraph(
        output_dir,
        len(data),
        regenerate_missing=True,
        base_model=base_model,
        api_key=api_key,
        num_inference=num_inference,
        data=data,
        llm_provider=llm_provider
    )

    # Save final merged predictions
    print("Saving merged predictions...")
    np.save(save_npy_path, results)

    # Also save as JSON for easier inspection
    with open(save_json_path, "w") as f:
        json.dump(
            {
                "predictions": results["predictions"],
                "reasoning": results["reasoning"],
                "actions": results["actions"],
                "reference_sentences": results["reference_sentences"],
                "total_cases": len(data),
                "successful_cases": len(valid_results),
            },
            f,
            indent=2,
        )

    print("Processing complete!")
    print(f"Total cases: {len(data)}")
    print(f"Successful predictions: {len(valid_results)}")
    print(f"Results saved to: {save_npy_path} and {save_json_path}")


if __name__ == "__main__":
    main_langgraph()
