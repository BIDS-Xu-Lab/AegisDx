"""Orchestrate the multi-agent diagnostic reasoning pipeline.

The pipeline is an async generator that yields stage events as it runs and
emits a single final `result` event matching the schema OpenDX persists.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from .agents import ActionAgent, AggregateAgent, DiagnosisAgent, ReasoningAgent, WarningAgent


def _group_and_rank(
    parsed_predictions: list[str],
    parsed_reasonings: list[str],
    cluster_results: list[dict[str, str]],
) -> tuple[list[str], list[str]]:
    """Map each raw diagnosis to its cluster, then rank clusters by vote count.

    Returns `(ranked_groups, one_reasoning_per_group)`.
    """
    pred_to_group = {item["diagnosis"]: item["group"] for item in cluster_results}
    group_reasonings: dict[str, list[str]] = {}
    for pred, reason in zip(parsed_predictions, parsed_reasonings):
        group = pred_to_group.get(pred, pred)
        group_reasonings.setdefault(group, []).append(reason)
    ranked = sorted(group_reasonings.items(), key=lambda kv: len(kv[1]), reverse=True)
    return [g for g, _ in ranked], [reasons[0] for _, reasons in ranked]


async def run_pipeline(
    case_description: str,
    *,
    llm_model: str,
    aggregate_model: str = "gpt-4o-mini",
    llm_provider: str = "openai",
    num_inference: int = 10,
    add_references: bool = True,
) -> AsyncIterator[dict[str, Any]]:
    """Run the multi-agent diagnosis pipeline.

    Yields events compatible with OpenDX's SSE forwarder:
        {"type": "progress", "message": "<stage>"}
        {"type": "result",   "data":    {...payload...}}
        {"type": "error",    "message": "<reason>"}
    """
    try:
        diagnosis_agent = DiagnosisAgent(llm_model, num_inference=num_inference, llm_provider=llm_provider)
        aggregate_agent = AggregateAgent(model=aggregate_model, llm_provider=llm_provider)
        warning_agent = WarningAgent(llm_model, llm_provider=llm_provider)
        reasoning_agent = ReasoningAgent(llm_model, add_references=add_references, llm_provider=llm_provider)
        action_agent = ActionAgent(llm_model, llm_provider=llm_provider)

        # 1. Sample candidate diagnoses + flag must-not-miss differentials in parallel.
        yield {"type": "progress", "message": "Generating candidate diagnoses..."}
        candidates_task = asyncio.create_task(diagnosis_agent.diagnose(case_description))
        warning_task = asyncio.create_task(warning_agent.diagnose(case_description))
        candidates = await candidates_task
        parsed_predictions = [c["diagnosis"] for c in candidates]
        parsed_reasonings = [c.get("reasoning", "") for c in candidates]

        # 2. Cluster equivalents and rank by vote.
        yield {"type": "progress", "message": "Aggregating predictions..."}
        cluster_results = await aggregate_agent.aggregate(parsed_predictions)
        ranked_predictions, ranked_reasonings = _group_and_rank(
            parsed_predictions, parsed_reasonings, cluster_results
        )

        # Resolve the warning task we kicked off in parallel with diagnosis.
        warning_items = await warning_task
        warning_diagnoses = [w.get("warning_diagnosis", "") for w in warning_items if w.get("warning_diagnosis")]

        # 3. Per-diagnosis rationale (optionally PubMed-grounded).
        yield {"type": "progress", "message": "Generating per-diagnosis reasoning..."}
        reasoning_payload = await asyncio.gather(
            *(reasoning_agent.reason(case_description, d) for d in ranked_predictions)
        )

        # 4. Per-diagnosis actions + overall synthesis in parallel.
        yield {"type": "progress", "message": "Generating actions and management plan..."}
        actions_task = asyncio.create_task(
            asyncio.gather(*(action_agent.action_plan(case_description, d) for d in ranked_predictions))
        )
        overall_reasoning_task = asyncio.create_task(
            reasoning_agent.reasoning_all(
                case_description,
                ranked_predictions,
                [r.get("reasoning", "") for r in reasoning_payload],
                warning_diagnoses,
            )
        )
        actions, overall_reasoning = await asyncio.gather(actions_task, overall_reasoning_task)

        management = await action_agent.management_plan(
            case_description, ranked_predictions, actions, overall_reasoning
        )

        yield {
            "type": "result",
            "data": {
                "case_description": case_description,
                "predictions": ranked_predictions,
                "warning_diagnosis": warning_diagnoses,
                "reasoning": reasoning_payload,
                "overall_reasoning": overall_reasoning,
                "management": management,
                "actions": actions,
            },
        }

    except Exception as e:  # noqa: BLE001 — surface every failure to the client.
        yield {"type": "error", "message": f"Pipeline failure: {e!s}"}
