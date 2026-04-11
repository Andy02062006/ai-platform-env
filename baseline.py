#!/usr/bin/env python3
"""
baseline.py — Deterministic rule-based agent for AIPlatformEnv benchmark.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from env import AIPlatformEnv
from models import Action, Observation, Response, Reward
from tasks import GRADERS

SEED: int = 42

TASK_QUERIES: Dict[str, List[str]] = {
    "easy": [
        "What is the capital of France?",
    ],
    "medium": [
        "What were the main causes of the French Revolution?",
        "Summarise the key events of the French Revolution.",
        "What were the consequences and lasting effects of the French Revolution?",
    ],
    "hard": [
        "Write a Python function called binary_search that takes a sorted list and a target value.",
        "Debug and improve the binary_search function to handle edge cases.",
        "Add a recursive version of binary_search and compare with the iterative approach.",
        "What is the time and space complexity of binary search?",
        "Optimise the binary_search implementation for production use.",
    ],
}

def _best_response_index(responses: List[Response]) -> int:
    if not responses:
        return 0
    return max(range(len(responses)), key=lambda i: responses[i].relevance)

def run_episode(
    env: AIPlatformEnv,
    difficulty: str,
) -> Tuple[float, List[Action]]:
    env.reset(difficulty, seed=SEED)
    actions: List[Action] = []
    
    # 1. Plan (Medium/Hard)
    if difficulty in ("medium", "hard"):
        plan_action = Action(type="plan_task", plan=f"Workflow for {difficulty}: query -> compare -> refine.")
        obs, reward, *_ = env.step(plan_action)
        actions.append(plan_action)

    # 2. Query
    query_text = TASK_QUERIES[difficulty][0]
    submit_action = Action(type="submit_query", query=query_text)
    obs, reward, *_ = env.step(submit_action)
    actions.append(submit_action)

    # 3. Compare & Refine (Medium/Hard)
    if difficulty in ("medium", "hard"):
        compare_action = Action(type="compare_responses")
        obs, reward, *_ = env.step(compare_action)
        actions.append(compare_action)

        refine_action = Action(type="refine_query", query=f"Expert detail on: {query_text}")
        obs, reward, *_ = env.step(refine_action)
        actions.append(refine_action)

    # 4. Summarize (Hard)
    if difficulty == "hard":
        sum_action = Action(type="summarize")
        obs, reward, *_ = env.step(sum_action)
        actions.append(sum_action)

    # 5. Rate & Select
    if obs.responses:
        best_idx = _best_response_index(obs.responses)
        best_relevance = obs.responses[best_idx].relevance

        if difficulty == "easy":
            # For easy tasks, rate BEFORE selecting
            rate_action = Action(type="rate_response", score=best_relevance)
            obs, reward, *_ = env.step(rate_action)
            actions.append(rate_action)
        else:
            rate_action = Action(type="rate_response", score=best_relevance)
            obs, reward, *_ = env.step(rate_action)
            actions.append(rate_action)

        select_action = Action(type="select_response", selected_index=best_idx)
        obs, reward, *_ = env.step(select_action)
        actions.append(select_action)

    env_state = env.state()
    grader = GRADERS[difficulty]
    score = grader(actions, env_state)
    return score, actions

def main() -> None:
    env = AIPlatformEnv(seed=SEED)
    difficulties = ["easy", "medium", "hard"]
    scores: Dict[str, float] = {}

    for difficulty in difficulties:
        score, _actions = run_episode(env, difficulty)
        scores[difficulty] = score

    overall = sum(scores.values()) / len(scores)

    print("=" * 58)
    print("AIPlatformEnv — Baseline Agent")
    print("=" * 58)
    for difficulty in difficulties:
        label = difficulty.upper()
        print(f"  {label:8s} score = {scores[difficulty]:.4f}")
    print("-" * 58)
    print(f"  {'OVERALL':8s} score = {overall:.4f}")
    print("=" * 58)

if __name__ == "__main__":
    main()