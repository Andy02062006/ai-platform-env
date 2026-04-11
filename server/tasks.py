"""
tasks.py
--------
Grading functions for the AI platform reinforcement-learning environment.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Sequence, Tuple
from models import Action, Response

GraderFn = Callable[[Sequence[Action], Dict], float]

_EASY_WEIGHTS: Dict[str, float] = {
    "submitted_query":       0.20,
    "positive_total_reward": 0.20,
    "selected_best_response":0.30,
    "accurate_rating":       0.30,
}

_MEDIUM_WEIGHTS: Dict[str, float] = {
    "submitted_query":        0.10,
    "plan_task_used":         0.15,
    "multi_turn_reasoning":   0.20,
    "compare_responses_used": 0.15,
    "selected_best_response": 0.20,
    "accurate_rating":        0.10,
    "positive_total_reward":  0.10,
}

_HARD_WEIGHTS: Dict[str, float] = {
    "submitted_query":        0.10,
    "plan_task_used":         0.10,
    "refine_query_used":      0.15,
    "summarize_used":         0.15,
    "selected_best_response": 0.20,
    "accurate_rating":        0.10,
    "positive_total_reward":  0.10,
    "action_diversity":       0.10,
}

# ---------------------------------------------------------------------------
# Shared primitive criteria (return a float in [0, 1])
# ---------------------------------------------------------------------------

def _criterion_submitted_query(actions: Sequence[Action], _state: Dict) -> float:
    return 1.0 if any(a.type == "submit_query" for a in actions) else 0.0

def _criterion_positive_total_reward(_actions: Sequence[Action], state: Dict) -> float:
    total: float = state.get("total_reward", 0.0)
    difficulty = state.get("difficulty", "")
    divisor = 1.0 if difficulty == "easy" else 2.0
    return max(0.0, min(1.0, total / divisor))

def _criterion_selected_best_response(actions: Sequence[Action], state: Dict) -> float:
    responses: List[Response] = state.get("last_responses", [])
    if not responses: return 0.0
    select_actions = [a for a in actions if a.type == "select_response"]
    if not select_actions: return 0.0

    last_select = select_actions[-1]
    idx = last_select.selected_index
    if idx is None or idx >= len(responses): return 0.0

    selected_relevance = responses[idx].relevance
    best_relevance = max(r.relevance for r in responses)

    if best_relevance == 0.0: return 1.0 if selected_relevance == 0.0 else 0.0
    
    # Selection credit: must select the best AND the best must be at least 'decent'
    # Penalty for selecting low-relevance even if it's the 'best' among context
    selection_ratio = selected_relevance / best_relevance
    quality_factor = min(1.0, best_relevance / 0.5)
    
    return min(1.0, selection_ratio * quality_factor)

def _criterion_accurate_rating(actions: Sequence[Action], state: Dict) -> float:
    responses: List[Response] = state.get("last_responses", [])
    rate_actions = [a for a in actions if a.type == "rate_response"]
    if not rate_actions or not responses: return 0.0

    best_relevance = max(r.relevance for r in responses)
    last_score = rate_actions[-1].score
    if last_score is None: return 0.0
    
    error = abs(last_score - best_relevance)
    if error < 0.05: return 1.0
    if error < 0.12: return 0.8
    if error < 0.25: return 0.5
    return 0.0

def _criterion_multi_turn_reasoning(actions: Sequence[Action], state: Dict) -> float:
    # Requires comprehensive workflow matching the task's complexity
    queries = [a for a in actions if a.type in ("submit_query", "refine_query")]
    return 1.0 if len(queries) >= 1 and len(actions) >= 5 else 0.0

def _criterion_refine_query_used(actions: Sequence[Action], state: Dict) -> float:
    return 1.0 if any(a.type == "refine_query" for a in actions) else 0.0

def _criterion_plan_task_used(actions: Sequence[Action], state: Dict) -> float:
    return 1.0 if any(a.type == "plan_task" for a in actions) else 0.0

def _criterion_compare_responses_used(actions: Sequence[Action], state: Dict) -> float:
    return 1.0 if any(a.type == "compare_responses" for a in actions) else 0.0

def _criterion_summarize_used(actions: Sequence[Action], state: Dict) -> float:
    return 1.0 if any(a.type == "summarize" for a in actions) else 0.0

def _criterion_action_diversity(actions: Sequence[Action], _state: Dict) -> float:
    used_types = {a.type for a in actions}
    # For hard task, expect at least 5 action types (submit, plan, refine, rate, select)
    return min(1.0, len(used_types) / 5.0)

# ---------------------------------------------------------------------------
# Weighted score aggregation
# ---------------------------------------------------------------------------

def _weighted_score(criteria: List[Tuple[str, float, float]], env_state: Dict | None = None) -> float:
    total_weight = sum(w for _, w, s in criteria)
    if total_weight <= 0.0: return 0.01
    
    raw_score = sum(w * s for name, w, s in criteria) / total_weight
    # Uncapped score to allow for perfect 100/100 results.
    final_score = max(0.01, min(1.0, raw_score))
    
    # Efficiency bonus for high-speed performance
    # Reward finishing well under the turn limit.
    turns = env_state.get("turns_used", 0) if env_state else 0
    max_turns = env_state.get("max_turns", 10) if env_state else 10
    efficiency = max(0, 1.0 - (turns / max_turns))
    efficiency_bonus = efficiency * 0.05
    
    return max(0.01, min(1.0, final_score + efficiency_bonus))

# ---------------------------------------------------------------------------
# Public grading functions
# ---------------------------------------------------------------------------

def grade_easy(actions: Sequence[Action], env_state: Dict) -> float:
    criteria = [
        ("submitted_query", _EASY_WEIGHTS["submitted_query"], _criterion_submitted_query(actions, env_state)),
        ("positive_total_reward", _EASY_WEIGHTS["positive_total_reward"], _criterion_positive_total_reward(actions, env_state)),
        ("selected_best_response", _EASY_WEIGHTS["selected_best_response"], _criterion_selected_best_response(actions, env_state)),
        ("accurate_rating", _EASY_WEIGHTS["accurate_rating"], _criterion_accurate_rating(actions, env_state)),
    ]
    return _weighted_score(criteria, env_state)

def grade_medium(actions: Sequence[Action], env_state: Dict) -> float:
    criteria = [
        ("submitted_query", _MEDIUM_WEIGHTS["submitted_query"], _criterion_submitted_query(actions, env_state)),
        ("plan_task_used", _MEDIUM_WEIGHTS["plan_task_used"], _criterion_plan_task_used(actions, env_state)),
        ("multi_turn_reasoning", _MEDIUM_WEIGHTS["multi_turn_reasoning"], _criterion_multi_turn_reasoning(actions, env_state)),
        ("compare_responses_used", _MEDIUM_WEIGHTS["compare_responses_used"], _criterion_compare_responses_used(actions, env_state)),
        ("selected_best_response", _MEDIUM_WEIGHTS["selected_best_response"], _criterion_selected_best_response(actions, env_state)),
        ("accurate_rating", _MEDIUM_WEIGHTS["accurate_rating"], _criterion_accurate_rating(actions, env_state)),
        ("positive_total_reward", _MEDIUM_WEIGHTS["positive_total_reward"], _criterion_positive_total_reward(actions, env_state)),
    ]
    return _weighted_score(criteria, env_state)

def grade_hard(actions: Sequence[Action], env_state: Dict) -> float:
    criteria = [
        ("submitted_query", _HARD_WEIGHTS["submitted_query"], _criterion_submitted_query(actions, env_state)),
        ("plan_task_used", _HARD_WEIGHTS["plan_task_used"], _criterion_plan_task_used(actions, env_state)),
        ("refine_query_used", _HARD_WEIGHTS["refine_query_used"], _criterion_refine_query_used(actions, env_state)),
        ("summarize_used", _HARD_WEIGHTS["summarize_used"], _criterion_summarize_used(actions, env_state)),
        ("selected_best_response", _HARD_WEIGHTS["selected_best_response"], _criterion_selected_best_response(actions, env_state)),
        ("accurate_rating", _HARD_WEIGHTS["accurate_rating"], _criterion_accurate_rating(actions, env_state)),
        ("positive_total_reward", _HARD_WEIGHTS["positive_total_reward"], _criterion_positive_total_reward(actions, env_state)),
        ("action_diversity", _HARD_WEIGHTS["action_diversity"], _criterion_action_diversity(actions, env_state)),
    ]
    return _weighted_score(criteria, env_state)

GRADERS: Dict[str, GraderFn] = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}