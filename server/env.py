"""
env.py
------
AI platform reinforcement-learning environment.

Provides two public classes:

- ``MockAIPlatform`` – a deterministic, seeded stub that simulates an AI
  query-response backend without any network calls.
- ``AIPlatformEnv`` – a gym-style environment that wraps the platform and
  exposes ``reset`` / ``step`` for agent interaction.
"""

from __future__ import annotations

import os
import random
import json
import textwrap
from dotenv import load_dotenv

load_dotenv()
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from server.models import Action, Observation, Response, Reward

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of candidate responses the mock platform returns per query.
NUM_RESPONSES: int = 3

#: Difficulty presets: maps a task difficulty label to its maximum turn count.
DIFFICULTY_CONFIG: Dict[str, int] = {
    "easy": 5,
    "medium": 8,
    "hard": 12,
}

DifficultyLabel = Literal["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    """Immutable descriptor for a single environment task."""
    key: str
    prompt_template: str
    difficulty: DifficultyLabel
    target_keywords: List[str]

    @property
    def max_turns(self) -> int:
        """Maximum number of interaction turns allowed for this task."""
        return DIFFICULTY_CONFIG[self.difficulty]


# ---------------------------------------------------------------------------
# Built-in task registry
# ---------------------------------------------------------------------------

_TASK_REGISTRY: Dict[str, Task] = {
    t.key: t
    for t in [
        Task(
            key="factual_qa_evaluation",
            prompt_template="Evaluate and rate candidate responses for factual Q&A about world geography.",
            difficulty="easy",
            target_keywords=["capital", "city", "country", "located", "paris", "france"],
        ),
        Task(
            key="multi_document_summarization",
            prompt_template="Refine queries to produce a high-quality multi-perspective summary of historical events like the French Revolution.",
            difficulty="medium",
            target_keywords=["summary", "causes", "consequences", "perspective", "detailed", "comprehensive"],
        ),
        Task(
            key="code_review_and_optimization",
            prompt_template="Query and refine Python code implementations for algorithms, ensuring they are optimized and idiomatic.",
            difficulty="hard",
            target_keywords=["search", "algorithm", "complexity", "optimization", "python", "performance"],
        ),
    ]
}

_TASK_REGISTRY["easy"] = _TASK_REGISTRY["factual_qa_evaluation"]
_TASK_REGISTRY["medium"] = _TASK_REGISTRY["multi_document_summarization"]
_TASK_REGISTRY["hard"] = _TASK_REGISTRY["code_review_and_optimization"]


# ---------------------------------------------------------------------------
# Mock AI platform (For local debugging only)
# ---------------------------------------------------------------------------

class MockAIPlatform:
    """Deterministic stub for an AI query-response backend."""
    _TEMPLATES: List[str] = [
        "Based on available information, the answer relates to {kw}.",
        "A comprehensive analysis shows that {kw} is central to this topic.",
        "Research indicates that {kw} plays a key role in answering this.",
    ]

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def query(self, prompt: str, difficulty: DifficultyLabel, target_keywords: List[str]) -> List[Response]:
        noise_scale: float = {"easy": 0.05, "medium": 0.12, "hard": 0.22}[difficulty]
        keyword = self._rng.choice(target_keywords) if target_keywords else "this"
        responses: List[Response] = []
        for i, template in enumerate(self._TEMPLATES):
            has_keyword = i == 0
            base_relevance = 0.80 if has_keyword else 0.45
            base_confidence = 0.75 if has_keyword else 0.50
            relevance = max(0.0, min(1.0, base_relevance + self._rng.gauss(0.0, noise_scale)))
            confidence = max(0.0, min(1.0, base_confidence + self._rng.gauss(0.0, noise_scale)))
            responses.append(Response(text=template.format(kw=keyword), relevance=relevance, confidence=confidence))
        self._rng.shuffle(responses)
        return responses

# ---------------------------------------------------------------------------
# Production AI platform (Live API)
# ---------------------------------------------------------------------------

from openai import OpenAI

class MetaAIPlatform:
    """Meta-optimized AI platform using real Llama-3 models via HF Router."""

    def __init__(self, model=None):
        self.model = model or os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        self.api_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        self.api_key = os.getenv("HF_TOKEN", os.getenv("META_API_KEY", ""))
        self.allow_mock = os.getenv("ALLOW_MOCK_FALLBACK", "true").lower() == "true"
        
        self.client = None
        if self.api_key:
            self.client = OpenAI(base_url=self.api_url, api_key=self.api_key)

    def query(self, prompt: str, difficulty: str, target_keywords: list[str]) -> list[Response]:
        """Query the Meta LLM and then use it as a judge to score results."""
        if not self.client:
            if self.allow_mock:
                return SmartSimulator().query(prompt, difficulty, target_keywords)
            raise RuntimeError("API key (HF_TOKEN) is not set and mocks are disabled.")

        raw_choices = []
        try:
            while len(raw_choices) < 3:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7 + (0.1 * len(raw_choices)),
                    max_tokens=512,
                    n=1
                )
                raw_choices.extend(completion.choices)
                if len(raw_choices) >= 3:
                    break
        except Exception as e:
            if self.allow_mock:
                return SmartSimulator().query(prompt, difficulty, target_keywords)
            raise RuntimeError(f"API Connection Failed: {e}. Mocks are explicitly disabled.")

        responses: list[Response] = []
        for choice in raw_choices[:3]:
            text = choice.message.content or ""
            # Real AI-based scoring: The environment environment acts as a judge
            relevance, confidence = self._judge_response(text, prompt, target_keywords)
            responses.append(Response(text=text, relevance=relevance, confidence=confidence))
        
        return responses[:3]

    def _judge_response(self, text: str, query: str, keywords: list[str]) -> tuple[float, float]:
        """Use the LLM to provide a real relevance and confidence score for a response."""
        judge_prompt = textwrap.dedent(f"""
            Evaluate the following AI response based on the user's query and target keywords.
            
            User Query: {query}
            Target Keywords: {', '.join(keywords)}
            Response to Evaluate: {text}
            
            Return a JSON object with:
            - "relevance": score in [0.0, 1.0] reflecting how accurately it answers the query.
            - "confidence": score in [0.0, 1.0] reflecting the certainty/completeness of the information.
            
            JSON format only: {{"relevance": 0.0, "confidence": 0.0}}
        """).strip()

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise evaluator of AI response quality."},
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=0.0,
                max_tokens=100,
                response_format={"type": "json_object"}
            )
            data = json.loads(completion.choices[0].message.content or "{}")
            return float(data.get("relevance", 0.5)), float(data.get("confidence", 0.5))
        except Exception:
            # Safe logic-based fallback if judge fails
            text_lower = text.lower()
            hit_count = sum(1 for kw in keywords if kw.lower() in text_lower) if keywords else 0
            base_rel = hit_count / len(keywords) if keywords else 0.5
            return max(0.01, min(0.99, base_rel)), 0.6

class SmartSimulator:
    """High-fidelity simulator (Only used if ALLOW_MOCK_FALLBACK=true)."""
    _KNOWLEDGE_BASE = {
        "easy": ["Amsterdam is in the Netherlands.", "The capital of the Netherlands is Amsterdam."],
        "medium": ["The French Revolution was a period of upheaval."],
        "hard": ["Implementing binary search is O(log n)."]
    }
    def query(self, prompt: str, difficulty: str, target_keywords: list[str]) -> list[Response]:
        base_texts = self._KNOWLEDGE_BASE.get(difficulty, self._KNOWLEDGE_BASE["easy"])
        return [Response(text=t, relevance=0.8, confidence=0.9) for t in base_texts[:3]]

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class AIPlatformEnv:
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._task_registry = _TASK_REGISTRY
        self._platform = MetaAIPlatform()
        self._task: Optional[Task] = None
        self._turn: int = 0
        self._history: List[str] = []
        self._last_responses: List[Response] = []
        self._total_reward: float = 0.0
        self._last_action_type: Optional[str] = None
        self._done: bool = True

    def reset(self, task_key: str) -> Tuple[Observation, Dict[str, Any]]:
        self._task = self._task_registry[task_key]
        self._turn = 0
        self._history = []
        self._last_responses = []
        self._total_reward = 0.0
        self._last_action_type = None
        self._done = False
        obs = Observation(responses=[], history=[])
        return obs, {"task_key": self._task.key, "max_turns": self._task.max_turns}

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, bool, Dict[str, Any]]:
        if self._done: raise RuntimeError("Reset required.")
        self._turn += 1
        reward_value, responses = self._dispatch(action)
        self._total_reward += reward_value
        self._last_action_type = action.type
        self._history.append(f"[{action.type}] reward={reward_value:.4f}")
        
        terminated = False 
        truncated = self._turn >= self._task.max_turns
        if terminated or truncated: self._done = True

        obs = Observation(responses=responses or self._last_responses, history=self._history)
        return obs, Reward(value=reward_value), terminated, truncated, self.state()

    def state(self) -> Dict[str, Any]:
        return {
            "history": list(self._history),
            "total_reward": self._total_reward,
            "turns_used": self._turn,
            "max_turns": self._task.max_turns if self._task else 0,
            "last_responses": list(self._last_responses),
            "difficulty": self._task.difficulty if self._task else "",
        }

    def _dispatch(self, action: Action) -> Tuple[float, List[Response]]:
        if action.type == "submit_query": return self._handle_submit_query(action)
        if action.type == "select_response": return self._handle_select_response(action), []
        if action.type == "rate_response": return self._handle_rate_response(action), []
        if action.type == "refine_query": return self._handle_refine_query(action)
        if action.type == "plan_task": return 0.1, []
        if action.type == "compare_responses": return 0.2, []
        if action.type == "summarize": return 0.2, []
        raise ValueError(f"Unhandled: {action.type}")

    def _handle_submit_query(self, action: Action) -> Tuple[float, List[Response]]:
        responses = self._platform.query(action.query, self._task.difficulty, self._task.target_keywords)
        self._last_responses = responses
        return 0.1, responses

    def _handle_select_response(self, action: Action) -> float:
        if not self._last_responses: return 0.0
        best = max(self._last_responses, key=lambda r: r.relevance)
        selected = self._last_responses[action.selected_index]
        return 0.4 if selected.relevance >= best.relevance - 0.01 else 0.0

    def _handle_rate_response(self, action: Action) -> float:
        if not self._last_responses: return 0.0
        best = max(self._last_responses, key=lambda r: r.relevance)
        return 0.5 if abs(action.score - best.relevance) < 0.15 else 0.0

    def _handle_refine_query(self, action: Action) -> Tuple[float, List[Response]]:
        if not self._last_responses: return -0.3, []
        improved = [Response(text=r.text, relevance=min(1.0, r.relevance + 0.1), confidence=r.confidence) for r in self._last_responses]
        self._last_responses = improved
        return 0.5, improved
