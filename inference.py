"""
inference.py
------------
Baseline inference script for AIPlatformEnv.
Compliant with OpenEnv structured logging requirements.
"""

import os
import json
import textwrap
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path to ensure 'server' package is discoverable
sys.path.append(str(Path(__file__).parent))

from openai import OpenAI
from server.models import Action
from server.env import AIPlatformEnv

# OpenEnv Compliance Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("META_API_KEY", ""))

# Benchmark configuration
BENCHMARK = "AIPlatformEnv"
MAX_STEPS = 12
TEMPERATURE = 0.7
MAX_TOKENS = 300

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI research assistant. Your goal is to maximize performance in AIPlatformEnv.
    The environment's grader rewards high-quality autonomous research workflows. 
    
    MANDATORY WORKFLOW:
    1. plan_task: ALWAYS start by describing your research plan.
    2. submit_query: Use descriptive queries with relevant keywords.
    3. refine_query: If initial responses are weak, refine your search to improve quality.
    4. compare_responses: Use this to analyze the trade-offs between candidates.
    5. summarize: Use this for medium/hard tasks to consolidate knowledge.
    6. rate_response: Provide a precise score in [0.0, 1.0] reflecting the best candidate's RELEVANCE.
    7. select_response: Finalize by selecting the index of the most accurate response.

    To get a high score, you must demonstrate "Action Diversity"—don't just skip to selection.
    Respond strictly in JSON format.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_model_action(client: OpenAI, history: List[Action], current_obs: str) -> Action:
    history_block = "\n".join([f"- {a.type}: {json.dumps(a.model_dump(exclude_none=True))}" for a in history[-5:]]) if history else "None"
    prompt = textwrap.dedent(f"""
        Analyze the current observation and history, then decide on the next strategic action.
        
        Available Actions:
        - plan_task: {{"type": "plan_task", "plan": "..."}}
        - submit_query: {{"type": "submit_query", "query": "..."}}
        - refine_query: {{"type": "refine_query", "query": "..."}}
        - compare_responses: {{"type": "compare_responses"}}
        - summarize: {{"type": "summarize"}}
        - rate_response: {{"type": "rate_response", "score": 0.0}}
        - select_response: {{"type": "select_response", "selected_index": 0}}
        
        Current History:
        {history_block}

        Current Observation:
        {current_obs}

        Return ONLY the JSON action object.
    """).strip()
    
    content = "INITIAL_STATE"
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"}
        )
        content = (completion.choices[0].message.content or "").strip()
        data = json.loads(content)
        
        # Heuristic fix for some common LLM nesting errors
        if "action" in data and isinstance(data["action"], dict):
            data = data["action"]
            
        return Action(**data)
    except Exception as e:
        print(f"[DEBUG] Model returned invalid action: {content}. Error: {e}")
        # Logical fallback instead of crashing
        if not history:
            return Action(type="plan_task", plan="I will start by analyzing the requirements.")
        return Action(type="rate_response", score=0.5)


def main():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN (or META_API_KEY) environment variable is not set.")
    
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for task_name in ["easy", "medium", "hard"]:
        try:
            env = AIPlatformEnv()
        except Exception as e:
            print(f"[DEBUG] Environment initialization failed: {e}")
            continue

        rewards: List[float] = []
        actions_taken = []
        steps_taken = 0
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
        
        try:
            obs, info = env.reset(task_name)
            terminated = False
            truncated = False

            while not (terminated or truncated) and steps_taken < MAX_STEPS:
                obs_data = {
                    "current_task": task_name,
                    "turn": steps_taken + 1,
                    "max_turns": MAX_STEPS,
                    "responses_count": len(obs.responses),
                    "responses": [{"index": i, "text": r.text} for i, r in enumerate(obs.responses)],
                    "history": obs.history
                }
                
                action = get_model_action(client, actions_taken, json.dumps(obs_data))
                obs, reward, terminated, truncated, info = env.step(action)
                
                rewards.append(reward.value)
                actions_taken.append(action)
                steps_taken += 1
                log_step(step=steps_taken, action=action.type, reward=reward.value, done=(terminated or truncated), error=None)
                
                if terminated or truncated:
                    break

            from server.tasks import GRADERS
            difficulty = info.get("difficulty", task_name)
            grader_fn = GRADERS.get(difficulty)
            final_score = grader_fn(actions_taken, env.state()) if grader_fn else 0.05
            
            final_score = max(0.01, min(0.99, final_score))
            log_end(success=(final_score >= 0.5), steps=steps_taken, score=final_score, rewards=rewards)
            
        except Exception as e:
            log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=str(e))
            log_end(success=False, steps=steps_taken, score=0.01, rewards=rewards)

if __name__ == "__main__":
    main()
