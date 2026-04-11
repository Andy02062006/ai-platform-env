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

# Add current directory to path to ensure robust imports in all environments
sys.path.append(str(Path(__file__).parent.absolute()))

from openai import OpenAI
from models import Action
from env import AIPlatformEnv

# Environment configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
# Backward compatibility
API_KEY = HF_TOKEN

# Benchmark configuration
BENCHMARK = "AIPlatformEnv"
MAX_STEPS = 12
TEMPERATURE = 0.7
MAX_TOKENS = 300

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI research agent interacting with an AI Platform Environment.
    Your goal is to evaluate the platform's capabilities through various tasks.
    
    Available actions:
    1. submit_query(query: str): Send a question to the platform.
    2. refine_query(query: str): Improve the previous query results.
    3. plan_task(plan: str): Describe your strategy.
    4. compare_responses(): Compare the returned candidate responses.
    5. summarize(): Summarize the findings.
    6. rate_response(score: float): Provide a quality rating (0.0 to 1.0) for the last response.
    7. select_response(selected_index: int): Choose the best candidate response (0, 1, or 2).

    Respond ONLY with a JSON object representing the action:
    {"type": "submit_query", "query": "..."}
    {"type": "select_response", "selected_index": 0}
    ...and so on.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_model_action(client: OpenAI, history: List[str], current_obs: str) -> Action:
    history_block = "\n".join(history[-5:]) if history else "None"
    user_prompt = f"History:\n{history_block}\n\nCurrent Observation:\n{current_obs}\n\nChoose your next action (JSON):"
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"}
        )
        content = (completion.choices[0].message.content or "").strip()
        data = json.loads(content)
        return Action(**data)
    except Exception as exc:
        # Fallback action
        if "rate" not in str(history):
            return Action(type="submit_query", query="Tell me more about this topic.")
        return Action(type="select_response", selected_index=0)


def main():
    # Corrected loop with action tracking
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    for task_name in ["easy", "medium", "hard"]:
        env = AIPlatformEnv(seed=42)
        rewards: List[float] = []
        actions_taken = []
        steps_taken = 0
        
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
        
        try:
            obs, info = env.reset(task_name)
            
            for step in range(1, MAX_STEPS + 1):
                if env.is_done:
                    break

                obs_summary = f"Responses: {len(obs.responses)}, Previous steps: {len(actions_taken)}"
                action = get_model_action(client, [a.type for a in actions_taken], obs_summary)

                obs, reward, terminated, truncated, info = env.step(action)
                
                reward_val = reward.value
                done = terminated or truncated
                
                rewards.append(reward_val)
                actions_taken.append(action)
                steps_taken = step
                
                log_step(step=step, action=action.type, reward=reward_val, done=done, error=None)
                
                if done:
                    break

            # End of episode: run grader
            from tasks import GRADERS
            difficulty = info.get("difficulty", task_name)
            grader_fn = GRADERS.get(difficulty)
            
            final_score = 0.0
            if grader_fn:
                final_score = grader_fn(actions_taken, env.state())
            
            # Ensure strictly between 0 and 1
            final_score = max(0.01, min(0.99, final_score))
            success = final_score >= 0.5
            
            log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)
            
        except Exception as e:
            # print(f"[DEBUG] Task {task_name} failed: {e}")
            log_end(success=False, steps=steps_taken, score=0.01, rewards=rewards)

if __name__ == "__main__":
    main()
