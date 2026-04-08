import os
import time
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from models import Action
from env import AIPlatformEnv

load_dotenv()

def run_inference(difficulty: str, seed: int = 42):
    print(f"[START] Inference wrapper for {difficulty}")
    
    # use OpenAI client with Mistral API base as requested
    client = OpenAI(
        api_key=os.environ.get("MISTRAL_API_KEY", "sk-mock-key"),
        base_url="https://api.mistral.ai/v1"
    )
    
    # Initialize env
    env = AIPlatformEnv(seed=seed)
    obs, info = env.reset(difficulty)
    
    print(f"[STEP] Environment initialized. Env ID: {info['task_key']}")
    
    # Provide appropriate queries based on difficulty
    if difficulty == "easy":
        prompts = ["What is the capital of France?"]
    elif difficulty == "medium":
        prompts = ["Summarize the French Revolution.", "Detail its main causes."]
    else:
        prompts = ["Write binary_search in Python.", "Optimize the function."]
        
    for i, p in enumerate(prompts):
        print(f"[STEP] Submitting query: {p}")
        action = Action(type="submit_query", query=p)
        obs, reward, term, trunc, info = env.step(action)
        print(f"[STEP] Query sent. Reward: {reward.value:.3f}")
        
        if difficulty == "medium" and i == 0:
            act_refine = Action(type="refine_query", query="Make it more detailed.")
            obs, reward, term, trunc, info = env.step(act_refine)
            print(f"[STEP] Refined query. Reward: {reward.value:.3f}")
            
        if difficulty == "hard" and i == 0:
            act_plan = Action(type="plan_task", plan="Break down the problem.")
            obs, reward, term, trunc, info = env.step(act_plan)
            print(f"[STEP] Planned task. Reward: {reward.value:.3f}")

    if obs.responses:
        # Rate last response
        act_rate = Action(type="rate_response", score=0.85)
        obs, reward, term, trunc, info = env.step(act_rate)
        print(f"[STEP] Rated response. Reward: {reward.value:.3f}")
        
        # Select best response
        act_sel = Action(type="select_response", selected_index=0)
        obs, reward, term, trunc, info = env.step(act_sel)
        print(f"[STEP] Selected response. Reward: {reward.value:.3f}")

    final_score = sum([r for r in info.get("history", []) if isinstance(r, float)])  # Dummy aggregation
    print(f"[END] Inference loop complete for {difficulty}.")

if __name__ == "__main__":
    os.environ["MISTRAL_API_KEY"] = os.environ.get("MISTRAL_API_KEY", "sQYHOg1EOojOX1OiC5fWpGoZpT9BvMG6")
    for diff in ["easy", "medium", "hard"]:
        run_inference(diff)
