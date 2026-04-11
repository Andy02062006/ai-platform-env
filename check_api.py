import os
import sys
from pathlib import Path

# Add current directory to path
ROOT_DIR = Path(__file__).parent.absolute()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from server.env import AIPlatformEnv
from server.models import Action

def test_api_connectivity():
    print("--- AI Platform API Connectivity Test ---")
    
    # Load token from environment or .env
    token = os.getenv("HF_TOKEN", "")
    if not token:
        print("[!] HF_TOKEN not found in environment. This will likely trigger the fallback.")
    else:
        print(f"[*] HF_TOKEN found: {token[:5]}...{token[-4:]}")

    env = AIPlatformEnv()
    obs, info = env.reset("easy")
    
    print(f"[*] Task: {info['task_key']} ({info['difficulty']})")
    print("[*] Sending test query to platform...")
    
    action = Action(type="submit_query", query="What is the capital of France?")
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"[*] Received {len(obs.responses)} candidate responses.")
    
    is_live = True
    for i, resp in enumerate(obs.responses):
        print(f"\nResponse {i+1}:")
        print(f"  Text: {resp.text[:100]}...")
        print(f"  Relevance: {resp.relevance:.2f}")
        
        # Check for SmartSimulator signatures
        if "Amsterdam" in resp.text or "French Revolution" in resp.text or "binary_search" in resp.text:
            is_live = False

    if is_live:
        print("\n[SUCCESS] The environment is generating LIVE responses from the Meta/Llama API!")
    else:
        print("\n[INFO] The environment is currently using the Verified High-Fidelity Simulator.")
        print("       To enable live API calls, ensure HF_TOKEN is set in your environment or .env file.")

if __name__ == "__main__":
    test_api_connectivity()
