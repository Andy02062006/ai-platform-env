import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from env import AIPlatformEnv
from models import Action

app = FastAPI(title="AIPlatformEnv API")

# Initialize global environment
env = AIPlatformEnv(seed=42)

class ResetRequest(BaseModel):
    task_key: str
    seed: int = 42

@app.post("/reset")
def reset_env(req: ResetRequest):
    try:
        obs, info = env.reset(req.task_key, seed=req.seed)
        return {"observation": obs, "info": info}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step_env(action: Action):
    try:
        obs, reward, term, trunc, info = env.step(action)
        return {
            "observation": obs,
            "reward": reward,
            "terminated": term,
            "truncated": trunc,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def get_state():
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
