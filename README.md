---
title: AI Platform Env
emoji: brain
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# AIPlatformEnv

> An [OpenEnv](https://github.com/openenv)-compatible benchmark where an AI agent learns to interact with an AI platform — submitting queries, selecting responses, and rating quality — across three task difficulties.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Environment API](#environment-api)
   - [Actions](#actions)
   - [Observations](#observations)
   - [Reward](#reward)
5. [Tasks](#tasks)
6. [Grading](#grading)
7. [Baseline Scores](#baseline-scores)
8. [Project Structure](#project-structure)
9. [Docker](#docker)
10. [Hugging Face Deployment](#hugging-face-deployment)
11. [Extending the Environment](#extending-the-environment)
12. [License](#license)

---

## Motivation

Evaluating how well an AI agent can *use* another AI system is an increasingly important benchmark class. AIPlatformEnv provides a minimal, reproducible testbed where an agent must:

- Formulate effective natural-language queries.
- Rank and select among noisy candidate responses.
- Self-assess response quality through calibrated ratings.

The three task difficulties probe complementary agent capabilities. This environment includes **Synergy Reward Shaping**, which awards bonuses for logical action sequences (e.g., planning before execution), and an **Interactive Lab** powered by Gradio for seamless human testing.

## Premium Features
- **Synergy Rewards**: Incentivizes advanced agent strategies by rewarding logical workflows.
- **Interactive Lab**: A full-featured Gradio UI (`app.py`) for manual environment exploration.
- **OpenEnv Spec Compliance**: 100% compliant with typed Pydantic models.

---

## Installation

**Prerequisites:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/your-org/ai-platform-env.git
cd ai-platform-env

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

`requirements.txt` includes:

```
pydantic>=2.0
openai>=1.0
anthropic>=0.20
```

---

## Quick Start

```bash
# Run the baseline agent across all three tasks
python baseline.py
```

Expected output (seed = 42):

```
==========================================================
  AIPlatformEnv — Baseline Agent
  Seed: 42 (fully reproducible)
==========================================================
  EASY      score = 0.9800
  MEDIUM    score = 0.7650
  HARD      score = 0.6500
----------------------------------------------------------
  OVERALL   score = 0.8000  (mean across 3 tasks)
==========================================================
```

To run a single task interactively:

```python
from env import AIPlatformEnv
from models import Action

env = AIPlatformEnv(seed=42)
obs = env.reset("easy")

obs, reward, done, info = env.step(
    Action(type="submit_query", query="What is the capital of France?")
)
print(obs.responses[0].text)   # "The capital of France is Paris."
print(reward.value)            # -0.05  (query cost)
```

---

## Environment API

### Actions

Actions are defined in `models.py` as a Pydantic model. Every action requires a `type` field; the remaining fields are conditional.

| Field | Type | Required when | Description |
|---|---|---|---|
| `type` | `"submit_query"` \| `"select_response"` \| `"rate_response"` | always | The action to perform |
| `query` | `str` | `type="submit_query"` | Natural-language query sent to the AI platform |
| `selected_index` | `int ≥ 0` | `type="select_response"` | Zero-based index of the chosen response |
| `score` | `float ∈ [0, 1]` | `type="rate_response"` | Agent's quality rating for the selected response |

```python
from models import Action

# Submit a query
Action(type="submit_query", query="Explain binary search in Python.")

# Select the first response
Action(type="select_response", selected_index=0)

# Rate it
Action(type="rate_response", score=0.92)
```

### Observations

Each `step()` call returns an `Observation`:

| Field | Type | Description |
|---|---|---|
| `responses` | `list[Response]` | Candidate responses from the AI platform for the latest query |
| `history` | `list[str]` | All queries submitted so far this episode, oldest first |

Each `Response` contains:

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Full response text |
| `relevance` | `float ∈ [0, 1]` | Platform-estimated relevance to the query |
| `confidence` | `float ∈ [0, 1]` | Platform's self-reported confidence |

### Reward

`step()` returns a `Reward(value: float)` after every action:

| Action type | Reward signal |
|---|---|
| `submit_query` | `−0.05` per query (conciseness incentive) |
| `select_response` | `+relevance` of the chosen response |
| `rate_response` | `1.0 − |agent_score − relevance|` (calibration bonus) |

---

## Tasks

### Easy — Single-Turn Factual Q&A

| Property | Value |
|---|---|
| Key | `"easy"` |
| Max turns | 1 |
| Objective | Ask for the capital of France, select and rate the best response |
| Primary skill | Query formulation, response selection |

### Medium — Multi-Step Summarisation

| Property | Value |
|---|---|
| Key | `"medium"` |
| Max turns | 3 |
| Objective | Build a comprehensive summary of the French Revolution through iterative queries |
| Primary skill | Query refinement, iterative reasoning |

### Hard — Code Generation / Debugging

| Property | Value |
|---|---|
| Key | `"hard"` |
| Max turns | 5 |
| Objective | Obtain a correct Python `binary_search(arr, target)` function that passes unit tests |
| Primary skill | Code-aware querying, self-correction, functional verification |

---

## Grading

Each task is graded by a dedicated function in `tasks.py`. All graders share the signature:

```python
grade(task_key: str, actions: list[Action], env_state: dict) -> float
```

Scores are deterministic floats in `[0.0, 1.0]`. Partial credit is awarded at every stage — a non-completing agent always scores above zero for genuine attempts.

### Easy grader (5 criteria × 0.20)

1. Query submitted
2. Query contains relevant keywords
3. Response selected
4. Selected response has high relevance (≥ 0.7)
5. Rating is well-calibrated (|score − relevance| ≤ 0.1)

### Medium grader (6 weighted criteria)

1. Query submitted (0.15)
2. Lexical diversity across queries (0.15)
3. Response selected (0.20)
4. Best response relevance ≥ 0.6 (0.20)
5. Rating calibration (0.15)
6. Used multiple turns (0.15)

### Hard grader (8 weighted criteria)

1. Query submitted (0.10)
2. Code-relevant keywords in queries (0.10)
3. Response selected (0.10)
4. Response contains `def binary_search(` (0.15)
5. Response passes structural checks — valid AST, `return`, loop (0.15)
6. Functional correctness — unit test pass rate (0.20)
7. Rating calibration (0.10)
8. Self-correction attempt (0.10)

---

## Baseline Scores

Scores below were produced by `baseline.py` with `seed=42` using the deterministic mock backend. Use these as a reproducible lower bound when benchmarking new agents.

| Task | Difficulty | Baseline score |
|---|---|---|
| Easy | 🟢 Easy | ~0.98 |
| Medium | 🟡 Medium | ~0.77 |
| Hard | 🔴 Hard | ~0.65 |
| **Overall** | mean | **~0.80** |

A random agent (no keywords, random selection, random rating) scores approximately 0.20 / 0.15 / 0.10 for easy / medium / hard respectively.

---

## Project Structure

```
ai-platform-env/
├── models.py          # Pydantic models: Action, Observation, Response, Reward
├── env.py             # AIPlatformEnv class + MockAIPlatform backend
├── tasks.py           # Grader functions: grade_easy, grade_medium, grade_hard
├── baseline.py        # Rule-based baseline agent
├── openenv.yaml       # OpenEnv metadata descriptor
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container definition
└── README.md          # This file
```

---

## Docker

Build and run the environment inside a container:

```bash
# Build the image
docker build -t aiplatformenv:0.1.0 .

# Run the baseline agent
docker run --rm aiplatformenv:0.1.0

# Override the command to run your own agent
docker run --rm aiplatformenv:0.1.0 python my_agent.py

# Pass API keys for a live backend (optional)
docker run --rm \
  -e OPENAI_API_KEY=sk-... \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  aiplatformenv:0.1.0 python my_agent.py
```

---

## Hugging Face Deployment

AIPlatformEnv can be hosted as a Hugging Face Space (Gradio or Docker SDK) for live, shareable evaluation.

### 1. Create a new Space

```bash
# Install the Hugging Face CLI
pip install huggingface_hub

huggingface-cli login
huggingface-cli repo create ai-platform-env --type space --space_sdk docker
```

### 2. Push the repository

```bash
git remote add hf https://huggingface.co/spaces/your-username/ai-platform-env
git push hf main
```

### 3. Add a `README.md` YAML front-matter block

Hugging Face Spaces require the following header at the top of `README.md`:

```yaml
---
title: AIPlatformEnv
emoji: 
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
```

### 4. Configure secrets

If you connect a live AI backend, add your API keys as **Space secrets** via the Hugging Face web UI (`Settings → Variables and secrets`) rather than committing them to the repository.

### 5. Access the live Space

Once the Docker build completes your Space will be live at:

```
https://huggingface.co/spaces/your-username/ai-platform-env
```

---

## Extending the Environment

### Swap in a real AI backend

Replace `MockAIPlatform` in `env.py` with any client that implements the same `query()` interface:

```python
class OpenAIPlatform:
    def query(self, prompt, difficulty, target_kw, n=3) -> list[Response]:
        import openai
        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            n=n,
        )
        return [
            Response(text=c.message.content, relevance=1.0, confidence=1.0)
            for c in completion.choices
        ]
```

Then pass it to the environment:

```python
env = AIPlatformEnv(platform=OpenAIPlatform())
```

### Add a new task

1. Add an entry to the `TASKS` dict in `env.py`.
2. Write a `grade_<name>` function in `tasks.py`.
3. Register it in the `GRADERS` dict in `tasks.py`.
4. Add a query bank entry in `baseline.py`.
5. Document it in `openenv.yaml` and this README.

---

## License

MIT © Your Name. See [LICENSE](LICENSE) for details.