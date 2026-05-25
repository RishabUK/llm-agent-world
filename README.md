# LLM Agent in a Virtual World

A Llama-powered autonomous agent that navigates a 2D grid world, picks up a key, unlocks a door, and reaches a goal — entirely driven by an LLM reasoning loop.

```
+─────────────────────────────┐
│ . . . . . . ★ │   ★ = Goal
│ . ▓ ▓ . ▓ ▓ . │   @ = Agent
│ . ▓ . . . . D │   K = Key
│ . ▓ . ▓ ▓ ▓ . │   D = Door (locked)
│ . . K . . . . │   ▓ = Wall
│ @ . . . ▓ . . │
+─────────────────────────────┘
```

## How it works

Each step of the agent loop:

1. **Observe** — The world state is serialised into a structured JSON observation (position, adjacent cells, nearby objects, inventory)
2. **Reason** — The observation is sent to Llama with a system prompt instructing it to act as a grid-world agent
3. **Act** — Llama returns a JSON `{action, reasoning}` object; the action is applied to the world
4. **Repeat** — Until the goal is reached or max steps exceeded

```
World State
    │
    ▼
get_observation()   →   JSON observation dict
    │
    ▼
ask_Llama()        →   Llama API call (llama3.2)
    │
    ▼
{action, reasoning} →   world.step(action)
    │
    ▼
Updated world state  →  loop again
```

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/llm-agent-world.git
cd llm-agent-world
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the agent

```bash
python agent.py
```

You'll see the grid printed each step with Llama's chosen action and reasoning. A full log is saved to `run_log.json` when the run ends.

## Example output

```
════════════════════════════════════════
  LLM AGENT — VIRTUAL WORLD
  @ = Agent  K = Key  D = Door  ★ = Goal  ▓ = Wall
════════════════════════════════════════

────────────────────────────────────────
  STEP 1  |  Key: ✗
+─────────────────────────┐
│ . . . . . . ★ │
│ . ▓ ▓ . ▓ ▓ . │
│ . ▓ . . . . D │
│ . ▓ . ▓ ▓ ▓ . │
│ . . K . . . . │
│ @ . . . ▓ . . │
+─────────────────────────┘
  🧠 Asking Llama...
  → Action:    move_east
  → Reasoning: Moving east to begin navigating toward the key at row=4 col=2.
  → Result:    Moved east

... (steps 2–10) ...

  STEP 11  |  Key: ✓
  🧠 Asking Llama...
  → Action:    move_north
  → Reasoning: One final step north to reach the goal star.
  → Result:    ★ GOAL REACHED — task complete!

════════════════════════════════════════
  ★  AGENT COMPLETED THE TASK!
  Finished in 11 steps.
════════════════════════════════════════

📄 Full run log saved to run_log.json
```

See `run_log.json` for the full example output included in this repo.

## Project structure

```
llm-agent-world/
├── agent.py          # Main agent loop + world engine
├── requirements.txt  # Python dependencies (just ollama)
├── run_log.json      # Example run output
└── README.md
```

## Design choices

### Observation representation
The agent receives a structured JSON observation rather than a raw string. This gives Llama consistent, parseable data: exact position, per-direction adjacency, and a nearby-objects scan within a 3-cell radius. Structured data reduces hallucination — Llama doesn't have to infer its position from prose.

### Action space
Four cardinal movement actions (`move_north/south/east/west`). Small and unambiguous — Llama never has to choose between synonyms like "go north" vs "walk north". Actions encode interaction automatically: stepping onto a KEY picks it up, stepping onto a DOOR with the key unlocks it.

### Why JSON responses
Asking Llama to return `{"action": "...", "reasoning": "..."}` means the harness can parse deterministically and also log Llama's reasoning at every step. This makes debugging straightforward — you can read exactly why the agent made each decision.

### What worked
- Structured JSON observation → very reliable action selection
- Including the goal description in every observation prevents Llama forgetting the objective
- Scanning a 3×3 neighbourhood for nearby objects gives Llama enough spatial context without overwhelming the prompt

### What didn't work / future ideas
- No memory across steps (each call is stateless) — adding a short history of recent actions in the observation improves performance on longer mazes
- Llama occasionally tries to walk into a wall — adding a `blocked_directions` field to the observation reduces this
- The world is handcrafted; a procedural maze generator would make evaluation more rigorous
