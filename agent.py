"""
LLM Agent in a Virtual World
==============================
A 2D grid world where a Llama-powered agent must:
  1. Navigate to pick up a KEY
  2. Unlock a DOOR
  3. Reach the GOAL (star)

Run:
    python agent.py
"""


import json
import time
import ollama # type: ignore


# ── World symbols ────────────────────────────────────────────────────────────
EMPTY  = "."
WALL   = "W"
KEY    = "K"
DOOR   = "D"
GOAL   = "G"
AGENT  = "A"

ICONS = {EMPTY: ".", WALL: "▓", KEY: "K", DOOR: "D", GOAL: "★", AGENT: "@"}

# ── Initial world layout (6 rows × 7 cols) ───────────────────────────────────
INITIAL_MAP = [
    [".", ".", ".", ".", ".", ".", "G"],
    [".", "W", "W", ".", "W", "W", "."],
    [".", "W", ".", ".", ".", ".", "D"],
    [".", "W", ".", "W", "W", "W", "."],
    [".", ".", "K", ".", ".", ".", "."],
    ["A", ".", ".", ".", "W", ".", "."],
]

ROWS = len(INITIAL_MAP)
COLS = len(INITIAL_MAP[0])

# ── Action → (delta_row, delta_col) ─────────────────────────────────────────
ACTIONS = {
    "move_north": (-1,  0),
    "move_south": ( 1,  0),
    "move_east":  ( 0,  1),
    "move_west":  ( 0, -1),
}

MAX_STEPS = 40


# ─────────────────────────────────────────────────────────────────────────────
#  World class
# ─────────────────────────────────────────────────────────────────────────────
class World:
    def __init__(self):
        self.reset()

    def reset(self):
        import copy
        self.grid = copy.deepcopy(INITIAL_MAP)
        # Find starting agent position
        for r in range(ROWS):
            for c in range(COLS):
                if self.grid[r][c] == AGENT:
                    self.agent = [r, c]
                    self.grid[r][c] = EMPTY   # agent tracked separately
        self.has_key = False
        self.steps   = 0
        self.done    = False
        self.win     = False

    # ── Rendering ────────────────────────────────────────────────────────────
    def render(self) -> str:
        lines = []
        border = "+" + "─" * (COLS * 2 - 1) + "+"
        lines.append(border)
        for r in range(ROWS):
            row_chars = []
            for c in range(COLS):
                if [r, c] == self.agent:
                    row_chars.append(ICONS[AGENT])
                else:
                    row_chars.append(ICONS.get(self.grid[r][c], "?"))
            lines.append("│" + " ".join(row_chars) + "│")
        lines.append(border)
        return "\n".join(lines)

    # ── Observation (what the LLM sees) ─────────────────────────────────────
    def get_observation(self) -> dict:
        r, c = self.agent
        def cell(dr, dc):
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                return self.grid[nr][nc] or EMPTY
            return "wall"

        # Scan 3×3 neighbourhood
        nearby = []
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                nr, nc = r + dr, c + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    cell_val = self.grid[nr][nc]
                    if cell_val not in (EMPTY, WALL):
                        nearby.append(f"{cell_val} at row={nr} col={nc}")

        return {
            "position":      {"row": r, "col": c},
            "grid_size":     {"rows": ROWS, "cols": COLS},
            "has_key":       self.has_key,
            "adjacent": {
                "north": cell(-1,  0),
                "south": cell( 1,  0),
                "east":  cell( 0,  1),
                "west":  cell( 0, -1),
            },
            "nearby_objects": nearby if nearby else ["none"],
            "goal_description": (
                "Reach GOAL (G) at row=0 col=6. "
                "First pick up KEY (K). Then walk through DOOR (D) — "
                "you can only pass the door if you have the key. "
                "You cannot walk into WALL (W) cells."
            ),
            "available_actions": list(ACTIONS.keys()),
            "step": self.steps,
        }

    # ── Apply action ─────────────────────────────────────────────────────────
    def step(self, action: str) -> tuple[bool, str]:
        """Returns (success, message)."""
        if action not in ACTIONS:
            return False, f"Unknown action '{action}'"

        dr, dc = ACTIONS[action]
        r, c   = self.agent
        nr, nc = r + dr, c + dc

        if not (0 <= nr < ROWS and 0 <= nc < COLS):
            return False, "Hit the boundary wall"

        target = self.grid[nr][nc]

        if target == WALL:
            return False, "Blocked by WALL"

        if target == DOOR:
            if not self.has_key:
                return False, "DOOR is locked — need the KEY first"
            self.grid[nr][nc] = EMPTY
            self.agent = [nr, nc]
            return True, "Unlocked and passed through DOOR"

        if target == KEY:
            self.has_key = True
            self.grid[nr][nc] = EMPTY
            self.agent = [nr, nc]
            return True, "Picked up KEY"

        if target == GOAL:
            self.agent = [nr, nc]
            self.done  = True
            self.win   = True
            return True, "★ GOAL REACHED — task complete!"

        # Empty cell
        self.agent = [nr, nc]
        return True, f"Moved {action.replace('move_', '')}"


# ─────────────────────────────────────────────────────────────────────────────
#  llama agent harness
# ─────────────────────────────────────────────────────────────────────────────
def ask_llama(observation: dict) -> dict:
        """
        Send observation to local Ollama model.
        Returns JSON action + reasoning.
        """

        system = (
            "You are an autonomous agent navigating a 2D grid world. "
            "Reply ONLY with valid JSON in this exact format:\n"
            '{"action":"move_north","reasoning":"example"}\n'
            "No markdown. No backticks. No extra text."
        )

        prompt = (
            system
            + "\n\nCurrent observation:\n"
            + json.dumps(observation, indent=2)
        )

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        raw = response["message"]["content"].strip()

        raw = raw.replace("```json", "").replace("```", "").strip()

        return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────────────────────────────────────
def run():

    world  = World()

    print("\n" + "═" * 40)
    print("  LLM AGENT — VIRTUAL WORLD")
    print("  @ = Agent  K = Key  D = Door  ★ = Goal  ▓ = Wall")
    print("═" * 40)

    log = []   # full run log for saving

    for step in range(1, MAX_STEPS + 1):
        world.steps = step

        # ── Render ────────────────────────────────────────────────────────
        print(f"\n{'─'*40}")
        print(f"  STEP {step}  |  Key: {'✓' if world.has_key else '✗'}")
        print(world.render())

        # ── Observe ───────────────────────────────────────────────────────
        obs = world.get_observation()

        # ── Ask Llama ───────────────────────────────────────────────────
        print(" Asking Llama...")
        try:
            result = result = ask_llama(obs)
        except Exception as e:
            print(f"  Llama API error: {e}")
            break

        action    = result.get("action", "")
        reasoning = result.get("reasoning", "")
        print(f"   Action:    {action}")
        print(f"   Reasoning: {reasoning}")

        # ── Apply action ──────────────────────────────────────────────────
        success, msg = world.step(action)
        print(f"  Result:    {msg}")

        log.append({
            "step":      step,
            "action":    action,
            "reasoning": reasoning,
            "success":   success,
            "message":   msg,
            "has_key":   world.has_key,
            "position":  world.agent[:],
        })

        if world.done:
            print(f"\n{'═'*40}")
            print("  ★  AGENT COMPLETED THE TASK!")
            print(f"  Finished in {step} steps.")
            print("═" * 40)
            break

        time.sleep(0.5)   # small delay so you can follow along

    else:
        print(f"\n⏱ Agent did not complete the task within {MAX_STEPS} steps.")

    # ── Save log ─────────────────────────────────────────────────────────
    with open("run_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print("Full run log saved to run_log.json")


if __name__ == "__main__":
    run()
