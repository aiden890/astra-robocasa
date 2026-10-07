"""Memory carried between turns (adapted from RoboDawn ``harness/agent/memory.py``, single arm).

* scratchpad   - free text the model rewrites every turn (what is done, what it learned, what is next);
* history      - machine-written log of the last ``history_turns`` turns: commands, outcomes, pose after;
* grasp fact   - fingertip height at which the gripper last closed on something (cleared on open).

Every call of ``codex exec`` is a fresh, stateless session, so this text is the only memory the model has.
"""

from __future__ import annotations

from dataclasses import dataclass, field

NOTE_LIMIT = 90


@dataclass
class TurnRecord:
    turn: int
    commands: list[str]
    outcomes: list[str]
    fingertip_cm: list[float]
    opening: float
    steps_used: int


@dataclass
class AgentMemory:
    history_turns: int = 12
    scratchpad: str = ""
    turns: list[TurnRecord] = field(default_factory=list)
    grasp: dict | None = None

    def update_scratchpad(self, text) -> None:
        text = str(text or "").strip()
        if text:
            self.scratchpad = text[:2500]

    def record_turn(self, turn: int, results: list[dict], state: dict, surface_cm: float | None = None) -> None:
        """Log one turn; ``state`` is the state after the turn, ``surface_cm`` the work surface seen before it."""
        commands, outcomes = [], []
        for r in results:
            commands.append(r["command"])
            note = (r.get("note") or "").strip()
            if len(note) > NOTE_LIMIT:
                note = note[: NOTE_LIMIT - 3] + "..."
            outcomes.append(("ok" if r.get("ok") else "FAILED") + (f" - {note}" if note else ""))
            if r.get("kind") == "gripper" and r.get("command") == "gripper close" and r.get("gripper_opening", 0) > 0.06:
                tip = r.get("state_after", state)["fingertip_cm"]
                self.grasp = {"height_cm": tip[2], "opening": r["gripper_opening"], "surface_cm": surface_cm}
            elif r.get("command") == "gripper open":
                self.grasp = None
        self.turns.append(TurnRecord(turn, commands, outcomes, list(state["fingertip_cm"]),
                                     float(state["gripper_opening"]), int(state["steps_used"])))

    def render(self) -> str:
        parts = ["YOUR NOTES FROM PREVIOUS TURNS (you wrote these; rewrite them in the `memory` field):\n"
                 + (self.scratchpad or "(empty - this is the first turn)")]
        recent = self.turns[-self.history_turns:]
        lines = []
        if len(self.turns) > len(recent):
            lines.append(f"... {len(self.turns) - len(recent)} earlier turns omitted ...")
        for t in recent:
            cmds = "; ".join(f"{c} -> {o}" for c, o in zip(t.commands, t.outcomes)) or "(no command)"
            f, l, h = t.fingertip_cm
            lines.append(f"turn {t.turn}: {cmds} | after: tip (fwd {f:.0f}, left {l:.0f}, height {h:.0f}), "
                         f"opening {t.opening:.2f}, steps used {t.steps_used}")
        parts.append("HISTORY OF YOUR COMMANDS AND THEIR OUTCOMES:\n" + ("\n".join(lines) if lines else "(none yet)"))
        if self.grasp:
            g = self.grasp
            fact = (f"GRASP FACT: the gripper closed on an object with the fingertips at height {g['height_cm']:.0f} cm "
                    f"(opening {g['opening']:.2f})")
            if g.get("surface_cm") is not None:
                fact += (f", while the work surface was at {g['surface_cm']:.0f} cm. To set the object down on a surface "
                         f"at height H, lower the fingertips to about {g['height_cm']:.0f} + (H - {g['surface_cm']:.0f}) "
                         "before opening")
            parts.append(fact + ".")
        return "\n\n".join(parts)

    def to_json(self) -> dict:
        return {"scratchpad": self.scratchpad, "grasp": self.grasp, "turns": [t.__dict__ for t in self.turns]}
