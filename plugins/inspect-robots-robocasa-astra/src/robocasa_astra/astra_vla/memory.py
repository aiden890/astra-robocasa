"""VLA feedback memory, restored from the durable trace after host restart."""

from ..astra_robodawn.memory import AgentMemory


class ChunkMemory(AgentMemory):
    """The skill variant's memory; the grasp fact is taken from chunks that close the gripper on
    something."""

    def record_turn(
        self, turn: int, results: list[dict], state: dict, surface_cm: float | None = None
    ) -> None:
        """Retain measured chunk outcomes and grasp information between decisions."""
        super().record_turn(turn, results, state, surface_cm)
        for r in results:
            if r.get("kind") != "chunk":
                continue
            if r.get("gripper_closed") and r.get("gripper_opening", 0) > 0.06:
                if self.grasp is None:
                    self.grasp = {
                        "height_cm": state["fingertip_cm"][2],
                        "opening": r["gripper_opening"],
                        "surface_cm": surface_cm,
                    }
            elif not r.get("gripper_closed"):
                self.grasp = None
