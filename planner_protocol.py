"""
Shared Planner interface for the puzzle repo.

Both AlphaZeroPusher (PUCT) and MOREPlanner implement this Protocol so the
eval harness can swap them without any structural changes.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Planner(Protocol):
    """
    Interface every planner in this repo satisfies.

    plan() and verify() are the only two required entry-points.
    All planners also accept an `env` constructor argument, but env
    construction is planner-specific so it is not part of this interface.
    """

    def plan(
        self,
        initial_state: dict | None = None,
        verbose: bool = True,
        pause_before_verify: bool = False,
    ) -> list[dict] | None:
        """
        Search for a plan from *initial_state* (or env.get_state(0) if None).

        Returns an ordered list of action dicts on success, or None if
        planning failed (no goal found or verification did not pass).
        """
        ...

    def verify(
        self,
        plan: list[dict],
        initial_state: dict,
        verbose: bool = True,
    ) -> tuple[int, float, float, bool]:
        """
        Re-execute *plan* from *initial_state* and return
        (successes, avg_reward, rate, passed).

        passed is True when rate >= planner's verify_threshold.
        """
        ...
