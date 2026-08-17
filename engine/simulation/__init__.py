"""Closed-loop TravelUAV simulation components.

``data`` owns trajectory semantics and coordinate transforms, ``runtime`` owns
AirSim/TravelUAV lifecycle and movement, and ``evaluator`` owns model rollout
and output/metric policy.  The package intentionally avoids eager imports so
data-only tooling does not require Torch or AirSim.
"""

__all__ = ["data", "runtime", "evaluator"]
