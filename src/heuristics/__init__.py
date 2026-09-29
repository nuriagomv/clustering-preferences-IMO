"""Alternating heuristics used to warm-start the mixed-integer models.

Import the function you need from the submodule. Both modules define a
function named ``heuristic``; they are not interchangeable.

* :func:`src.heuristics.known_objectives.heuristic` — costs ``C`` are known.
* :func:`src.heuristics.unknown_objectives.heuristic` — costs are estimated,
  preferences come from a catalogue.
* :func:`src.heuristics.known_objectives.cluster_assignments` — nearest
  cluster under the duality gap. The experiment drivers use it at test time.
"""

from src.heuristics.known_objectives import cluster_assignments
from src.heuristics.known_objectives import heuristic as heuristic_known
from src.heuristics.unknown_objectives import heuristic as heuristic_unknown

__all__ = ["cluster_assignments", "heuristic_known", "heuristic_unknown"]
