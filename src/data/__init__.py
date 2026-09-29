"""Forward instances: the sustainable-diet problem and the synthetic generator."""

from src.data.diet import get_diet_instances, original_problem
from src.data.synthetic import (
    generate_synthetic_instances,
    solve_forward_problem,
    to_diet_tuple,
)

__all__ = [
    "get_diet_instances",
    "original_problem",
    "generate_synthetic_instances",
    "solve_forward_problem",
    "to_diet_tuple",
]
