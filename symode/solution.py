import sympy as sy


class Solution:
    """Store and incrementally update symbolic parameter values."""

    def __init__(self, values: dict[sy.Symbol, sy.Expr] | None = None) -> None:
        self.values = {} if values is None else values

    def update(
        self, new_solution_part: dict[sy.Symbol, sy.Expr]
    ) -> dict[sy.Symbol, sy.Expr]:
        """Update the stored solution with new symbolic values."""
        for parameter, value in self.values.items():
            self.values[parameter] = value.subs(new_solution_part).cancel()

        self.values.update(new_solution_part)
        return self.values
