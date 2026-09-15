import sympy as sy


class Solution:
    """Store and incrementally update symbolic parameter values."""

    def __init__(self, values: dict[sy.Symbol, sy.Expr] | None = None) -> None:
        self.values = {} if values is None else values

    def get(self) -> dict[sy.Symbol, sy.Expr]:
        return self.values

    def update(self, new_solution_part: dict[sy.Symbol, sy.Expr]) -> None:
        """Update the stored solution with new symbolic values."""
        for parameter, value in self.values.items():
            value = sy.sympify(value)
            self.values[parameter] = sy.cancel(value.subs(new_solution_part))

        self.values.update(new_solution_part)
