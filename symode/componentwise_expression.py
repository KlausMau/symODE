import sympy as sy


class ComponentwiseExpression:
    """
    A class that represents and manipulates SymPy expressions as a sum of components.
    Each component consist of a basis expression and a coefficient.
    """

    def __init__(self, components: dict[sy.Expr, sy.Expr]) -> None:
        self._expression = components.copy()

    def prune(self) -> None:
        """removes all components that have a coefficient of zero"""
        self._expression = {
            component: term
            for component, term in self._expression.items()
            if term != 0 and component != 0
        }

    def drop(self, key: sy.Expr) -> None:
        """Remove the component associated with ``key``."""
        self._expression.pop(key, None)

    def get_components(self) -> dict[sy.Expr, sy.Expr]:
        """Return the components as a monomial-to-coefficient mapping."""
        return self._expression.copy()

    def sum_up(self) -> sy.Expr:
        """returns the full SymPy expression"""
        return sum(component * term for component, term in self._expression.items())

    def subs(self, substitutions: dict[sy.Expr, sy.Expr]) -> None:
        """substitutes one expression for another in bases and coefficients"""
        self._expression = {
            component.subs(substitutions): term.subs(substitutions).cancel()
            for component, term in self._expression.items()
        }

    def show(self, number_of_ops=None) -> None:
        """
        displays the components of the expression.
        number_of_ops: maximum number of operations of the displayed components.
        Set to None to display all components
        """
        for component, term in self._expression.items():
            if number_of_ops is None:
                print(f"{component}: {term}")
            else:
                if sy.count_ops(term) <= number_of_ops:
                    print(f"{component}: {term}")
