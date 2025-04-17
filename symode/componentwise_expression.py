"""This module contains the class ComponentwiseExpression, which is used to represent an expression
as a sum of components."""

import sympy as sy
from sympy.simplify.fu import TR10


def get_coefficients_of_trigonometric_expression(
    equation: sy.Expr, variable: sy.Symbol, order_of_trigonometrics: int
):
    """returns the coefficients of an expression with sin and cos"""
    # transform products of sin and cos to sums of sin and cos
    equation = TR10(equation)

    # replace sin/cos terms by exponential of dummy variable
    exp_dummy = sy.symbols("exp_dummy")
    equation = equation.replace(sy.cos(variable), (exp_dummy + exp_dummy**-1) / 2)
    equation = equation.replace(
        sy.sin(variable), (exp_dummy - exp_dummy**-1) / (2 * sy.I)
    )

    # collect coefficients
    real_coefficients = []
    complex_coefficients = sy.Poly(
        equation * exp_dummy**order_of_trigonometrics, exp_dummy
    ).all_coeffs()

    for i in range(order_of_trigonometrics):
        real_coefficients.append(sy.re(complex_coefficients[i]))
        real_coefficients.append(sy.im(complex_coefficients[i]))

    real_coefficients.append(sy.re(complex_coefficients[order_of_trigonometrics]))
    return real_coefficients


def get_coefficients_of_polynomial_expression(
    polynomial: sy.Expr, variable: sy.Expr, carry: sy.Expr
) -> dict[sy.Expr, sy.Expr]:
    """returns the coefficients of a polynomial"""

    coefficients = sy.Poly(polynomial, variable).all_coeffs()
    maximum_power = len(coefficients)
    return {
        carry * variable ** (maximum_power - power - 1): coeff
        for power, coeff in enumerate(coefficients)
    }


class ComponentwiseExpression:
    """
    A class that represents and manipulates SymPy expressions as a sum of components.
    Each component consist of a basis expression and a coefficient.
    """

    def __init__(self, expression: sy.Expr) -> None:
        self._expression = {1: expression}

    def split(self, split_term: sy.Expr) -> None:
        """splits the components based on the split term"""
        new_components = {}
        for component, term in self._expression.items():
            new_components.update(
                {
                    **get_coefficients_of_polynomial_expression(
                        term, split_term, component
                    )
                }
            )
        self._expression = new_components

    def prune(self) -> None:
        """removes all components that have a coefficient of zero"""
        self._expression = {
            component: term
            for component, term in self._expression.items()
            if term != 0 and component != 0
        }

    def sum_up(self) -> sy.Expr:
        """returns the full SymPy expression"""
        return sum([component * term for component, term in self._expression.items()])

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
