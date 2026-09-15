"""This module contains utility functions."""

import itertools

import sympy as sy
from sympy.simplify.fu import TR10

from symode.componentwise_expression import ComponentwiseExpression


def get_coefficients_of_polynomial_expression(
    polynomial: sy.Expr, variable: sy.Expr, carry: sy.Expr
) -> dict[sy.Expr, sy.Expr]:
    """Return the coefficients of a polynomial."""
    coefficients = sy.Poly(polynomial, variable).all_coeffs()
    maximum_power = len(coefficients)
    return {
        carry * variable ** (maximum_power - power - 1): coefficient
        for power, coefficient in enumerate(coefficients)
    }


def create_componentwise_expression(
    expression: sy.Expr, variables: list[sy.Symbol]
) -> ComponentwiseExpression:
    """Create components by expanding ``expression`` in ``variables``."""
    components = {sy.Integer(1): expression}
    for variable in variables:
        new_components = {}
        for component, term in components.items():
            new_components.update(
                get_coefficients_of_polynomial_expression(term, variable, component)
            )
        components = new_components

    result = ComponentwiseExpression(components)
    result.prune()
    return result


def create_parametrized_polynomial(
    degree: int, variables: list[sy.Symbol]
) -> tuple[sy.Expr, list[sy.Symbol]]:
    """Create a polynomial ansatz of the given total degree."""
    exponent_tuples = [
        exponents
        for exponents in itertools.product(range(degree + 1), repeat=len(variables))
        if sum(exponents) <= degree
    ]
    coefficient_symbols = {
        exponents: sy.Symbol("a_" + "_".join(map(str, exponents)))
        for exponents in exponent_tuples
    }
    polynomial = sum(
        coefficient_symbols[exponents]
        * sy.prod(
            variable**exponent for variable, exponent in zip(variables, exponents)
        )
        for exponents in exponent_tuples
    )

    return polynomial, list(coefficient_symbols.values())


def get_polynomial_coefficients(
    equation: sy.Expr, variables: list[sy.Symbol]
) -> dict[sy.Expr, sy.Expr]:
    """Return the coefficients grouped by their associated monomial."""
    coefficients = {}
    polynomial = sy.Poly(equation, *variables)
    for monomial, coefficient in polynomial.terms():
        monomial_expression = sy.prod(
            variable**power for variable, power in zip(variables, monomial)
        )
        coefficients[monomial_expression] = coefficient

    return coefficients


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
