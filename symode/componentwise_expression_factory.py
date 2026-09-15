"""This module contains utility functions."""

import itertools

import sympy as sy


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
