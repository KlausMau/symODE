import itertools

import sympy as sy
from sympy.simplify.fu import TR10

from symode.componentwise_expression import ComponentwiseExpression


def create_componentwise_expression_with_monomial_bases_from_polynomial_expression(
    expression: sy.Expr, variables: list[sy.Symbol]
) -> ComponentwiseExpression:
    """Create components by expanding ``expression`` in ``variables``."""
    coefficients = {}
    polynomial = sy.Poly(expression, *variables)
    for monomial, coefficient in polynomial.terms():
        monomial_expression = sy.prod(
            variable**power for variable, power in zip(variables, monomial)
        )
        coefficients[monomial_expression] = coefficient

    return ComponentwiseExpression(coefficients)


def create_parametrized_polynomial(
    degree: int, variables: list[sy.Symbol]
) -> ComponentwiseExpression:
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

    return (
        create_componentwise_expression_with_monomial_bases_from_polynomial_expression(
            polynomial, variables
        )
    )


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
