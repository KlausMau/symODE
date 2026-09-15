import sympy as sy

from symode.dynamical_system import DynamicalSystem
from symode.observable_expression_factory import (
    get_remainder_with_complex_ansatz,
    get_remainder_with_exponential_ansatz,
    get_remainder_with_rational_ansatz,
)


def make_linear_system():
    x, a = sy.symbols("x a")
    return DynamicalSystem({x: a * x}), x, a


def test_get_remainder_with_rational_ansatz():
    system, x, a = make_linear_system()
    ld = sy.symbols("ld")

    observable, remainder = get_remainder_with_rational_ansatz(system, x, 1, ld)

    assert observable == x
    assert sy.expand(remainder.sum_up() - (a - ld) * x) == 0


def test_get_remainder_with_exponential_ansatz():
    system, x, a = make_linear_system()
    exponent, ld = sy.symbols("exponent ld")

    observable, remainder = get_remainder_with_exponential_ansatz(
        system, x, exponent, ld
    )

    assert observable == x * sy.exp(exponent)
    assert sy.expand(remainder.sum_up() - (a - ld) * x) == 0


def test_get_remainder_with_complex_ansatz():
    system, x, a = make_linear_system()
    ld, beta = sy.symbols("ld beta")

    observable, remainder = get_remainder_with_complex_ansatz(system, x, 1, ld, beta)

    assert observable == x * sy.exp(beta * sy.log(1))
    assert sy.expand(remainder.sum_up() - (a - ld) * x) == 0
