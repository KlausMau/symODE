import sympy as sy

from symode.componentwise_expression_factory import (
    create_parametrized_polynomial,
    get_coefficients_of_trigonometric_expression,
    get_polynomial_coefficients,
)


def test_create_parametrized_polynomial():
    x, y = sy.symbols("x y")

    polynomial = create_parametrized_polynomial(2, [x, y])

    assert polynomial.get_components() == {
        sy.Integer(1): sy.Symbol("a_0_0"),
        y: sy.Symbol("a_0_1"),
        y**2: sy.Symbol("a_0_2"),
        x: sy.Symbol("a_1_0"),
        x * y: sy.Symbol("a_1_1"),
        x**2: sy.Symbol("a_2_0"),
    }


def test_get_polynomial_coefficients_from_equation():
    x, y, a, b = sy.symbols("x y a b")

    coefficients = get_polynomial_coefficients(a * x**2 + b * y - 1, [x, y])

    assert coefficients.get_components() == {x**2: a, y: b, 1: -1}


def test_get_polynomial_coefficients_from_equation_with_addition():
    x, y, a, b = sy.symbols("x y a b")

    coefficients = get_polynomial_coefficients(a * x**2 + b * x**2 - 1, [x, y])

    assert coefficients.get_components() == {x**2: a + b, 1: -1}


def test_get_polynomial_coefficients_can_be_applied_to_multiple_equations():
    x, y, a = sy.symbols("x y a")

    coefficients = get_polynomial_coefficients(a * x + 3, [x, y])

    assert coefficients.get_components() == {x: a, 1: 3}


def test_get_coefficients_of_trigonometric_expression():
    x = sy.symbols("x", real=True)

    coefficients = get_coefficients_of_trigonometric_expression(
        sy.cos(x) + 2 * sy.sin(x) + 3, x, order_of_trigonometrics=1
    )

    assert coefficients == [sy.Rational(1, 2), -1, 3]
