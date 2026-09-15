import sympy as sy

from symode.componentwise_expression_factory import (
    create_parametrized_polynomial,
    get_coefficients_of_trigonometric_expression,
    get_polynomial_coefficients,
)


def test_create_parametrized_polynomial():
    x, y = sy.symbols("x y")

    polynomial, coefficients = create_parametrized_polynomial(2, [x, y])

    assert polynomial == (
        coefficients[0]
        + coefficients[1] * y
        + coefficients[2] * y**2
        + coefficients[3] * x
        + coefficients[4] * x * y
        + coefficients[5] * x**2
    )
    assert coefficients == [
        sy.Symbol("a_0_0"),
        sy.Symbol("a_0_1"),
        sy.Symbol("a_0_2"),
        sy.Symbol("a_1_0"),
        sy.Symbol("a_1_1"),
        sy.Symbol("a_2_0"),
    ]


def test_get_polynomial_coefficients_from_equation():
    x, y, a, b = sy.symbols("x y a b")

    coefficients = get_polynomial_coefficients(a * x**2 + b * y - 1, [x, y])

    assert coefficients == {x**2: a, y: b, 1: -1}


def test_get_polynomial_coefficients_from_equation_with_addition():
    x, y, a, b = sy.symbols("x y a b")

    coefficients = get_polynomial_coefficients(a * x**2 + b * x**2 - 1, [x, y])

    assert coefficients == {x**2: a + b, 1: -1}


def test_get_polynomial_coefficients_can_be_applied_to_multiple_equations():
    x, y, a = sy.symbols("x y a")

    coefficients = [
        get_polynomial_coefficients(equation, [x, y]) for equation in [a * x + 3, y**2]
    ]

    assert coefficients == [{x: a, 1: 3}, {y**2: 1}]


def test_get_coefficients_of_trigonometric_expression():
    x = sy.symbols("x", real=True)

    coefficients = get_coefficients_of_trigonometric_expression(
        sy.cos(x) + 2 * sy.sin(x) + 3, x, order_of_trigonometrics=1
    )

    assert coefficients == [sy.Rational(1, 2), -1, 3]
