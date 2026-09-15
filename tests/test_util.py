import sympy as sy

from symode.componentwise_expression import ComponentwiseExpression
from symode.dynamical_system import DynamicalSystem
from symode.util import (
    create_parametrized_polynomial,
    find_solution_of_equation_by_inserting_values,
    get_polynomial_coefficients,
    get_reduced_expression,
    get_remainder_with_complex_ansatz,
    get_remainder_with_exponential_ansatz,
    get_remainder_with_rational_ansatz,
    update_solution,
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


def test_get_reduced_expression_eliminates_numeric_symbol_coefficients():
    x, y, a, b = sy.symbols("x y a b")
    expression = ComponentwiseExpression(a * x + 2 * b * y + 3)
    expression.split(x)
    expression.split(y)
    expression.prune()

    assert expression.get_components() == {1: 3, x: a, y: 2 * b}

    reduced_expression, solution = get_reduced_expression(expression)

    assert reduced_expression is expression
    assert solution == {a: 0, b: 0}
    assert reduced_expression.get_components() == {1: 3}


def test_update_solution_substitutes_existing_values():
    a, b = sy.symbols("a b")
    solution = {a: b + 1}

    updated_solution = update_solution(solution, {b: 2})

    assert updated_solution == {a: 3, b: 2}
    assert updated_solution is solution


def test_find_solution_of_equation_by_inserting_values(capsys):
    a, x = sy.symbols("a x")

    solution = find_solution_of_equation_by_inserting_values(
        a * x - 4, x, {a: 2}, show_process=True
    )

    assert solution == {a: 2}
    assert "a=2" in capsys.readouterr().out


def test_find_solution_of_equation_returns_partial_solution_when_unsolved(capsys):
    a, x = sy.symbols("a x")

    solution = find_solution_of_equation_by_inserting_values(
        a + 1, x, {x: 2}, show_process=False
    )

    assert solution == {}
    assert "No solution found for x" in capsys.readouterr().out


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
