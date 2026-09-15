import sympy as sy

from symode.componentwise_expression import ComponentwiseExpression
from symode.root_finder import (
    find_solution_of_equation_by_inserting_values,
    get_reduced_expression,
)


def test_get_reduced_expression_eliminates_numeric_symbol_coefficients():
    x, y, a, b = sy.symbols("x y a b")
    expression = ComponentwiseExpression({sy.Integer(1): sy.Integer(3), x: a, y: 2 * b})

    assert expression.get_components() == {1: 3, x: a, y: 2 * b}

    reduced_expression, solution = get_reduced_expression(expression)

    assert reduced_expression is expression
    assert solution == {a: 0, b: 0}
    assert reduced_expression.get_components() == {1: 3}


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
