import sympy as sy

from symode.componentwise_expression import ComponentwiseExpression
from symode.componentwise_expression_factory import (
    create_componentwise_expression_with_monomial_bases_from_polynomial_expression,
    get_coefficients_of_polynomial_expression,
)


def test_get_coefficients_of_polynomial_expression():
    x, carry = sy.symbols("x carry")

    coefficients = get_coefficients_of_polynomial_expression(
        3 * x**2 + 2 * x + 1, x, carry
    )

    assert coefficients == {
        carry * x**2: 3,
        carry * x: 2,
        carry: 1,
    }


def test_componentwise_expression_from_expression_and_subs_preserve_expression():
    x, y = sy.symbols("x y")
    expression = (
        create_componentwise_expression_with_monomial_bases_from_polynomial_expression(
            2 * x + 3, [x]
        )
    )

    expression.subs({x: y})

    assert sy.expand(expression.sum_up() - (2 * y + 3)) == 0


def test_componentwise_expression_prune_removes_zero_components():
    x = sy.symbols("x")
    expression = (
        create_componentwise_expression_with_monomial_bases_from_polynomial_expression(
            0, [x]
        )
    )

    assert expression.sum_up() == 0


def test_componentwise_expression_drop_removes_component():
    x, y = sy.symbols("x y")
    expression = (
        create_componentwise_expression_with_monomial_bases_from_polynomial_expression(
            x + y, [x]
        )
    )

    expression.drop(1)

    assert expression.get_components() == {x: 1}


def test_componentwise_expression_show_filters_by_operation_count(capsys):
    expression = ComponentwiseExpression({sy.Integer(1): sy.Integer(2)})

    expression.show(number_of_ops=0)

    assert capsys.readouterr().out == "1: 2\n"
