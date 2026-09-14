import sympy as sy

from symode.componentwise_expression import (
    ComponentwiseExpression,
    get_coefficients_of_polynomial_expression,
    get_coefficients_of_trigonometric_expression,
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


def test_get_coefficients_of_trigonometric_expression():
    x = sy.symbols("x", real=True)

    coefficients = get_coefficients_of_trigonometric_expression(
        sy.cos(x) + 2 * sy.sin(x) + 3, x, order_of_trigonometrics=1
    )

    assert coefficients == [sy.Rational(1, 2), -1, 3]


def test_componentwise_expression_split_and_subs_preserve_expression():
    x, y = sy.symbols("x y")
    expression = ComponentwiseExpression(2 * x + 3)

    expression.split(x)
    expression.subs({x: y})

    assert sy.expand(expression.sum_up() - (2 * y + 3)) == 0


def test_componentwise_expression_prune_removes_zero_components():
    x = sy.symbols("x")
    expression = ComponentwiseExpression(0)

    expression.split(x)
    expression.prune()

    assert expression.sum_up() == 0


def test_componentwise_expression_drop_removes_component():
    x, y = sy.symbols("x y")
    expression = ComponentwiseExpression(x + y)
    expression.split(x)

    expression.drop(1)

    assert expression.get_components() == {x: 1}


def test_componentwise_expression_show_filters_by_operation_count(capsys):
    expression = ComponentwiseExpression(2)

    expression.show(number_of_ops=0)

    assert capsys.readouterr().out == "1: 2\n"
