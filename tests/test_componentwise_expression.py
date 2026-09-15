import sympy as sy

from symode.componentwise_expression import ComponentwiseExpression


def test_componentwise_expression_get_free_symbols_from_values():
    x, y, a, b = sy.symbols("x y a b")
    expression = ComponentwiseExpression({x: a + b, y: a})

    assert expression.get_free_symbols() == {a, b}


def test_componentwise_expression_show_filters_by_operation_count(capsys):
    expression = ComponentwiseExpression({sy.Integer(1): sy.Integer(2)})

    expression.show(number_of_ops=0)

    assert capsys.readouterr().out == "1: 2\n"
