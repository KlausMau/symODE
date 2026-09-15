import sympy as sy

from symode.solution import Solution


def test_solution_update_substitutes_existing_values():
    a, b = sy.symbols("a b")
    solution = Solution({a: b + 1})

    updated_solution = solution.update_solution({b: 2})

    assert updated_solution == {a: 3, b: 2}
    assert updated_solution is solution.values
