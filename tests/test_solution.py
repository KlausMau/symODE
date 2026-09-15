import sympy as sy

from symode.solution import Solution


def test_solution_get_returns_saved_values():
    a = sy.symbols("a")
    solution = Solution({a: 1})

    assert solution.get() == {a: 1}
    assert solution.get() is solution.values


def test_solution_update_substitutes_existing_values():
    a, b = sy.symbols("a b")
    solution = Solution({a: b + 1})

    solution.update({b: 2})

    assert solution.get() == {a: 3, b: 2}


def test_solution_update_accepts_integer_values():
    a, b = sy.symbols("a b")
    solution = Solution({a: 1})

    solution.update({b: 2})

    assert solution.get() == {a: 1, b: 2}


def test_solution_update_accepts_mixed_python_and_sympy_values():
    a, b, c = sy.symbols("a b c")
    solution = Solution({a: 1, b: c + 1})

    solution.update({c: 2})

    assert solution.get() == {a: 1, b: 3, c: 2}
