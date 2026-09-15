import sympy as sy

from symode.componentwise_expression import ComponentwiseExpression
from symode.solution import Solution


def get_reduced_expression(
    expression: ComponentwiseExpression,
) -> tuple[ComponentwiseExpression, dict[sy.Symbol, sy.Expr]]:
    """Eliminate coefficients that occur as nonzero numeric multiples of symbols."""
    print(f"found {len(expression.get_components())} components")
    print("eliminating components with trivial coefficient ...")

    eliminated_coefficients = {}
    while True:
        new_eliminated_coefficients = {}
        keys_to_drop = []
        for monomial, coefficient in expression.get_components().items():
            literal, symbol = coefficient.as_coeff_Mul()
            if symbol.is_Symbol and literal.is_number and literal != 0:
                new_eliminated_coefficients[symbol] = 0
                keys_to_drop.append(monomial)

        if not new_eliminated_coefficients:
            print("no new components to eliminate found. Resuming ...")
            break

        print(f"eliminating {len(keys_to_drop)} components ...")
        for key in keys_to_drop:
            expression.drop(key)
        expression.subs(new_eliminated_coefficients)
        expression.prune()
        eliminated_coefficients.update(new_eliminated_coefficients)

    print(f"eliminated {len(eliminated_coefficients)} coefficients in total")
    return expression, eliminated_coefficients


def find_solution_of_equation_by_inserting_values(
    equation: sy.Expr,
    variable: sy.Symbol,
    value_parameter_list: dict[sy.Symbol, sy.Expr],
    show_process: bool = False,
):
    """returns the solution of the equation by inserting the given values"""
    solutions = Solution()

    for solvable_parameter, variable_value in value_parameter_list.items():
        sol = sy.solve(equation.subs({variable: variable_value}), solvable_parameter)

        if not sol:
            print(f"No solution found for {solvable_parameter}")
            return solutions.values

        solved_parameter_expression = sol[0]

        if show_process is True:
            print(f"{solvable_parameter}={solved_parameter_expression}")

        new_solution_part = {solvable_parameter: solved_parameter_expression.simplify()}

        # update the solutions
        solutions.update(new_solution_part)

        # update equation
        equation = equation.subs(new_solution_part)

    # check whether the solution is complete (complete = prints "0")
    if show_process is True:
        print("remaining terms of equation:")
        print(equation.simplify())

    return solutions.values
